from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from PIL import Image

    from matera.core.layout import PageLayout
    from matera.core.profile import FormProfile
    from matera.vision.contracts import AlignedPage

from matera.vision.detectors.global_topology import compute_global_topology_evidence
from matera.vision.detectors.local import compute_local_option_evidence
from matera.vision.diagnostics.global_shape import attach_global_shape_diagnostics
from matera.vision.evidence.models import (
    AlignmentEvidence,
    HsvFallbackEvidence,
    MarkThresholdEvidence,
    OptionMarkEvidence,
    PageMarkEvidence,
    QuestionMarkEvidence,
    ReferenceEvidence,
    RoiEvidence,
)


def extract_mark_evidence(
    aligned_page: AlignedPage,
    profile: FormProfile,
    layout: PageLayout,
    reference_image: Image.Image,
    *,
    debug_dir: str | None = None,
    debug_full_mask: np.ndarray | None = None,
    q14_vlm_runtime: Any | None = None,
    question_vlm_runtime: Any | None = None,
) -> PageMarkEvidence:

    from matera.vision.mark_hsv import (
        get_local_roi_crops_hsv,
        process_roi_hsv,
    )
    from matera.vision.mark_radial import (
        get_local_roi_crops,
        get_text_bounding_box,
    )
    from matera.vision.mark_thresholds import (
        BLANK_THRESHOLD_DEG,
        DIFF_THRESHOLD,
        GLOBAL_PAD,
        LOCAL_PAD,
        MARKED_THRESHOLD_DEG,
        MIN_INK_PER_BIN,
        NUM_BINS,
        OUTER_RADIUS,
    )

    median_ref_bgr = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
    orig_bgr = cv2.cvtColor(np.array(aligned_page.image), cv2.COLOR_RGB2BGR)

    if debug_full_mask is None:
        debug_full_mask = np.zeros_like(orig_bgr)

    strategy_map = {q.question_id: q.mark_strategy for q in profile.questions}

    scale_x = orig_bgr.shape[1] / layout.width_px
    scale_y = orig_bgr.shape[0] / layout.height_px

    # Save original bboxes before scaling for RoiEvidence.bbox
    original_bboxes: dict[tuple[str, str], dict[str, int]] = {}
    for roi in layout.rois:
        key = (roi.question_id, roi.option_id)
        original_bboxes[key] = {
            "x": roi.bbox.x,
            "y": roi.bbox.y,
            "w": roi.bbox.w,
            "h": roi.bbox.h,
        }

    scaled_rois = []
    for roi in layout.rois:
        new_bbox = dataclasses.replace(
            roi.bbox,
            x=int(roi.bbox.x * scale_x),
            y=int(roi.bbox.y * scale_y),
            w=int(roi.bbox.w * scale_x),
            h=int(roi.bbox.h * scale_y),
        )
        scaled_rois.append(dataclasses.replace(roi, bbox=new_bbox))

    rois_by_q = defaultdict(list)
    for roi in scaled_rois:
        rois_by_q[roi.question_id].append(roi)

    global_topology_evidence_dict = {}
    for q_id, rois in rois_by_q.items():
        if "Q14" not in q_id:
            gt_evidence = compute_global_topology_evidence(aligned_page.image, median_ref_bgr, rois)
            global_topology_evidence_dict[q_id] = gt_evidence

            # Debug full mask update
            if gt_evidence.ran and gt_evidence.skip_reason is None:
                crop_x1 = gt_evidence.crop["x1"]
                crop_y1 = gt_evidence.crop["y1"]
                for cl in gt_evidence.clusters:
                    if cl.hull_area > 1000 and cl.solidity < 0.4:
                        hull_pts = np.array(
                            [[[pt[0] + crop_x1, pt[1] + crop_y1]] for pt in cl.hull_points],
                            dtype=np.int32,
                        )
                        cv2.drawContours(
                            debug_full_mask,
                            [hull_pts],
                            0,
                            (0, 255, 255),
                            2,
                        )
        else:
            global_topology_evidence_dict[q_id] = None

    from matera.vision.q14_vlm_resolver import resolve_q14_with_vlm

    q14_vlm_evidence = None
    if "Q14" in rois_by_q:
        if q14_vlm_runtime is not None:
            q14_vlm_evidence = resolve_q14_with_vlm(aligned_page.image, q14_vlm_runtime)
        else:
            # If no runtime is supplied and Q14 exists, generate NEED_REVIEW for all options.
            from matera.vision.evidence.models import VlmOptionEvidence, VlmQuestionEvidence

            options = []
            for opt_roi in rois_by_q["Q14"]:
                options.append(
                    VlmOptionEvidence(
                        option_id=opt_roi.option_id,
                        decision="NEED_REVIEW",
                        raw_state=None,
                        reason=None,
                        parse_error="Q14 VLM runtime unavailable",
                    )
                )
            # Find the actual crop box for Q14 to be safe, or just mock it.
            q14_vlm_evidence = VlmQuestionEvidence(
                question_id="Q14",
                provider="unknown",
                model="unknown",
                prompt_version="unknown",
                crop_box={"x": 0, "y": 0, "w": 0, "h": 0},
                crop_path=None,
                raw_response_path=None,
                provider_error="Q14 VLM runtime unavailable",
                latency_ms=None,
                options=tuple(options),
            )

    question_evidences = []

    for q_id, rois in rois_by_q.items():
        opt_evidences = []
        roi_evidences = []
        gt_evidence = global_topology_evidence_dict[q_id]
        local_evidence_by_option: dict[str, Any] = {}
        global_marked_set = gt_evidence.global_marked if gt_evidence else set()

        if "Q14" in q_id:
            # Fast path for Q14
            strategy = strategy_map.get(q_id, "unknown")
            vlm_opt_map = (
                {o.option_id: o for o in q14_vlm_evidence.options} if q14_vlm_evidence else {}
            )

            for roi in rois:
                opt_id = roi.option_id
                orig_bbox = original_bboxes.get(
                    (q_id, opt_id),
                    {"x": roi.bbox.x, "y": roi.bbox.y, "w": roi.bbox.w, "h": roi.bbox.h},
                )
                scaled_bbox = {
                    "x": roi.bbox.x,
                    "y": roi.bbox.y,
                    "w": roi.bbox.w,
                    "h": roi.bbox.h,
                }

                roi_evidences.append(
                    RoiEvidence(
                        question_id=q_id,
                        option_id=opt_id,
                        bbox=orig_bbox,
                        scaled_bbox=scaled_bbox,
                        strategy=roi.mark_strategy_override or strategy,
                    )
                )

                vlm_opt_ev = vlm_opt_map.get(opt_id)
                # Map VLM decision to legacy prediction string
                pred = "AMBIGUOUS"
                method = "Q14_VLM_NEED_REVIEW"
                if vlm_opt_ev:
                    if vlm_opt_ev.decision == "MARKED":
                        pred = "MARKED"
                        method = "Q14_VLM_MARKED"
                    elif vlm_opt_ev.decision == "BLANK":
                        pred = "BLANK"
                        method = "Q14_VLM_BLANK"
                    else:
                        pred = "AMBIGUOUS"
                        if vlm_opt_ev.parse_error:
                            if "unavailable" in vlm_opt_ev.parse_error:
                                method = "Q14_VLM_CLIENT_UNAVAILABLE"
                            else:
                                method = "Q14_VLM_PARSE_ERROR"

                opt_evidences.append(
                    OptionMarkEvidence(
                        question_id=q_id,
                        option_id=opt_id,
                        strategy=roi.mark_strategy_override or strategy,
                        legacy_prediction=pred,
                        legacy_method=method,
                        selected_by_global=False,
                        local=None,
                        hsv_fallback=None,
                        checkbox_stroke=None,
                        checkbox_decision=None,
                        suspicion_notes=(),
                        vlm=vlm_opt_ev,
                    )
                )

            q_def = next(
                (q for q in profile.questions if q.question_id == q_id),
                None,
            )
            response_type = q_def.response_type if q_def else "single_select"

            question_evidences.append(
                QuestionMarkEvidence(
                    question_id=q_id,
                    strategy=strategy,
                    response_type=response_type,
                    rois=tuple(roi_evidences),
                    global_topology=None,
                    option_evidence=tuple(opt_evidences),
                    vlm=q14_vlm_evidence,
                )
            )
            continue

        for roi in rois:
            strategy = roi.mark_strategy_override or strategy_map.get(roi.question_id)
            if not strategy or strategy not in ("circle", "tick", "checkbox", "rating"):
                raise ValueError(
                    f"Unknown or missing mark strategy: {strategy} for question {q_id}"
                )
            opt_id = roi.option_id

            orig_bbox = original_bboxes.get(
                (q_id, opt_id),
                {"x": roi.bbox.x, "y": roi.bbox.y, "w": roi.bbox.w, "h": roi.bbox.h},
            )
            scaled_bbox = {
                "x": roi.bbox.x,
                "y": roi.bbox.y,
                "w": roi.bbox.w,
                "h": roi.bbox.h,
            }

            roi_evidences.append(
                RoiEvidence(
                    question_id=q_id,
                    option_id=opt_id,
                    bbox=orig_bbox,
                    scaled_bbox=scaled_bbox,
                    strategy=strategy,
                )
            )

            pred = "AMBIGUOUS"
            method = "UNKNOWN"
            local_evidence = None
            # Store local evidence for diagnostics (if any)
            if local_evidence is not None:
                local_evidence_by_option[opt_id] = local_evidence
            hsv_evidence = None
            stroke_ev = None
            dec = None

            target_bgr, mask_raw, crop_coords, ref_gray = get_local_roi_crops(
                aligned_page.image,
                median_ref_bgr,
                roi.bbox,
                LOCAL_PAD,
            )
            text_bbox = get_text_bounding_box(ref_gray)

            local_evidence = compute_local_option_evidence(
                mask_raw,
                text_bbox,
                crop_coords,
                OUTER_RADIUS,
                NUM_BINS,
                MIN_INK_PER_BIN,
            )
            # Store local evidence for diagnostics (including for global-selected options)
            if local_evidence is not None:
                local_evidence_by_option[opt_id] = local_evidence

            if opt_id in global_marked_set:
                pred = "MARKED"
                method = "GLOBAL_HULL"
            else:

                def _draw_radial_mask():
                    if debug_full_mask is not None:
                        h, w = mask_raw.shape
                        cx, cy = w / 2.0, h / 2.0
                        bx, by, bw, bh = text_bbox
                        bx = max(0, bx - 2)
                        by = max(0, by - 2)
                        bw, bh = bw + 4, bh + 4

                        Y, X = np.ogrid[:h, :w]
                        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
                        outer_mask = dist_sq > OUTER_RADIUS**2

                        core_mask = np.zeros((h, w), dtype=bool)
                        core_mask[by : min(h, by + bh), bx : min(w, bx + bw)] = True

                        mask_radial = mask_raw.copy()
                        mask_radial[core_mask] = 0
                        mask_radial[outer_mask] = 0

                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        mask_dilated = cv2.dilate(mask_radial, kernel, iterations=1)
                        mask_closed = cv2.morphologyEx(
                            mask_dilated, cv2.MORPH_CLOSE, kernel, iterations=1
                        )

                        mask_bgr = cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR)
                        cx1, cy1, cx2, cy2 = crop_coords
                        try:
                            debug_full_mask[cy1:cy2, cx1:cx2] = cv2.addWeighted(
                                debug_full_mask[cy1:cy2, cx1:cx2], 0.5, mask_bgr, 0.5, 0
                            )
                        except Exception:
                            pass

                if local_evidence.radial_degrees_covered >= MARKED_THRESHOLD_DEG:
                    pred = "MARKED"
                    method = "LOCAL_RADIAL"
                    _draw_radial_mask()
                elif local_evidence.radial_degrees_covered <= BLANK_THRESHOLD_DEG:
                    pred = "BLANK"
                    method = "LOCAL_RADIAL"
                    _draw_radial_mask()
                else:
                    hsv_res = process_roi_hsv(
                        aligned_page,
                        median_ref_bgr,
                        roi,
                        "FALLBACK",
                    )
                    pred = hsv_res.legacy_prediction
                    method = hsv_res.legacy_method
                    if debug_full_mask is not None:
                        _, mask_hsv, _ = get_local_roi_crops_hsv(
                            aligned_page.image, median_ref_bgr, roi.bbox, 0
                        )
                        mask_bgr = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)
                        mask_bgr[np.where((mask_bgr == [255, 255, 255]).all(axis=2))] = (
                            255,
                            0,
                            255,
                        )
                        cx1, cy1, cx2, cy2 = crop_coords
                        try:
                            debug_full_mask[cy1:cy2, cx1:cx2] = cv2.addWeighted(
                                debug_full_mask[cy1:cy2, cx1:cx2], 0.5, mask_bgr, 0.5, 0
                            )
                        except Exception:
                            pass
                    hsv_evidence = HsvFallbackEvidence(
                        method_prefix="FALLBACK",
                        hsv_ink_pixels=None,
                        classifier_probability=None,
                        classifier_available=False,
                        decision=pred,
                        method=method,
                    )

            opt_evidences.append(
                OptionMarkEvidence(
                    question_id=q_id,
                    option_id=opt_id,
                    strategy=strategy,
                    legacy_prediction=pred,
                    legacy_method=method,
                    selected_by_global=(opt_id in global_marked_set),
                    local=local_evidence,
                    hsv_fallback=hsv_evidence,
                    checkbox_stroke=stroke_ev,
                    checkbox_decision=dec,
                    suspicion_notes=(),
                )
            )

        # Compute global shape diagnostics after processing all ROIs
        gt_evidence = attach_global_shape_diagnostics(
            q_id=q_id,
            gt_evidence=gt_evidence,
            aligned_page=aligned_page,
            median_ref_bgr=median_ref_bgr,
            rois=rois,
            local_evidence_by_option=local_evidence_by_option,
            marked_threshold_deg=MARKED_THRESHOLD_DEG,
            blank_threshold_deg=BLANK_THRESHOLD_DEG,
        )
        global_topology_evidence_dict[q_id] = gt_evidence

        # --- VLM Escalation Logic ---
        q_vlm_evidence = None
        ambiguous_opt_ids = {
            o.option_id for o in opt_evidences if o.legacy_prediction == "AMBIGUOUS"
        }

        global_risk_opt_ids = set()
        if gt_evidence and gt_evidence.shape_diagnostics:
            for diag in gt_evidence.shape_diagnostics.clusters:
                if diag.should_call_vlm_later and diag.recommended_action == "CALL_VLM_LATER":
                    global_risk_opt_ids.update(diag.inside_options)

        escalation_scope = ambiguous_opt_ids | global_risk_opt_ids

        if escalation_scope:
            from matera.vision.question_vlm_resolver import resolve_question_with_vlm

            # If runtime provided, call it. Else create NEED_REVIEW evidence.
            if question_vlm_runtime is not None:
                q_vlm_evidence = resolve_question_with_vlm(
                    aligned_image=aligned_page.image,
                    question_id=q_id,
                    rois=rois,
                    option_ids=list(escalation_scope),
                    runtime=question_vlm_runtime,
                )
            else:
                from matera.vision.evidence.models import VlmOptionEvidence, VlmQuestionEvidence

                options = []
                for opt in escalation_scope:
                    options.append(
                        VlmOptionEvidence(
                            option_id=opt,
                            decision="NEED_REVIEW",
                            raw_state=None,
                            reason=None,
                            parse_error="Q1-Q13 VLM runtime unavailable",
                        )
                    )
                q_vlm_evidence = VlmQuestionEvidence(
                    question_id=q_id,
                    provider="unknown",
                    model="unknown",
                    prompt_version="v1",
                    crop_box={"x": 0, "y": 0, "w": 0, "h": 0},
                    crop_path=None,
                    raw_response_path=None,
                    provider_error="Q1-Q13 VLM runtime unavailable",
                    latency_ms=None,
                    options=tuple(options),
                )

            # Map VLM results back to opt_evidences
            vlm_opt_map = {o.option_id: o for o in q_vlm_evidence.options}
            new_opt_evidences = []
            for opt_ev in opt_evidences:
                if opt_ev.option_id in escalation_scope:
                    vlm_ev = vlm_opt_map.get(opt_ev.option_id)
                    pred = "AMBIGUOUS"
                    method = "Q1_Q13_VLM_NEED_REVIEW"
                    if vlm_ev:
                        if vlm_ev.decision == "MARKED":
                            pred = "MARKED"
                            method = "Q1_Q13_VLM_MARKED"
                        elif vlm_ev.decision == "BLANK":
                            pred = "BLANK"
                            method = "Q1_Q13_VLM_BLANK"
                        else:
                            pred = "AMBIGUOUS"
                            if q_vlm_evidence.provider_error:
                                if "unavailable" in q_vlm_evidence.provider_error:
                                    method = "Q1_Q13_VLM_CLIENT_UNAVAILABLE"
                                else:
                                    method = "Q1_Q13_VLM_PROVIDER_ERROR"
                            elif vlm_ev.parse_error:
                                method = "Q1_Q13_VLM_PARSE_ERROR"

                    new_opt_evidences.append(
                        dataclasses.replace(
                            opt_ev,
                            legacy_prediction=pred,
                            legacy_method=method,
                            vlm=vlm_ev,
                        )
                    )
                else:
                    new_opt_evidences.append(opt_ev)
            opt_evidences = new_opt_evidences

        q_def = next(
            (q for q in profile.questions if q.question_id == q_id),
            None,
        )
        response_type = q_def.response_type if q_def else "single_select"

        question_evidences.append(
            QuestionMarkEvidence(
                question_id=q_id,
                strategy=strategy_map.get(q_id, "unknown"),
                response_type=response_type,
                rois=tuple(roi_evidences),
                global_topology=gt_evidence,
                option_evidence=tuple(opt_evidences),
                vlm=q_vlm_evidence,
            )
        )

    return PageMarkEvidence(
        page_number=aligned_page.page_number,
        form_id=aligned_page.profile_form_id,
        form_version=aligned_page.profile_version,
        alignment=AlignmentEvidence(
            alignment_score=aligned_page.alignment_score,
            warp_matrix=(
                tuple(tuple(r) for r in aligned_page.warp_matrix.tolist())
                if isinstance(aligned_page.warp_matrix, np.ndarray)
                else ()
            ),
            image_size=(orig_bgr.shape[1], orig_bgr.shape[0]),
            layout_size=(layout.width_px, layout.height_px),
            scale_x=scale_x,
            scale_y=scale_y,
        ),
        reference=ReferenceEvidence(
            width=reference_image.width,
            height=reference_image.height,
            dpi=aligned_page.reference_dpi,
        ),
        questions=tuple(question_evidences),
        thresholds=MarkThresholdEvidence(
            marked_threshold_deg=MARKED_THRESHOLD_DEG,
            blank_threshold_deg=BLANK_THRESHOLD_DEG,
            routing_low_threshold=None,
            routing_high_threshold=None,
            local_pad_px=LOCAL_PAD,
            outer_radius_px=OUTER_RADIUS,
            diff_threshold=DIFF_THRESHOLD,
            global_pad_px=GLOBAL_PAD,
        ),
    )

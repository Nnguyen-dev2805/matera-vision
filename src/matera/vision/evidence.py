from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from PIL import Image

    from matera.core.layout import PageLayout
    from matera.core.profile import FormProfile
    from matera.vision.contracts import AlignedPage, MarkScore


@dataclass(frozen=True)
class AlignmentEvidence:
    alignment_score: float
    warp_matrix: tuple[tuple[float, ...], ...]
    image_size: tuple[int, int]
    layout_size: tuple[int, int]
    scale_x: float
    scale_y: float


@dataclass(frozen=True)
class ReferenceEvidence:
    width: int
    height: int
    dpi: int


@dataclass(frozen=True)
class MarkThresholdEvidence:
    marked_threshold_deg: float
    blank_threshold_deg: float
    routing_low_threshold: float | None
    routing_high_threshold: float | None
    local_pad_px: int
    outer_radius_px: int
    diff_threshold: int
    global_pad_px: int


@dataclass(frozen=True)
class RoiEvidence:
    question_id: str
    option_id: str
    bbox: dict[str, int]
    scaled_bbox: dict[str, int]
    strategy: str


@dataclass(frozen=True)
class LocalOptionEvidence:
    crop_coords: dict[str, int]
    diff_ink_pixels: int
    diff_contour_count: int
    diff_contour_areas: tuple[float, ...]
    text_bbox: dict[str, int]
    core_mask_pixels: int
    outer_mask_pixels: int
    radial_mask_before_morph_px: int
    radial_ink_pixels: int
    radial_active_bin_count: int
    radial_degrees_covered: float
    radial_histogram: tuple[int, ...]


@dataclass(frozen=True)
class OptionCenterEvidence:
    option_id: str
    center_x: int
    center_y: int


@dataclass(frozen=True)
class GlobalContourEvidence:
    contour_id: str
    area: float
    bbox: dict[str, int]
    status: Literal["ACCEPTED", "REJECTED"]
    rejection_reason: str | None


@dataclass(frozen=True)
class GlobalClusterEvidence:
    cluster_id: str
    contour_ids: tuple[str, ...]
    total_area: float
    hull_area: float
    solidity: float
    bbox: dict[str, int]
    hull_points: tuple[tuple[int, int], ...]
    inside_options: tuple[str, ...]
    option_center_distances: dict[str, float]


@dataclass(frozen=True)
class GlobalTopologyEvidence:
    question_id: str
    ran: bool
    skip_reason: str | None
    crop: dict[str, int]
    crop_expansion: dict[str, Any] | None
    option_centers: tuple[OptionCenterEvidence, ...]
    contours: tuple[GlobalContourEvidence, ...]
    clusters: tuple[GlobalClusterEvidence, ...]
    global_marked: frozenset[str]


@dataclass(frozen=True)
class HsvFallbackEvidence:
    method_prefix: str
    hsv_ink_pixels: int | None
    classifier_probability: float | None
    classifier_available: bool
    decision: Literal["MARKED", "BLANK", "AMBIGUOUS"]
    method: str


@dataclass(frozen=True)
class OptionMarkEvidence:
    question_id: str
    option_id: str
    strategy: str
    legacy_prediction: Literal["MARKED", "BLANK", "AMBIGUOUS"]
    legacy_method: str
    selected_by_global: bool
    local: LocalOptionEvidence | None
    hsv_fallback: HsvFallbackEvidence | None
    suspicion_notes: tuple[str, ...]


@dataclass(frozen=True)
class QuestionMarkEvidence:
    question_id: str
    strategy: str
    response_type: str
    rois: tuple[RoiEvidence, ...]
    global_topology: GlobalTopologyEvidence | None
    option_evidence: tuple[OptionMarkEvidence, ...]


@dataclass(frozen=True)
class PageMarkEvidence:
    page_number: int
    form_id: str
    form_version: str
    alignment: AlignmentEvidence
    reference: ReferenceEvidence
    questions: tuple[QuestionMarkEvidence, ...]
    thresholds: MarkThresholdEvidence


def evidence_to_json_dict(evidence: PageMarkEvidence) -> dict[str, Any]:
    """Convert PageMarkEvidence into a JSON-serializable dictionary."""

    def _convert(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            if obj.size > 100:
                return {"_type": "ndarray", "shape": obj.shape, "dtype": str(obj.dtype)}
            return obj.tolist()
        elif isinstance(obj, (tuple, list, frozenset, set)):
            return [_convert(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    return _convert(evidence)


def compute_local_option_evidence(
    mask_raw: np.ndarray,
    text_bbox: tuple[int, int, int, int],
    crop_coords: tuple[int, int, int, int],
    outer_radius: int = 32,
    num_bins: int = 72,
    min_ink_per_bin: int = 2,
) -> LocalOptionEvidence:

    bx, by, bw, bh = text_bbox
    h, w = mask_raw.shape
    cx, cy = w / 2.0, h / 2.0

    bx = max(0, bx - 2)
    by = max(0, by - 2)
    bw = bw + 4
    bh = bh + 4

    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
    outer_mask = dist_sq > outer_radius**2

    core_mask = np.zeros((h, w), dtype=bool)
    by_e = min(h, by + bh)
    bx_e = min(w, bx + bw)
    core_mask[by:by_e, bx:bx_e] = True

    mask_radial = mask_raw.copy()
    mask_radial[core_mask] = 0
    mask_radial[outer_mask] = 0

    radial_mask_before_morph_px = int(np.sum(mask_radial > 0))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask_radial, kernel, iterations=1)
    mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

    ys, xs = np.where(mask_closed > 0)
    radial_ink_pixels = len(xs)

    degrees_covered = 0.0
    active_bin_count = 0
    hist_counts = np.zeros(num_bins, dtype=int)
    degrees_per_bin = 360.0 / num_bins

    if radial_ink_pixels > 0:
        dx = xs.astype(float) - cx
        dy = ys.astype(float) - cy
        angles = np.arctan2(dy, dx)
        angles = np.degrees(angles) % 360
        hist_counts, _ = np.histogram(angles, bins=num_bins, range=(0, 360))
        active_bins = (hist_counts >= min_ink_per_bin).astype(int)
        for i in range(num_bins):
            left = active_bins[(i - 1) % num_bins]
            right = active_bins[(i + 1) % num_bins]
            if active_bins[i] == 0 and left == 1 and right == 1:
                active_bins[i] = 1
        active_bin_count = int(np.sum(active_bins))
        degrees_covered = float(active_bin_count * degrees_per_bin)

    core_mask_pixels = int(np.sum(core_mask))
    outer_mask_pixels = int(np.sum(outer_mask))

    crop_dict = {
        "x1": crop_coords[0],
        "y1": crop_coords[1],
        "x2": crop_coords[2],
        "y2": crop_coords[3],
    }
    text_bbox_dict = {
        "x": text_bbox[0],
        "y": text_bbox[1],
        "w": text_bbox[2],
        "h": text_bbox[3],
    }

    diff_ink_pixels = int(np.sum(mask_raw > 0))
    cnts, _ = cv2.findContours(mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diff_contour_count = len(cnts)
    diff_contour_areas = tuple(float(cv2.contourArea(c)) for c in cnts)

    return LocalOptionEvidence(
        crop_coords=crop_dict,
        diff_ink_pixels=diff_ink_pixels,
        diff_contour_count=diff_contour_count,
        diff_contour_areas=diff_contour_areas,
        text_bbox=text_bbox_dict,
        core_mask_pixels=core_mask_pixels,
        outer_mask_pixels=outer_mask_pixels,
        radial_mask_before_morph_px=radial_mask_before_morph_px,
        radial_ink_pixels=radial_ink_pixels,
        radial_active_bin_count=active_bin_count,
        radial_degrees_covered=degrees_covered,
        radial_histogram=tuple(hist_counts.tolist()),
    )


def compute_global_topology_evidence(
    aligned_image_rgb: Image.Image,
    median_ref_bgr: np.ndarray,
    rois: list,
) -> GlobalTopologyEvidence:
    from collections import defaultdict

    from matera.vision.adaptive_crop import compute_adaptive_global_crop
    from matera.vision.mark import (
        CLOSE_ITERATIONS,
        CLOSE_KERNEL_SIZE,
        DIFF_THRESHOLD,
        EXTREMES_REJECT_THRESHOLD,
        GAUSS_KERNEL,
        MERGE_THRESHOLD,
        UnionFind,
        get_horizontal_extremes,
        min_contour_distance,
    )

    question_id = rois[0].question_id if rois else ""

    if len(rois) <= 1:
        return GlobalTopologyEvidence(
            question_id=question_id,
            ran=False,
            skip_reason="<= 1 rois",
            crop={"x1": 0, "y1": 0, "x2": 0, "y2": 0},
            crop_expansion=None,
            option_centers=(),
            contours=(),
            clusters=(),
            global_marked=frozenset(),
        )

    crop_result = compute_adaptive_global_crop(aligned_image_rgb, median_ref_bgr, rois)
    crop = crop_result.crop
    crop_x1, crop_y1, crop_x2, crop_y2 = crop["x1"], crop["y1"], crop["x2"], crop["y2"]

    crop_img = aligned_image_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    target_bgr = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
    ref_crop = median_ref_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(ref_gray, target_gray)
    blurred = cv2.GaussianBlur(diff, GAUSS_KERNEL, 0)
    _, mask_raw = cv2.threshold(blurred, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    h, w = mask_raw.shape
    mask_raw[:15, :] = 0
    mask_raw[h - 15 :, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERATIONS)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    global_marked = set()
    option_centers = {}
    option_centers_ev = []
    for r in rois:
        cx = (r.bbox.x + r.bbox.w / 2.0) - crop_x1
        cy = (r.bbox.y + r.bbox.h / 2.0) - crop_y1
        option_centers[r.option_id] = (cx, cy)
        option_centers_ev.append(OptionCenterEvidence(r.option_id, int(cx), int(cy)))

    valid_cnts = []
    contours_ev = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x, y, w_b, h_b = cv2.boundingRect(cnt)
        bbox = {"x": x, "y": y, "w": w_b, "h": h_b}
        if area <= 30:
            contours_ev.append(
                GlobalContourEvidence(
                    f"cnt_{i}",
                    float(area),
                    bbox,
                    "REJECTED",
                    "area <= 30",
                )
            )
            continue
        if (h_b > 100 and w_b < 25) or (w_b > 100 and h_b < 25):
            contours_ev.append(
                GlobalContourEvidence(
                    f"cnt_{i}",
                    float(area),
                    bbox,
                    "REJECTED",
                    "aspect ratio",
                )
            )
            continue

        valid_cnts.append((f"cnt_{i}", cnt))
        contours_ev.append(
            GlobalContourEvidence(
                f"cnt_{i}",
                float(area),
                bbox,
                "ACCEPTED",
                None,
            )
        )

    n = len(valid_cnts)
    if n == 0:
        return GlobalTopologyEvidence(
            question_id=question_id,
            ran=True,
            skip_reason="no valid contours",
            crop={"x1": crop_x1, "y1": crop_y1, "x2": crop_x2, "y2": crop_y2},
            crop_expansion={
                "pads": crop_result.pads,
                "iterations": crop_result.iterations,
                "expanded": crop_result.expanded,
                "stop_reason": crop_result.stop_reason,
            }
            if hasattr(crop_result, "expanded")
            else None,
            option_centers=tuple(option_centers_ev),
            contours=tuple(contours_ev),
            clusters=(),
            global_marked=frozenset(),
        )

    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            dist = min_contour_distance(valid_cnts[i][1], valid_cnts[j][1])
            if dist <= MERGE_THRESHOLD:
                l1, r1 = get_horizontal_extremes(valid_cnts[i][1])
                l2, r2 = get_horizontal_extremes(valid_cnts[j][1])
                dist_left = np.linalg.norm(l1 - l2)
                dist_right = np.linalg.norm(r1 - r2)

                if dist_left > EXTREMES_REJECT_THRESHOLD and dist_right > EXTREMES_REJECT_THRESHOLD:
                    pass
                else:
                    uf.union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(valid_cnts[i])

    clusters_ev = []
    for root, cnt_list in clusters.items():
        combined_points = np.vstack([c[1] for c in cnt_list])
        total_area = 0.0
        for _, c in cnt_list:
            mask = np.zeros_like(mask_raw)
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
            total_area += float(cv2.countNonZero(cv2.bitwise_and(mask_raw, mask)))

        hull = cv2.convexHull(combined_points)
        hull_area = cv2.contourArea(hull)

        # Calculate cluster bbox and points
        cx_min, cy_min, cx_max, cy_max = float("inf"), float("inf"), 0.0, 0.0
        for pt in hull:
            px, py = pt[0]
            cx_min = min(cx_min, px)
            cy_min = min(cy_min, py)
            cx_max = max(cx_max, px)
            cy_max = max(cy_max, py)
        cluster_bbox = {
            "x": int(cx_min),
            "y": int(cy_min),
            "w": int(cx_max - cx_min),
            "h": int(cy_max - cy_min),
        }
        hull_pts = tuple((int(pt[0][0]), int(pt[0][1])) for pt in hull)

        inside_options = []
        option_dists = {}
        solidity = total_area / float(hull_area) if hull_area > 0 else 1.0

        if hull_area > 1000 and solidity < 0.4:
            for opt_id, (cx, cy) in option_centers.items():
                dist = cv2.pointPolygonTest(hull, (cx, cy), True)
                option_dists[opt_id] = float(dist)
                if dist >= 0:
                    global_marked.add(opt_id)
                    inside_options.append(opt_id)

        clusters_ev.append(
            GlobalClusterEvidence(
                cluster_id=f"cluster_{root}",
                contour_ids=tuple(c[0] for c in cnt_list),
                total_area=float(total_area),
                hull_area=float(hull_area),
                solidity=float(solidity),
                bbox=cluster_bbox,
                hull_points=hull_pts,
                inside_options=tuple(inside_options),
                option_center_distances=option_dists,
            )
        )

    return GlobalTopologyEvidence(
        question_id=question_id,
        ran=True,
        skip_reason=None,
        crop={"x1": crop_x1, "y1": crop_y1, "x2": crop_x2, "y2": crop_y2},
        crop_expansion={
            "pads": crop_result.pads,
            "iterations": crop_result.iterations,
            "expanded": crop_result.expanded,
            "stop_reason": crop_result.stop_reason,
        }
        if hasattr(crop_result, "expanded")
        else None,
        option_centers=tuple(option_centers_ev),
        contours=tuple(contours_ev),
        clusters=tuple(clusters_ev),
        global_marked=frozenset(global_marked),
    )


def extract_mark_evidence(
    aligned_page: AlignedPage,
    profile: FormProfile,
    layout: PageLayout,
    reference_image: Image.Image,
    *,
    debug_dir: str | None = None,
    debug_full_mask: np.ndarray | None = None,
) -> PageMarkEvidence:
    from collections import defaultdict

    from matera.vision.mark import (
        BLANK_THRESHOLD_DEG,
        DIFF_THRESHOLD,
        GLOBAL_PAD,
        LOCAL_PAD,
        MARKED_THRESHOLD_DEG,
        MIN_INK_PER_BIN,
        NUM_BINS,
        OUTER_RADIUS,
        get_local_roi_crops,
        get_local_roi_crops_hsv,
        get_text_bounding_box,
        process_roi_hsv_ai,
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

    question_evidences = []

    for q_id, rois in rois_by_q.items():
        opt_evidences = []
        roi_evidences = []
        gt_evidence = global_topology_evidence_dict[q_id]
        global_marked_set = gt_evidence.global_marked if gt_evidence else set()

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
            hsv_evidence = None

            if "Q14" not in q_id:
                if opt_id in global_marked_set:
                    pred = "MARKED"
                    method = "GLOBAL_HULL"
                else:
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
                        pred, method = process_roi_hsv_ai(
                            aligned_page,
                            median_ref_bgr,
                            roi,
                            "FALLBACK",
                        )
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
                            classifier_available=("_NO_AI" not in method),
                            decision=pred,
                            method=method,
                        )
            else:
                pred, method = process_roi_hsv_ai(aligned_page, median_ref_bgr, roi, "HSV_AI")
                if debug_full_mask is not None:
                    _, mask_hsv, _ = get_local_roi_crops_hsv(
                        aligned_page.image, median_ref_bgr, roi.bbox, 0
                    )
                    mask_bgr = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)
                    mask_bgr[np.where((mask_bgr == [255, 255, 255]).all(axis=2))] = (255, 0, 255)
                    cx1, cy1 = max(0, roi.bbox.x), max(0, roi.bbox.y)
                    cx2, cy2 = cx1 + roi.bbox.w, cy1 + roi.bbox.h
                    try:
                        debug_full_mask[cy1:cy2, cx1:cx2] = cv2.addWeighted(
                            debug_full_mask[cy1:cy2, cx1:cx2], 0.5, mask_bgr, 0.5, 0
                        )
                    except Exception:
                        pass
                hsv_evidence = HsvFallbackEvidence(
                    method_prefix="HSV_AI",
                    hsv_ink_pixels=None,
                    classifier_probability=None,
                    classifier_available=("_NO_AI" not in method),
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
                    suspicion_notes=(),
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
                strategy=strategy_map.get(q_id, "unknown"),
                response_type=response_type,
                rois=tuple(roi_evidences),
                global_topology=gt_evidence,
                option_evidence=tuple(opt_evidences),
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


def evidence_to_mark_scores(
    page_evidence: PageMarkEvidence,
    aligned_page: "AlignedPage",
    debug_dir: str | None = None,
) -> list["MarkScore"]:
    from pathlib import Path

    from matera.vision.contracts import MarkScore

    debug_path = Path(debug_dir) if debug_dir else None
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)

    scores = []
    for question in page_evidence.questions:
        for opt in question.option_evidence:
            score_val = (
                1.0
                if opt.legacy_prediction == "MARKED"
                else 0.0
                if opt.legacy_prediction == "BLANK"
                else 0.5
            )

            image_crop = None
            evidence_path = None

            # Spec requires behavior-preserving crop (exact unpadded scaled_bbox)
            roi_ev = next((r for r in question.rois if r.option_id == opt.option_id), None)
            if roi_ev:
                scaled = roi_ev.scaled_bbox
                cx1, cy1 = scaled["x"], scaled["y"]
                cx2, cy2 = cx1 + scaled["w"], cy1 + scaled["h"]
                image_crop = aligned_page.image.crop((cx1, cy1, cx2, cy2))

            if debug_path and image_crop:
                evidence_file = (
                    debug_path
                    / f"page_{page_evidence.page_number}_{opt.question_id}_{opt.option_id}.png"
                )
                image_crop.save(evidence_file)
                evidence_path = evidence_file

            scores.append(
                MarkScore(
                    question_id=opt.question_id,
                    option_id=opt.option_id,
                    score=score_val,
                    strategy=opt.strategy,
                    method=opt.legacy_method,
                    image_crop=image_crop,
                    evidence_path=evidence_path,
                )
            )

    return scores

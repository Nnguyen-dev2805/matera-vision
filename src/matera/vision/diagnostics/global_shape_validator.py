from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Sequence, Tuple

import cv2
import numpy as np


# === Config ===
@dataclass(frozen=True)
class GlobalShapeValidatorConfig:
    min_group_enclosure_score: float = 0.72
    max_local_cluster_score_for_group: float = 0.65
    min_vlm_risk_score: float = 0.60
    strong_local_degrees: float = 360.0  # placeholder, overridden by MARKED_THRESHOLD_DEG elsewhere
    blank_local_degrees: float = 0.0
    boundary_band_px: int = 18
    option_band_pad_px: int = 12
    diagnostic_only: bool = True


# === Evidence dataclasses ===
@dataclass(frozen=True)
class BoundaryEvidence:
    left_score: float
    right_score: float
    top_score: float
    bottom_score: float
    balance_score: float
    vertical_coverage_score: float
    horizontal_coverage_score: float
    missing_boundaries: Tuple[str, ...]
    debug_rects: Mapping[str, Mapping[str, int]] | None = None


@dataclass(frozen=True)
class LocalCircleClusterEvidence:
    score: float
    strong_local_options: Tuple[str, ...]
    weak_local_options_inside_hull: Tuple[str, ...]
    local_strength_by_option: Mapping[str, float]
    row_gap_breaks: Tuple[str, ...]
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class GlobalShapeCandidateDiagnostic:
    cluster_id: str
    inside_options: Tuple[str, ...]
    shape_class: Literal["GROUP_ENCLOSURE", "LOCAL_CIRCLE_CLUSTER", "UNCERTAIN"]
    group_enclosure_score: float
    local_circle_cluster_score: float
    vlm_risk_score: float
    should_call_vlm_later: bool
    boundary_evidence: BoundaryEvidence
    local_cluster_evidence: LocalCircleClusterEvidence
    recommended_action: Literal["KEEP_GLOBAL", "USE_LOCAL_OR_REVIEW", "CALL_VLM_LATER"]
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class OptionGlobalShapeDiagnostic:
    question_id: str
    option_id: str
    cluster_id: str | None
    selected_by_current_global: bool
    local_radial_degrees: float | None
    local_radial_ink_pixels: int | None
    center_signed_distance_px: float | None
    option_risk_score: float
    recommended_action: Literal["KEEP_GLOBAL", "USE_LOCAL", "CALL_VLM_LATER", "NO_GLOBAL_CANDIDATE"]
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class GlobalShapeDiagnostics:
    validator_mode: str = "diagnostic"
    config: GlobalShapeValidatorConfig = GlobalShapeValidatorConfig()
    clusters: Tuple[GlobalShapeCandidateDiagnostic, ...] = ()
    options: Tuple[OptionGlobalShapeDiagnostic, ...] = ()


# Helper stubs (real implementation would be more sophisticated)
def _compute_boundary_evidence(
    cluster: Any,
    rois: Sequence[Any],
    mask_raw: np.ndarray | None,
    crop_dict: Mapping[str, int] | None,
    cfg: GlobalShapeValidatorConfig,
) -> BoundaryEvidence:
    """Compute realistic boundary evidence based on hull vs option union bbox.
    Scores are fractions of cluster ink pixels that lie within a band of
    `cfg.boundary_band_px` pixels around the union bounding box of the inside options.
    Coordinates are adjusted to crop-local.
    """
    x_offset = crop_dict["x1"] if crop_dict else 0
    y_offset = crop_dict["y1"] if crop_dict else 0

    opt_bboxes = []
    for opt_id in getattr(cluster, "inside_options", []):
        for r in rois:
            if getattr(r, "option_id", None) == opt_id:
                # convert ROI to crop-local
                bx = r.bbox.x - x_offset
                by = r.bbox.y - y_offset
                bw = r.bbox.w
                bh = r.bbox.h
                opt_bboxes.append((bx, by, bw, bh))
                break

    if not opt_bboxes:
        # Fallback to hull bbox
        hull_pts = np.array(cluster.hull_points, dtype=np.int32)
        hx, hy, hw, hh = cv2.boundingRect(hull_pts)
        ox, oy, ow, oh = hx, hy, hw, hh
    else:
        hull_pts = np.array(cluster.hull_points, dtype=np.int32)
        hx, hy, hw, hh = cv2.boundingRect(hull_pts)
        ox = min(b[0] for b in opt_bboxes)
        oy = min(b[1] for b in opt_bboxes)
        ox2 = max(b[0] + b[2] for b in opt_bboxes)
        oy2 = max(b[1] + b[3] for b in opt_bboxes)
        ow = ox2 - ox
        oh = oy2 - oy

    pad_inner = cfg.option_band_pad_px
    pad_min = cfg.boundary_band_px

    ix1 = max(0, ox - pad_inner)
    iy1 = max(0, oy - pad_inner)
    ix2 = ox + ow + pad_inner
    iy2 = oy + oh + pad_inner

    left_x1 = min(hx, ix1 - pad_min)
    left_x2 = ix1
    right_x1 = ix2
    right_x2 = max(hx + hw, ix2 + pad_min)
    top_y1 = min(hy, iy1 - pad_min)
    top_y2 = iy1
    bottom_y1 = iy2
    bottom_y2 = max(hy + hh, iy2 + pad_min)
    v_y1, v_y2 = max(0, iy1), iy2
    h_x1, h_x2 = max(0, ix1), ix2
    if mask_raw is not None:
        mh, mw = mask_raw.shape
        mask = np.zeros_like(mask_raw)
        hull_pts = np.array(cluster.hull_points, dtype=np.int32)
        cv2.drawContours(mask, [hull_pts], -1, 255, thickness=cv2.FILLED)
        cluster_ink = cv2.bitwise_and(mask_raw, mask)
    else:
        # Fallback if mask_raw is missing (for tests)
        mh, mw = 10000, 10000
        cluster_ink = np.zeros((mh, mw), dtype=np.uint8)
        hull_pts = np.array(cluster.hull_points, dtype=np.int32)
        cv2.drawContours(cluster_ink, [hull_pts], -1, 255, thickness=cv2.FILLED)

    def clamp_x(val: int) -> int:
        return max(0, min(mw, val))

    def clamp_y(val: int) -> int:
        return max(0, min(mh, val))

    left_roi = cluster_ink[clamp_y(v_y1) : clamp_y(v_y2), clamp_x(left_x1) : clamp_x(left_x2)]
    right_roi = cluster_ink[clamp_y(v_y1) : clamp_y(v_y2), clamp_x(right_x1) : clamp_x(right_x2)]
    top_roi = cluster_ink[clamp_y(top_y1) : clamp_y(top_y2), clamp_x(h_x1) : clamp_x(h_x2)]
    bottom_roi = cluster_ink[clamp_y(bottom_y1) : clamp_y(bottom_y2), clamp_x(h_x1) : clamp_x(h_x2)]

    left_score = float(np.mean(np.any(left_roi > 0, axis=1))) if left_roi.size > 0 else 0.0
    right_score = float(np.mean(np.any(right_roi > 0, axis=1))) if right_roi.size > 0 else 0.0
    top_score = float(np.mean(np.any(top_roi > 0, axis=0))) if top_roi.size > 0 else 0.0
    bottom_score = float(np.mean(np.any(bottom_roi > 0, axis=0))) if bottom_roi.size > 0 else 0.0

    balance_score = max(0.0, 1.0 - abs(left_score - right_score))
    vertical_coverage_score = (top_score + bottom_score) / 2.0
    horizontal_coverage_score = (left_score + right_score) / 2.0

    missing = []
    if left_score < 0.25:
        missing.append("left")
    if right_score < 0.25:
        missing.append("right")
    if top_score < 0.25:
        missing.append("top")
    if bottom_score < 0.25:
        missing.append("bottom")

    debug_rects = {
        "option_group_bbox": {"x1": ox, "y1": oy, "x2": ox + ow, "y2": oy + oh},
        "expanded_option_group_bbox": {"x1": ix1, "y1": iy1, "x2": ix2, "y2": iy2},
        "hull_bbox": {"x1": hx, "y1": hy, "x2": hx + hw, "y2": hy + hh},
        "left_band_rect": {
            "x1": clamp_x(left_x1),
            "y1": clamp_y(v_y1),
            "x2": clamp_x(left_x2),
            "y2": clamp_y(v_y2),
        },
        "right_band_rect": {
            "x1": clamp_x(right_x1),
            "y1": clamp_y(v_y1),
            "x2": clamp_x(right_x2),
            "y2": clamp_y(v_y2),
        },
        "top_band_rect": {
            "x1": clamp_x(h_x1),
            "y1": clamp_y(top_y1),
            "x2": clamp_x(h_x2),
            "y2": clamp_y(top_y2),
        },
        "bottom_band_rect": {
            "x1": clamp_x(h_x1),
            "y1": clamp_y(bottom_y1),
            "x2": clamp_x(h_x2),
            "y2": clamp_y(bottom_y2),
        },
    }

    return BoundaryEvidence(
        left_score=left_score,
        right_score=right_score,
        top_score=top_score,
        bottom_score=bottom_score,
        balance_score=balance_score,
        vertical_coverage_score=vertical_coverage_score,
        horizontal_coverage_score=horizontal_coverage_score,
        missing_boundaries=tuple(missing),
        debug_rects=debug_rects,
    )


def _compute_local_circle_cluster_evidence(
    cluster: Any, local_by_option: Mapping[str, Any], cfg: GlobalShapeValidatorConfig
) -> LocalCircleClusterEvidence:
    """Compute local circle cluster evidence using configurable thresholds.
    Uses the config's `strong_local_degrees` and `blank_local_degrees`.
    """
    strong_opts: list[str] = []
    weak_opts: list[str] = []
    strengths: dict[str, float] = {}
    for opt in getattr(cluster, "inside_options", []):
        loc = local_by_option.get(opt)
        if loc is None:
            continue
        deg = getattr(loc, "radial_degrees_covered", 0.0)
        strengths[opt] = deg / 360.0
        if deg >= cfg.strong_local_degrees:
            strong_opts.append(opt)
        elif deg > cfg.blank_local_degrees:
            weak_opts.append(opt)
    score = len(strong_opts) / max(1, len(cluster.inside_options))
    return LocalCircleClusterEvidence(
        score=score,
        strong_local_options=tuple(strong_opts),
        weak_local_options_inside_hull=tuple(weak_opts),
        local_strength_by_option=strengths,
        row_gap_breaks=(),
        reasons=(),
    )


def _classify_shape(
    cluster: Any,
    boundary: BoundaryEvidence,
    local_cluster: LocalCircleClusterEvidence,
    cfg: GlobalShapeValidatorConfig,
) -> Tuple[str, float, float, bool, str, List[str]]:
    """Classify a cluster into a shape class and decide VLM action.

    Returns a tuple of
    (shape_class, group_score, local_score, should_vlm, recommended_action, reasons).
    """

    def normalized(value: float, threshold: float) -> float:
        return min(1.0, max(0.0, value / threshold))

    # A handwritten group enclosure is often not a neat rectangle. In real Q4
    # samples, the two long side strokes are the strongest signal, while one
    # top/bottom cap can be faint or missing after diff thresholding.
    left_quality = normalized(boundary.left_score, 0.45)
    right_quality = normalized(boundary.right_score, 0.45)
    cap_quality = normalized(max(boundary.top_score, boundary.bottom_score), 0.25)
    side_pair_quality = min(left_quality, right_quality)
    side_average_quality = (left_quality + right_quality) / 2.0
    group_score = (
        0.45 * side_pair_quality
        + 0.25 * side_average_quality
        + 0.20 * cap_quality
        + 0.10 * boundary.balance_score
    )
    group_score = min(1.0, max(0.0, group_score))
    local_score = local_cluster.score
    reasons: List[str] = []

    # Determine shape class based on thresholds.
    if group_score >= cfg.min_group_enclosure_score:
        shape_class = "GROUP_ENCLOSURE"
        reasons.append(
            f"group_score {group_score:.3f} >= "
            f"min_group_enclosure_score {cfg.min_group_enclosure_score}"
        )
    elif local_score >= cfg.max_local_cluster_score_for_group:
        shape_class = "LOCAL_CIRCLE_CLUSTER"
        reasons.append(
            f"local_score {local_score:.3f} >= "
            f"max_local_cluster_score_for_group {cfg.max_local_cluster_score_for_group}"
        )
    else:
        shape_class = "UNCERTAIN"
        reasons.append(
            f"group_score {group_score:.3f} and local_score {local_score:.3f} below thresholds"
        )

    # VLM risk based on how far group_score is from perfect.
    vlm_risk_score = 1.0 - group_score
    should_vlm = vlm_risk_score >= cfg.min_vlm_risk_score
    if should_vlm:
        reasons.append(
            f"vlm_risk_score {vlm_risk_score:.3f} >= min_vlm_risk_score {cfg.min_vlm_risk_score}"
        )

    recommended_action = "CALL_VLM_LATER" if should_vlm else "KEEP_GLOBAL"
    return shape_class, group_score, local_score, should_vlm, recommended_action, reasons


def diagnose_global_shape_candidates(
    *,
    question_id: str,
    rois: Sequence[Any],
    global_topology: Any,
    local_evidence_by_option: Mapping[str, Any],
    mask_raw: np.ndarray | None = None,
    config: GlobalShapeValidatorConfig | None = None,
) -> GlobalShapeDiagnostics:
    cfg = config or GlobalShapeValidatorConfig()
    cluster_diags: list[GlobalShapeCandidateDiagnostic] = []
    option_diags: list[OptionGlobalShapeDiagnostic] = []
    for cluster in getattr(global_topology, "clusters", []):
        boundary = _compute_boundary_evidence(
            cluster, rois, mask_raw, getattr(global_topology, "crop", None), cfg
        )
        local_cluster = _compute_local_circle_cluster_evidence(
            cluster, local_evidence_by_option, cfg
        )
        shape_class, group_score, local_score, should_vlm, action, reasons = _classify_shape(
            cluster, boundary, local_cluster, cfg
        )
        vlm_risk = 1.0 - group_score  # placeholder
        diag = GlobalShapeCandidateDiagnostic(
            cluster_id=cluster.cluster_id,
            inside_options=cluster.inside_options,
            shape_class=shape_class,
            group_enclosure_score=group_score,
            local_circle_cluster_score=local_score,
            vlm_risk_score=vlm_risk,
            should_call_vlm_later=should_vlm,
            boundary_evidence=boundary,
            local_cluster_evidence=local_cluster,
            recommended_action=action,
            reasons=tuple(reasons),
        )
        cluster_diags.append(diag)
        for opt_id in cluster.inside_options:
            local = local_evidence_by_option.get(opt_id)
            rad_deg = getattr(local, "radial_degrees_covered", None) if local else None
            rad_pix = getattr(local, "radial_ink_pixels", None) if local else None
            # Compute signed distance to hull if possible
            dist = None
            if getattr(cluster, "hull_points", None):
                hull_np = np.array(cluster.hull_points, dtype=np.int32)
                # find option center (scaled to crop coordinates)
                opt_center = None
                for r in rois:
                    if r.option_id == opt_id:
                        cx = (r.bbox.x + r.bbox.w / 2.0) - global_topology.crop["x1"]
                        cy = (r.bbox.y + r.bbox.h / 2.0) - global_topology.crop["y1"]
                        opt_center = (cx, cy)
                        break
                if opt_center is not None:
                    dist = float(cv2.pointPolygonTest(hull_np, opt_center, True))
            selected = opt_id in global_topology.global_marked
            # Simple risk aggregation
            risk = 0.0
            if rad_deg is not None and rad_deg < cfg.blank_local_degrees:
                risk += 0.4
            if shape_class != "GROUP_ENCLOSURE" and selected:
                risk += 0.3
            option_diag = OptionGlobalShapeDiagnostic(
                question_id=question_id,
                option_id=opt_id,
                cluster_id=cluster.cluster_id,
                selected_by_current_global=selected,
                local_radial_degrees=rad_deg,
                local_radial_ink_pixels=rad_pix,
                center_signed_distance_px=dist,
                option_risk_score=risk,
                recommended_action=action if should_vlm else "KEEP_GLOBAL",
                reasons=(),
            )
            option_diags.append(option_diag)
    return GlobalShapeDiagnostics(
        validator_mode="diagnostic",
        config=cfg,
        clusters=tuple(cluster_diags),
        options=tuple(option_diags),
    )

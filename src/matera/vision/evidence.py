import json
import dataclasses
from dataclasses import dataclass
from typing import Any, Literal
from pathlib import Path
import numpy as np

from matera.vision.contracts import MarkScore


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
    legacy_score: float
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


def evidence_to_mark_scores(evidence: PageMarkEvidence, *, debug_dir: str | Path | None = None) -> list[MarkScore]:
    """Convert the structured PageMarkEvidence into legacy MarkScore objects."""
    scores = []
    
    for question in evidence.questions:
        for opt in question.option_evidence:
            scores.append(MarkScore(
                question_id=opt.question_id,
                option_id=opt.option_id,
                score=opt.legacy_score,
                strategy=opt.strategy,
                method=opt.legacy_method,
                image_crop=None,
                evidence_path=None
            ))
            
    return scores


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
    min_ink_per_bin: int = 2
) -> LocalOptionEvidence:
    import cv2
    import numpy as np
    
    bx, by, bw, bh = text_bbox
    h, w = mask_raw.shape
    cx, cy = w / 2.0, h / 2.0
    
    bx = max(0, bx - 2)
    by = max(0, by - 2)
    bw = bw + 4
    bh = bh + 4
    
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - cx)**2 + (Y - cy)**2
    outer_mask = dist_sq > outer_radius**2
    
    core_mask = np.zeros((h, w), dtype=bool)
    by_e = min(h, by+bh)
    bx_e = min(w, bx+bw)
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

    crop_dict = {"x1": crop_coords[0], "y1": crop_coords[1], "x2": crop_coords[2], "y2": crop_coords[3]}
    text_bbox_dict = {"x": text_bbox[0], "y": text_bbox[1], "w": text_bbox[2], "h": text_bbox[3]}
        
    return LocalOptionEvidence(
        crop_coords=crop_dict,
        diff_ink_pixels=0,
        diff_contour_count=0,
        diff_contour_areas=(),
        text_bbox=text_bbox_dict,
        core_mask_pixels=core_mask_pixels,
        outer_mask_pixels=outer_mask_pixels,
        radial_mask_before_morph_px=radial_mask_before_morph_px,
        radial_ink_pixels=radial_ink_pixels,
        radial_active_bin_count=active_bin_count,
        radial_degrees_covered=degrees_covered,
        radial_histogram=tuple(hist_counts.tolist())
    )

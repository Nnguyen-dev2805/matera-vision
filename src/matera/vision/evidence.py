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

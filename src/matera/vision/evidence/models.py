from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from matera.core.layout import BoundingBox


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
    shape_diagnostics: Any | None = None


@dataclass(frozen=True)
class HsvFallbackEvidence:
    method_prefix: str
    hsv_ink_pixels: int | None
    classifier_probability: float | None
    classifier_available: bool
    decision: Literal["MARKED", "BLANK", "AMBIGUOUS"]
    method: str


@dataclass(frozen=True)
class CheckboxComponentEvidence:
    component_id: int
    area: int
    bbox: BoundingBox
    centroid: tuple[float, float]
    touches_border: bool
    touches_left: bool
    touches_right: bool
    touches_top: bool
    touches_bottom: bool
    aspect_ratio: float
    diagonal_span: float
    horizontal_span: int
    vertical_span: int
    width: int
    height: int
    orientation_deg: float
    interior_pixel_count: int
    border_pixel_count: int


@dataclass(frozen=True)
class CheckboxStrokeEvidence:
    crop_width: int
    crop_height: int
    safe_mask_pixels: int
    final_hsv_pixels: int
    diff_mask_pixels: int
    color_mask_pixels: int
    reference_border_pixels: int
    border_residue_pixels: int
    border_suppressed_pixels: int
    interior_ink_pixels: int
    border_touch_ratio: float | None
    interior_ink_ratio: float | None
    largest_component: CheckboxComponentEvidence | None
    components: tuple[CheckboxComponentEvidence, ...]
    validator_features: dict[str, Any]
    suspicion_notes: tuple[str, ...]


@dataclass(frozen=True)
class CheckboxDecision:
    decision: Literal["MARKED", "BLANK", "NEED_REVIEW"]
    confidence: float
    reason_code: str
    reason: str
    evidence: CheckboxStrokeEvidence


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
    checkbox_stroke: CheckboxStrokeEvidence | None = None
    checkbox_decision: CheckboxDecision | None = None
    vlm: "VlmOptionEvidence | None" = None


@dataclass(frozen=True)
class QuestionMarkEvidence:
    question_id: str
    strategy: str
    response_type: str
    rois: tuple[RoiEvidence, ...]
    global_topology: GlobalTopologyEvidence | None
    option_evidence: tuple[OptionMarkEvidence, ...]
    vlm: "VlmQuestionEvidence | None" = None


@dataclass(frozen=True)
class PageMarkEvidence:
    page_number: int
    form_id: str
    form_version: str
    alignment: AlignmentEvidence
    reference: ReferenceEvidence
    questions: tuple[QuestionMarkEvidence, ...]
    thresholds: MarkThresholdEvidence


@dataclass(frozen=True)
class VlmOptionEvidence:
    option_id: str
    decision: Literal["MARKED", "BLANK", "NEED_REVIEW"]
    raw_state: str | None
    reason: str | None
    parse_error: str | None


@dataclass(frozen=True)
class VlmQuestionEvidence:
    question_id: str
    provider: str
    model: str
    prompt_version: str
    crop_box: dict[str, int]
    crop_path: str | None
    raw_response_path: str | None
    provider_error: str | None
    latency_ms: float | None
    options: tuple[VlmOptionEvidence, ...]

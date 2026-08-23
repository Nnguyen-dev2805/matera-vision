from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration for the page alignment process."""

    algorithm: Literal["orb"] = "orb"
    transform_model: Literal["similarity", "affine"] = "affine"
    inlier_threshold: float = 0.6
    max_features: int = 5000
    reference_dpi: int = 300
    target_width: int = 0
    target_height: int = 0

    def __post_init__(self):
        if not (0.0 <= self.inlier_threshold <= 1.0):
            raise ValueError("inlier_threshold must be between 0.0 and 1.0")
        if self.max_features <= 0:
            raise ValueError("max_features must be > 0")
        if self.target_width < 0 or self.target_height < 0:
            raise ValueError("target dimensions must be non-negative")


@dataclass(frozen=True)
class AlignedPage:
    """A single aligned page mapped to the layout's coordinate system."""

    page_number: int
    image: Image.Image
    profile_form_id: str
    profile_version: str
    reference_dpi: int
    warp_matrix: np.ndarray
    alignment_score: float
    debug_evidence_path: Path | None = None


class AlignmentError(Exception):
    """Raised when page alignment fails or falls below the acceptable threshold."""

    def __init__(
        self, message: str, page_number: int | None = None, alignment_score: float | None = None
    ) -> None:
        super().__init__(message)
        self.page_number = page_number
        self.alignment_score = alignment_score


@dataclass(frozen=True)
class ROIFeature:
    """Interpretable, deterministic geometric and pixel measurements for a single ROI."""

    dark_pixel_ratio: float
    foreground_area_ratio: float
    contour_count: int
    largest_component_ratio: float
    bbox_fill_ratio: float


@dataclass(frozen=True)
class MarkScore:
    """The normalized mark score and corresponding features for an evaluated option."""

    question_id: str
    option_id: str
    score: float
    strategy: str
    method: str
    features: ROIFeature
    evidence_path: Path | None = None

    def __post_init__(self):
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Mark score must be between 0.0 and 1.0, got {self.score}")


@dataclass(frozen=True)
class MarkScoringConfig:
    """Configuration for template-difference mark scoring."""

    # Difference threshold (0-255) to consider a pixel as 'ink' vs 'background'
    diff_threshold: int = 30

    # Kernel size for morphological operations to clean up alignment noise
    morph_kernel_size: tuple[int, int] = (3, 3)

    # The multiplier for foreground_area_ratio to map to a 0.0-1.0 score for circle/checkbox
    area_score_multiplier: float = 10.0


@dataclass(frozen=True)
class RoutingConfig:
    """Configuration for decision routing and ambiguity thresholds."""

    low_threshold: float = 0.2
    high_threshold: float = 0.6

    def __post_init__(self):
        if not (0.0 <= self.low_threshold < self.high_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0.0 <= low < high <= 1.0, "
                f"got low={self.low_threshold}, high={self.high_threshold}"
            )

import dataclasses
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class GlobalCropConfig:
    base_pad: int = 20
    expansion_step_px: int = 20
    max_pad_px: int = 60
    edge_band_px: int = 4
    edge_ink_threshold_px: int = 20
    edge_contour_min_area: float = 30.0
    max_iterations: int = 5

@dataclass(frozen=True)
class EdgeInkStats:
    left_px: int
    right_px: int
    top_px: int
    bottom_px: int
    left_touching_contours: int
    right_touching_contours: int
    top_touching_contours: int
    bottom_touching_contours: int

@dataclass(frozen=True)
class GlobalCropIteration:
    iteration: int
    crop: dict[str, int]
    pads: dict[str, int]
    edge_ink: EdgeInkStats
    expanded_directions: list[str]
    stop_reason: str | None

@dataclass(frozen=True)
class GlobalCropResult:
    crop: dict[str, int]
    pads: dict[str, int]
    iterations: list[GlobalCropIteration]
    expanded: bool
    stop_reason: str

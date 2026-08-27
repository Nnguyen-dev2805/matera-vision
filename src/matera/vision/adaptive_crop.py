import dataclasses
from dataclasses import dataclass


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


import cv2
import numpy as np
from PIL import Image


def compute_adaptive_global_crop(
    aligned_image_rgb: Image.Image,
    median_ref_bgr: np.ndarray,
    rois: list,
    config: GlobalCropConfig | None = None,
) -> GlobalCropResult:
    if config is None:
        config = GlobalCropConfig()

    min_x = min(roi.bbox.x for roi in rois)
    min_y = min(roi.bbox.y for roi in rois)
    max_x = max(roi.bbox.x + roi.bbox.w for roi in rois)
    max_y = max(roi.bbox.y + roi.bbox.h for roi in rois)

    img_w, img_h = aligned_image_rgb.size

    pads = {
        "left": config.base_pad,
        "right": config.base_pad,
        "top": config.base_pad,
        "bottom": config.base_pad,
    }

    iterations = []
    expanded = False
    stop_reason = None

    from matera.vision.mark import DIFF_THRESHOLD, GAUSS_KERNEL

    img_bgr = cv2.cvtColor(np.array(aligned_image_rgb), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(median_ref_bgr, cv2.COLOR_BGR2GRAY)

    for i in range(config.max_iterations):
        crop_x1 = max(0, min_x - pads["left"])
        crop_y1 = max(0, min_y - pads["top"])
        crop_x2 = min(img_w, max_x + pads["right"])
        crop_y2 = min(img_h, max_y + pads["bottom"])

        crop_dict = {"x1": crop_x1, "y1": crop_y1, "x2": crop_x2, "y2": crop_y2}

        target_crop = img_gray[crop_y1:crop_y2, crop_x1:crop_x2]
        ref_crop = ref_gray[crop_y1:crop_y2, crop_x1:crop_x2]

        diff = cv2.absdiff(target_crop, ref_crop)
        blurred = cv2.GaussianBlur(diff, GAUSS_KERNEL, 0)
        _, mask = cv2.threshold(blurred, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        h, w = mask.shape
        if h > 30:
            mask[:15, :] = 0
            mask[h - 15 :, :] = 0

        edge_band = config.edge_band_px

        left_px = int(np.count_nonzero(mask[:, :edge_band]))
        right_px = int(np.count_nonzero(mask[:, -edge_band:])) if w > edge_band else 0
        top_px = int(np.count_nonzero(mask[:edge_band, :]))
        bottom_px = int(np.count_nonzero(mask[-edge_band:, :])) if h > edge_band else 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= config.edge_contour_min_area]

        left_touch = right_touch = top_touch = bottom_touch = 0
        for c in valid_contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if x < edge_band:
                left_touch += 1
            if x + cw > w - edge_band:
                right_touch += 1
            if y < edge_band:
                top_touch += 1
            if y + ch > h - edge_band:
                bottom_touch += 1

        edge_ink = EdgeInkStats(
            left_px=left_px,
            right_px=right_px,
            top_px=top_px,
            bottom_px=bottom_px,
            left_touching_contours=left_touch,
            right_touching_contours=right_touch,
            top_touching_contours=top_touch,
            bottom_touching_contours=bottom_touch,
        )

        expanded_dirs = []

        if (
            (left_px >= config.edge_ink_threshold_px or left_touch > 0)
            and pads["left"] < config.max_pad_px
            and crop_x1 > 0
        ):
            expanded_dirs.append("left")
            pads["left"] = min(pads["left"] + config.expansion_step_px, config.max_pad_px)

        if (
            (right_px >= config.edge_ink_threshold_px or right_touch > 0)
            and pads["right"] < config.max_pad_px
            and crop_x2 < img_w
        ):
            expanded_dirs.append("right")
            pads["right"] = min(pads["right"] + config.expansion_step_px, config.max_pad_px)

        if (
            (top_px >= config.edge_ink_threshold_px or top_touch > 0)
            and pads["top"] < config.max_pad_px
            and crop_y1 > 0
        ):
            expanded_dirs.append("top")
            pads["top"] = min(pads["top"] + config.expansion_step_px, config.max_pad_px)

        if (
            (bottom_px >= config.edge_ink_threshold_px or bottom_touch > 0)
            and pads["bottom"] < config.max_pad_px
            and crop_y2 < img_h
        ):
            expanded_dirs.append("bottom")
            pads["bottom"] = min(pads["bottom"] + config.expansion_step_px, config.max_pad_px)

        iteration = GlobalCropIteration(
            iteration=i,
            crop=crop_dict,
            pads=pads.copy(),
            edge_ink=edge_ink,
            expanded_directions=expanded_dirs,
            stop_reason=None,
        )
        iterations.append(iteration)

        if expanded_dirs:
            expanded = True
        else:
            # Check why we stopped
            has_pressure = any(
                [
                    left_px >= config.edge_ink_threshold_px or left_touch > 0,
                    right_px >= config.edge_ink_threshold_px or right_touch > 0,
                    top_px >= config.edge_ink_threshold_px or top_touch > 0,
                    bottom_px >= config.edge_ink_threshold_px or bottom_touch > 0,
                ]
            )
            if not has_pressure:
                stop_reason = "no_edge_pressure"
            else:
                at_boundary = True
                if (left_px >= config.edge_ink_threshold_px or left_touch > 0) and crop_x1 > 0:
                    at_boundary = False
                if (
                    right_px >= config.edge_ink_threshold_px or right_touch > 0
                ) and crop_x2 < img_w:
                    at_boundary = False
                if (top_px >= config.edge_ink_threshold_px or top_touch > 0) and crop_y1 > 0:
                    at_boundary = False
                if (
                    bottom_px >= config.edge_ink_threshold_px or bottom_touch > 0
                ) and crop_y2 < img_h:
                    at_boundary = False

                if at_boundary:
                    stop_reason = "image_boundary_reached"
                else:
                    stop_reason = "max_pad_reached"
            break

    else:
        stop_reason = "max_iterations_reached"

    iterations[-1] = dataclasses.replace(iterations[-1], stop_reason=stop_reason)

    crop_x1 = max(0, min_x - pads["left"])
    crop_y1 = max(0, min_y - pads["top"])
    crop_x2 = min(img_w, max_x + pads["right"])
    crop_y2 = min(img_h, max_y + pads["bottom"])
    final_crop = {"x1": crop_x1, "y1": crop_y1, "x2": crop_x2, "y2": crop_y2}

    return GlobalCropResult(
        crop=final_crop,
        pads=pads,
        iterations=iterations,
        expanded=expanded,
        stop_reason=stop_reason,
    )

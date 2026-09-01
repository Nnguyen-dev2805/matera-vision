from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from matera.vision.mark_thresholds import (
    CHECKBOX_INNER_MARGIN,
    DIFF_THRESHOLD,
    GAUSS_KERNEL,
    HSV_SATURATION_THRESHOLD,
)

if TYPE_CHECKING:
    from PIL import Image

    from matera.vision.contracts import AlignedPage
    from matera.vision.evidence.models import CheckboxDecision, CheckboxStrokeEvidence


def compute_hsv_safe_mask(target_bgr, ref_crop, inner_margin: int):
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(ref_gray, target_gray)
    blurred = cv2.GaussianBlur(diff, GAUSS_KERNEL, 0)
    _, mask_diff = cv2.threshold(blurred, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask_color = cv2.threshold(saturation, HSV_SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)

    mask_final = cv2.bitwise_and(mask_diff, mask_color)

    h, w = mask_final.shape
    safe_mask = np.zeros_like(mask_final)
    excluded_mask = np.zeros_like(mask_final)

    if h > 2 * inner_margin and w > 2 * inner_margin:
        safe_mask[inner_margin : h - inner_margin, inner_margin : w - inner_margin] = mask_final[
            inner_margin : h - inner_margin, inner_margin : w - inner_margin
        ]
        # Excluded is final - safe
        excluded_mask = cv2.bitwise_xor(mask_final, safe_mask)
    else:
        excluded_mask = mask_final.copy()

    return safe_mask, mask_final, mask_diff, mask_color, excluded_mask


@dataclass
class HsvAiResult:
    legacy_prediction: str
    legacy_method: str
    prob_mark: float | None
    stroke_ev: "CheckboxStrokeEvidence | None"
    checkbox_decision: "CheckboxDecision | None"


def get_crop_bgr(aligned_image_rgb: Image.Image, median_ref_bgr: np.ndarray, bbox, pad: int = 0):
    crop_x1 = max(0, bbox.x - pad)
    crop_y1 = max(0, bbox.y - pad)
    crop_x2 = min(aligned_image_rgb.width, bbox.x + bbox.w + pad)
    crop_y2 = min(aligned_image_rgb.height, bbox.y + bbox.h + pad)

    crop_img = aligned_image_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    target_bgr = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
    ref_crop = median_ref_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

    return target_bgr, ref_crop, (crop_x1, crop_y1, crop_x2, crop_y2)


def evaluate_safe_mask_decision(
    ink_pixels: int, method_prefix: str
) -> tuple[str | None, str | None]:
    if ink_pixels < 20:
        return "BLANK", f"{method_prefix}_L2"
    elif ink_pixels > 200:
        return "MARKED", f"{method_prefix}_L2"
    return None, None


def get_local_roi_crops_hsv(
    aligned_image_rgb: Image.Image, median_ref_bgr: np.ndarray, bbox, pad: int
):
    target_bgr, ref_crop, coords = get_crop_bgr(aligned_image_rgb, median_ref_bgr, bbox, pad)
    safe_mask, mask_final, _, _, _ = compute_hsv_safe_mask(target_bgr, ref_crop, 0)
    return target_bgr, mask_final, coords


def process_roi_hsv(
    aligned_page: "AlignedPage", median_ref_bgr: np.ndarray, roi, method_prefix: str
) -> HsvAiResult:
    target_bgr, ref_crop, _ = get_crop_bgr(aligned_page.image, median_ref_bgr, roi.bbox, pad=0)

    inner_margin = CHECKBOX_INNER_MARGIN
    is_q14 = "Q14" in roi.question_id

    safe_mask, mask_final, mask_diff, mask_color, excluded = compute_hsv_safe_mask(
        target_bgr, ref_crop, inner_margin
    )
    ink_pixels = int(np.sum(safe_mask > 0))

    pred, method = evaluate_safe_mask_decision(ink_pixels, method_prefix)
    prob_mark = None

    if pred is None or method is None:
        pred, method = "AMBIGUOUS", f"{method_prefix}_HSV_UNCERTAIN"

    stroke_ev = None
    dec = None
    if is_q14:
        from matera.vision.checkbox_validator import (
            compute_checkbox_masks,
            compute_checkbox_stroke_evidence,
            resolve_checkbox_state,
        )

        masks = compute_checkbox_masks(
            target_bgr=target_bgr,
            ref_crop_bgr=ref_crop,
            safe_mask=safe_mask,
            final_hsv_mask=mask_final,
            diff_mask=mask_diff,
            color_mask=mask_color,
        )
        stroke_ev = compute_checkbox_stroke_evidence(masks)
        dec = resolve_checkbox_state(
            stroke_ev,
            hsv_decision=pred,
            hsv_method=method,
            classifier_probability=prob_mark,
        )

    return HsvAiResult(
        legacy_prediction=pred,
        legacy_method=method,
        prob_mark=prob_mark,
        stroke_ev=stroke_ev,
        checkbox_decision=dec,
    )

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from matera.core.layout import RoiDef
from matera.vision.mark import CHECKBOX_INNER_MARGIN, GAUSS_KERNEL, HSV_SATURATION_THRESHOLD


@dataclass
class Q14CheckboxDiagnostic:
    page_number: int
    question_id: str
    option_id: str
    expected_state: str | None
    actual_state: str
    score: float
    method: str
    routing_status: str | None
    bbox: dict[str, int]
    crop_coords: dict[str, int]
    diff_mask_pixels: int
    saturation_mask_pixels: int
    hsv_final_mask_pixels: int
    safe_mask_pixels: int
    excluded_margin_pixels: int
    safe_mask_ratio: float | None
    hsv_to_diff_ratio: float | None
    classifier_available: bool
    classifier_probability: float | None
    decision_reason: str
    suspicion_notes: list[str]
    artifacts: dict[str, str]


def compute_q14_checkbox_diagnostic(
    *,
    aligned_image: Image.Image,
    reference_image: Image.Image,
    median_ref_bgr: np.ndarray,
    roi: RoiDef,
    trace: Any | None = None,
    expected_state: str | None = None,
    output_dir: Path | None = None,
) -> Q14CheckboxDiagnostic:
    # 1. Crops
    pad = 0
    bbox = roi.bbox
    crop_x1 = max(0, bbox.x - pad)
    crop_y1 = max(0, bbox.y - pad)
    crop_x2 = min(aligned_image.width, bbox.x + bbox.w + pad)
    crop_y2 = min(aligned_image.height, bbox.y + bbox.h + pad)

    crop_img = aligned_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    target_bgr = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
    ref_crop = median_ref_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

    # 2. Masks
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(ref_gray, target_gray)
    blurred = cv2.GaussianBlur(diff, GAUSS_KERNEL, 0)
    _, mask_diff = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask_color = cv2.threshold(saturation, HSV_SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)

    mask_final = cv2.bitwise_and(mask_diff, mask_color)

    # 3. Safe Mask
    h, w = mask_final.shape
    safe_mask = np.zeros_like(mask_final)
    m = CHECKBOX_INNER_MARGIN
    if h > 2 * m and w > 2 * m:
        safe_mask[m : h - m, m : w - m] = mask_final[m : h - m, m : w - m]

    # 4. Metrics
    diff_mask_pixels = int(np.sum(mask_diff > 0))
    saturation_mask_pixels = int(np.sum(mask_color > 0))
    hsv_final_mask_pixels = int(np.sum(mask_final > 0))
    safe_mask_pixels = int(np.sum(safe_mask > 0))
    excluded_margin_pixels = hsv_final_mask_pixels - safe_mask_pixels

    safe_mask_ratio = safe_mask_pixels / hsv_final_mask_pixels if hsv_final_mask_pixels > 0 else 0.0
    hsv_to_diff_ratio = hsv_final_mask_pixels / diff_mask_pixels if diff_mask_pixels > 0 else 0.0

    # 5. Decision Logic (mirroring production)
    classifier_available = False
    classifier_probability = None
    if trace:
        classifier_available = trace.classifier_available
        classifier_probability = trace.classifier_probability

    if safe_mask_pixels < 20:
        actual_state = "BLANK"
        decision_reason = f"BLANK (safe_mask_pixels={safe_mask_pixels} < 20)"
    elif safe_mask_pixels > 200:
        actual_state = "MARKED"
        decision_reason = f"MARKED (safe_mask_pixels={safe_mask_pixels} > 200)"
    else:
        actual_state = trace.prediction if trace else "UNKNOWN"
        decision_reason = f"CLASSIFIER (20 <= safe_mask_pixels={safe_mask_pixels} <= 200)"

    # Suspicion notes
    notes = []
    if expected_state and actual_state != expected_state:
        notes.append(f"expected {expected_state} but actual {actual_state}")
    if safe_mask_pixels > 200 and hsv_final_mask_pixels > safe_mask_pixels * 2:
        notes.append("safe mask is high but final HSV pixels hug printed checkbox border")
    if safe_mask_pixels > 200 and saturation_mask_pixels > diff_mask_pixels * 2:
        notes.append("safe mask is high while saturation-only pixels greatly exceed diff pixels")
    if 180 < safe_mask_pixels < 220:
        notes.append("safe mask is near threshold")

    # Write artifacts if output_dir provided
    artifacts = {}
    if output_dir:
        opt_dir = output_dir / "Q14" / roi.option_id
        opt_dir.mkdir(parents=True, exist_ok=True)

        def save_img(name, img_array):
            cv2.imwrite(str(opt_dir / name), img_array)
            artifacts[name] = str(Path("Q14") / roi.option_id / name)

        save_img("q14_crop_aligned.png", target_bgr)
        save_img("q14_crop_reference.png", ref_crop)
        save_img("q14_diff_mask.png", mask_diff)
        save_img("q14_saturation_mask.png", mask_color)
        save_img("q14_hsv_final_mask.png", mask_final)
        save_img("q14_safe_mask.png", safe_mask)

        # Overlays
        safe_mask_overlay = target_bgr.copy()
        safe_mask_overlay[safe_mask > 0] = [0, 255, 0]  # Green for safe
        save_img("q14_safe_mask_overlay.png", safe_mask_overlay)

        decision_overlay = target_bgr.copy()
        # Red for diff-only (not color)
        diff_only = cv2.bitwise_and(mask_diff, cv2.bitwise_not(mask_color))
        decision_overlay[diff_only > 0] = [0, 0, 255]
        # Blue for color-only
        color_only = cv2.bitwise_and(mask_color, cv2.bitwise_not(mask_diff))
        decision_overlay[color_only > 0] = [255, 0, 0]
        # Yellow for excluded margin
        margin_mask = cv2.bitwise_and(mask_final, cv2.bitwise_not(safe_mask))
        decision_overlay[margin_mask > 0] = [0, 255, 255]
        # Green for safe mask
        decision_overlay[safe_mask > 0] = [0, 255, 0]
        save_img("q14_decision_overlay.png", decision_overlay)

    return Q14CheckboxDiagnostic(
        page_number=trace.page_number if trace else 1,
        question_id=roi.question_id,
        option_id=roi.option_id,
        expected_state=expected_state,
        actual_state=actual_state,
        score=trace.score if trace else 0.0,
        method="HSV_AI_L2",
        routing_status=None,
        bbox={"x": bbox.x, "y": bbox.y, "w": bbox.w, "h": bbox.h},
        crop_coords={"x1": crop_x1, "y1": crop_y1, "x2": crop_x2, "y2": crop_y2},
        diff_mask_pixels=diff_mask_pixels,
        saturation_mask_pixels=saturation_mask_pixels,
        hsv_final_mask_pixels=hsv_final_mask_pixels,
        safe_mask_pixels=safe_mask_pixels,
        excluded_margin_pixels=excluded_margin_pixels,
        safe_mask_ratio=safe_mask_ratio,
        hsv_to_diff_ratio=hsv_to_diff_ratio,
        classifier_available=classifier_available,
        classifier_probability=classifier_probability,
        decision_reason=decision_reason,
        suspicion_notes=notes,
        artifacts=artifacts,
    )

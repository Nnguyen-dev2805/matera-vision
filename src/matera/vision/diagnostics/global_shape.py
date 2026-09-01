import dataclasses

import cv2
import numpy as np

from matera.vision.global_shape_validator import (
    GlobalShapeValidatorConfig,
    diagnose_global_shape_candidates,
)
from matera.vision.mark_thresholds import DIFF_THRESHOLD, GAUSS_KERNEL


def attach_global_shape_diagnostics(
    q_id: str,
    gt_evidence,
    aligned_page,
    median_ref_bgr: np.ndarray,
    rois: list,
    local_evidence_by_option: dict,
    marked_threshold_deg: float,
    blank_threshold_deg: float,
):
    if gt_evidence is None or not gt_evidence.ran or gt_evidence.skip_reason is not None:
        return gt_evidence

    crop = gt_evidence.crop
    crop_img = aligned_page.image.crop((crop["x1"], crop["y1"], crop["x2"], crop["y2"]))
    target_bgr_crop = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
    ref_crop = median_ref_bgr[crop["y1"] : crop["y2"], crop["x1"] : crop["x2"]]
    diff_crop = cv2.absdiff(
        cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(target_bgr_crop, cv2.COLOR_BGR2GRAY),
    )
    blurred_crop = cv2.GaussianBlur(diff_crop, GAUSS_KERNEL, 0)
    _, global_mask_raw = cv2.threshold(blurred_crop, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    diagnostics = diagnose_global_shape_candidates(
        question_id=q_id,
        rois=rois,
        global_topology=gt_evidence,
        local_evidence_by_option=local_evidence_by_option,
        mask_raw=global_mask_raw,
        config=GlobalShapeValidatorConfig(
            strong_local_degrees=marked_threshold_deg,
            blank_local_degrees=blank_threshold_deg,
        ),
    )
    return dataclasses.replace(gt_evidence, shape_diagnostics=diagnostics)

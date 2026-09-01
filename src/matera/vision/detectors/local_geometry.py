from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

from matera.vision.scoring.thresholds import (
    DIFF_THRESHOLD,
    GAUSS_KERNEL,
)

if TYPE_CHECKING:
    from PIL import Image


def get_text_bounding_box(ref_crop_gray: np.ndarray) -> tuple[int, int, int, int]:
    _, thresh = cv2.threshold(ref_crop_gray, 200, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    h, w = ref_crop_gray.shape
    cx, cy = w / 2.0, h / 2.0
    min_dist_to_center = float("inf")
    best_component_stats = None

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        centroid_x, centroid_y = centroids[i]

        if x <= 1 or y <= 1 or (x + bw) >= w - 1 or (y + bh) >= h - 1:
            continue
        aspect_ratio = max(bw / float(bh), bh / float(bw))
        if aspect_ratio > 4.0:
            continue

        dist_to_center = math.sqrt((centroid_x - cx) ** 2 + (centroid_y - cy) ** 2)
        if dist_to_center < min_dist_to_center:
            min_dist_to_center = dist_to_center
            best_component_stats = (x, y, bw, bh)

    if best_component_stats is not None:
        return best_component_stats
    else:
        return (int(cx - 7), int(cy - 7), 14, 14)


def get_local_roi_crops(
    aligned_image_rgb: Image.Image, median_ref_bgr: np.ndarray, bbox, pad: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], np.ndarray]:
    crop_x1 = max(0, bbox.x - pad)
    crop_y1 = max(0, bbox.y - pad)
    crop_x2 = min(aligned_image_rgb.width, bbox.x + bbox.w + pad)
    crop_y2 = min(aligned_image_rgb.height, bbox.y + bbox.h + pad)

    crop_img = aligned_image_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    target_bgr = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
    ref_crop = median_ref_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(ref_gray, target_gray)
    blurred = cv2.GaussianBlur(diff, GAUSS_KERNEL, 0)
    _, mask_raw = cv2.threshold(blurred, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    return target_bgr, mask_raw, (crop_x1, crop_y1, crop_x2, crop_y2), ref_gray

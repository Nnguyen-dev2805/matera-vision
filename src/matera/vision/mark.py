from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from matera.vision.contracts import MarkScore

if TYPE_CHECKING:
    from matera.classifier.model import AmbiguityClassifier
    from matera.core.layout import PageLayout
    from matera.core.profile import FormProfile
    from matera.vision.contracts import AlignedPage

import math

from scipy.spatial.distance import cdist

# --- V17 Geometric Configs ---
NUM_BINS = 72
DEGREES_PER_BIN = 360 / NUM_BINS
MIN_INK_PER_BIN = 2
MARKED_THRESHOLD_DEG = 180
BLANK_THRESHOLD_DEG = 90
LOCAL_PAD = 15
OUTER_RADIUS = 32
GLOBAL_PAD = 20
MERGE_THRESHOLD = 15.0
EXTREMES_REJECT_THRESHOLD = 50.0
CHECKBOX_INNER_MARGIN = 5
HSV_SATURATION_THRESHOLD = 40

DIFF_THRESHOLD = 30
GAUSS_KERNEL = (3, 3)
CLOSE_KERNEL_SIZE = 5
CLOSE_ITERATIONS = 2
LOCAL_CLOSE_KERNEL = 3
LOCAL_CLOSE_ITERS = 1


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def min_contour_distance(cnt1: np.ndarray, cnt2: np.ndarray) -> float:
    pts1 = cnt1.reshape(-1, 2)
    pts2 = cnt2.reshape(-1, 2)
    dists = cdist(pts1, pts2, metric="euclidean")
    return float(np.min(dists))


def get_horizontal_extremes(cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = cnt.reshape(-1, 2)
    left_pt = pts[np.argmin(pts[:, 0])]
    right_pt = pts[np.argmax(pts[:, 0])]
    return left_pt, right_pt


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


def run_v11_global_topology(
    aligned_image_rgb: Image.Image,
    median_ref_bgr: np.ndarray,
    rois: list,
    full_mask_bgr: np.ndarray,
) -> set[str]:
    from matera.vision.evidence import compute_global_topology_evidence

    evidence = compute_global_topology_evidence(aligned_image_rgb, median_ref_bgr, rois)

    # Draw debug hulls as before
    if evidence.ran and evidence.skip_reason is None:
        import cv2
        import numpy as np

        crop_x1 = evidence.crop["x1"]
        crop_y1 = evidence.crop["y1"]
        for cl in evidence.clusters:
            if cl.hull_area > 1000 and cl.solidity < 0.4:
                hull_pts = np.array(
                    [[[pt[0] + crop_x1, pt[1] + crop_y1]] for pt in cl.hull_points], dtype=np.int32
                )
                cv2.drawContours(full_mask_bgr, [hull_pts], 0, (0, 255, 255), 2)

    return set(evidence.global_marked)


def get_local_roi_crops_hsv(
    aligned_image_rgb: Image.Image, median_ref_bgr: np.ndarray, bbox, pad: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
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
    _, mask_diff = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask_color = cv2.threshold(saturation, HSV_SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)

    mask_final = cv2.bitwise_and(mask_diff, mask_color)

    return target_bgr, mask_final, (crop_x1, crop_y1, crop_x2, crop_y2)


def process_roi_hsv_ai(
    aligned_page: "AlignedPage", median_ref_bgr: np.ndarray, roi, method_prefix: str
) -> tuple[str, str]:
    from matera.classifier.model import extract_hog_features

    _, mask_hsv, _ = get_local_roi_crops_hsv(aligned_page.image, median_ref_bgr, roi.bbox, 0)
    h, w = mask_hsv.shape
    safe_mask = np.zeros_like(mask_hsv)
    m = CHECKBOX_INNER_MARGIN
    if h > 2 * m and w > 2 * m:
        safe_mask[m : h - m, m : w - m] = mask_hsv[m : h - m, m : w - m]

    ink_pixels = np.sum(safe_mask > 0)

    if ink_pixels < 20:
        return "BLANK", f"{method_prefix}_L2"
    elif ink_pixels > 200:
        return "MARKED", f"{method_prefix}_L2"
    else:
        feat = extract_hog_features(safe_mask)
        try:
            clf = _get_classifier()
        except FileNotFoundError:
            # Fallback to simple threshold if model missing
            return "AMBIGUOUS", f"{method_prefix}_L2_NO_AI"
        prob_mark = clf.predict_proba([feat])[0]
        if prob_mark > 0.85:
            return "MARKED", f"{method_prefix}_AI ({prob_mark:.2f})"
        elif prob_mark < 0.15:
            return "BLANK", f"{method_prefix}_AI ({prob_mark:.2f})"
        else:
            return "AMBIGUOUS", f"{method_prefix}_AI ({prob_mark:.2f})"


# --- Global Classifier Instance (Lazy loaded) ---
_clf = None


def _get_classifier() -> "AmbiguityClassifier":
    global _clf
    if _clf is None:
        model_path = Path("models/shape_classifier.pkl")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        from matera.classifier.model import AmbiguityClassifier

        _clf = AmbiguityClassifier.load(str(model_path))
    return _clf


def extract_mark_scores(
    aligned_page: AlignedPage,
    profile: FormProfile,
    layout: PageLayout,
    reference_image: Image.Image,
    debug_dir: str | None = None,
    debug_full_mask: np.ndarray | None = None,
) -> list[MarkScore]:
    from matera.vision.evidence import evidence_to_mark_scores, extract_mark_evidence

    evidence = extract_mark_evidence(
        aligned_page,
        profile,
        layout,
        reference_image,
        debug_dir=debug_dir,
        debug_full_mask=debug_full_mask,
    )

    return evidence_to_mark_scores(evidence, aligned_page, debug_dir=debug_dir)

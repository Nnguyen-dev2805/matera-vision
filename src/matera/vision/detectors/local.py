import cv2
import numpy as np

from matera.vision.evidence.models import LocalOptionEvidence


def compute_local_option_evidence(
    mask_raw: np.ndarray,
    text_bbox: tuple[int, int, int, int],
    crop_coords: tuple[int, int, int, int],
    outer_radius: int = 32,
    num_bins: int = 72,
    min_ink_per_bin: int = 2,
) -> LocalOptionEvidence:

    bx, by, bw, bh = text_bbox
    h, w = mask_raw.shape
    cx, cy = w / 2.0, h / 2.0

    bx = max(0, bx - 2)
    by = max(0, by - 2)
    bw = bw + 4
    bh = bh + 4

    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
    outer_mask = dist_sq > outer_radius**2

    core_mask = np.zeros((h, w), dtype=bool)
    by_e = min(h, by + bh)
    bx_e = min(w, bx + bw)
    core_mask[by:by_e, bx:bx_e] = True

    mask_radial = mask_raw.copy()
    mask_radial[core_mask] = 0
    mask_radial[outer_mask] = 0

    radial_mask_before_morph_px = int(np.sum(mask_radial > 0))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask_radial, kernel, iterations=1)
    mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

    ys, xs = np.where(mask_closed > 0)
    radial_ink_pixels = len(xs)

    degrees_covered = 0.0
    active_bin_count = 0
    hist_counts = np.zeros(num_bins, dtype=int)
    degrees_per_bin = 360.0 / num_bins

    if radial_ink_pixels > 0:
        dx = xs.astype(float) - cx
        dy = ys.astype(float) - cy
        angles = np.arctan2(dy, dx)
        angles = np.degrees(angles) % 360
        hist_counts, _ = np.histogram(angles, bins=num_bins, range=(0, 360))
        active_bins = (hist_counts >= min_ink_per_bin).astype(int)
        for i in range(num_bins):
            left = active_bins[(i - 1) % num_bins]
            right = active_bins[(i + 1) % num_bins]
            if active_bins[i] == 0 and left == 1 and right == 1:
                active_bins[i] = 1
        active_bin_count = int(np.sum(active_bins))
        degrees_covered = float(active_bin_count * degrees_per_bin)

    core_mask_pixels = int(np.sum(core_mask))
    outer_mask_pixels = int(np.sum(outer_mask))

    crop_dict = {
        "x1": crop_coords[0],
        "y1": crop_coords[1],
        "x2": crop_coords[2],
        "y2": crop_coords[3],
    }
    text_bbox_dict = {
        "x": text_bbox[0],
        "y": text_bbox[1],
        "w": text_bbox[2],
        "h": text_bbox[3],
    }

    diff_ink_pixels = int(np.sum(mask_raw > 0))
    cnts, _ = cv2.findContours(mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diff_contour_count = len(cnts)
    diff_contour_areas = tuple(float(cv2.contourArea(c)) for c in cnts)

    return LocalOptionEvidence(
        crop_coords=crop_dict,
        diff_ink_pixels=diff_ink_pixels,
        diff_contour_count=diff_contour_count,
        diff_contour_areas=diff_contour_areas,
        text_bbox=text_bbox_dict,
        core_mask_pixels=core_mask_pixels,
        outer_mask_pixels=outer_mask_pixels,
        radial_mask_before_morph_px=radial_mask_before_morph_px,
        radial_ink_pixels=radial_ink_pixels,
        radial_active_bin_count=active_bin_count,
        radial_degrees_covered=degrees_covered,
        radial_histogram=tuple(hist_counts.tolist()),
    )

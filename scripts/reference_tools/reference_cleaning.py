from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class InpaintConfig:
    radius: int = 3
    method: str = "TELEA"  # TELEA or NS


def load_image_bgr(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def save_image_bgr(bgr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path)


def inpaint_reference(
    image_bgr: np.ndarray, mask: np.ndarray, config: InpaintConfig | None = None
) -> np.ndarray:
    """Inpaint only masked pixels, preserve unmasked exactly."""
    if config is None:
        config = InpaintConfig()
    if mask is None or np.count_nonzero(mask) == 0:
        return image_bgr.copy()
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255
    flags = cv2.INPAINT_TELEA if config.method == "TELEA" else cv2.INPAINT_NS
    inpainted = cv2.inpaint(image_bgr, mask, config.radius, flags)
    result = image_bgr.copy()
    result[mask > 0] = inpainted[mask > 0]
    return result


def fill_paper_with_nearest_neighbor(
    image_bgr: np.ndarray, mask: np.ndarray, min_bright: int = 200
) -> np.ndarray:
    """Fill masked pixels with the nearest bright (paper) neighbor.

    For each masked pixel, locate the closest pixel outside the mask with
    gray > min_bright and copy its BGR value. If no such paper pixel exists
    anywhere outside the mask, fall back to flat 250.

    Args:
        image_bgr: BGR image, untouched outside mask.
        mask: 8-bit mask, >0 = fill region.
        min_bright: minimum grayscale value to consider a "paper" neighbor.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return image_bgr.copy()
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    paper_mask = (mask == 0) & (gray > min_bright)
    if not np.any(paper_mask):
        result = image_bgr.copy()
        result[mask > 0] = np.array([250, 250, 250], dtype=np.uint8)
        return result

    # distance transform: each pixel gets distance to nearest True pixel
    # We use paper_mask as the "foreground" for distance transform so each
    # pixel's value = distance to nearest paper pixel.
    inv_paper = (~paper_mask).astype(np.uint8)
    dist, idx = cv2.distanceTransformWithLabels(
        inv_paper, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    # for masked pixels, nearest paper label is given by idx; sample BGR
    h, w = image_bgr.shape[:2]
    labels = idx  # already indexed per pixel
    flat_paper = paper_mask.reshape(-1)
    flat_labels = labels.reshape(-1)
    # build lookup: for each label, pick BGR from one paper pixel with that label
    # (any paper pixel with the label is the source for that label's region)
    label_to_color = {}
    flat_bgr = image_bgr.reshape(-1, 3)
    paper_indices = np.where(flat_paper)[0]
    for idx_flat in paper_indices:
        lab = int(flat_labels[idx_flat])
        if lab not in label_to_color:
            label_to_color[lab] = flat_bgr[idx_flat]

    result = image_bgr.copy()
    flat_result = result.reshape(-1, 3)
    masked_indices = np.where(mask.reshape(-1) > 0)[0]
    for idx_flat in masked_indices:
        lab = int(flat_labels[idx_flat])
        if lab in label_to_color:
            flat_result[idx_flat] = label_to_color[lab]
    return result


def fill_paper_with_sample_color(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    sample_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Fill masked pixels with the median BGR color from a user-selected box."""
    if mask is None or np.count_nonzero(mask) == 0:
        return image_bgr.copy()
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8)

    h, w = image_bgr.shape[:2]
    x, y, bw, bh = sample_box
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w, x1 + max(1, int(bw)))
    y2 = min(h, y1 + max(1, int(bh)))
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid sample box: {sample_box}")

    sample = image_bgr[y1:y2, x1:x2]
    sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    paper_pixels = sample[sample_gray > 160]
    if paper_pixels.size == 0:
        paper_pixels = sample.reshape(-1, 3)
    fill_color = np.median(paper_pixels.reshape(-1, 3), axis=0).astype(np.uint8)

    result = image_bgr.copy()
    result[mask > 0] = fill_color
    return result


def inpaint_image_file(
    reference_path: Path, mask_path: Path, output_path: Path, config: InpaintConfig | None = None
) -> Path:
    ref_bgr = load_image_bgr(reference_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if mask.shape[0] != ref_bgr.shape[0] or mask.shape[1] != ref_bgr.shape[1]:
        mask = cv2.resize(
            mask, (ref_bgr.shape[1], ref_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    cleaned = inpaint_reference(ref_bgr, mask, config)
    save_image_bgr(cleaned, output_path)
    return output_path


def fill_paper_image_file(
    reference_path: Path, mask_path: Path, output_path: Path, min_bright: int = 200
) -> Path:
    ref_bgr = load_image_bgr(reference_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if mask.shape[0] != ref_bgr.shape[0] or mask.shape[1] != ref_bgr.shape[1]:
        mask = cv2.resize(
            mask, (ref_bgr.shape[1], ref_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    cleaned = fill_paper_with_nearest_neighbor(ref_bgr, mask, min_bright=min_bright)
    save_image_bgr(cleaned, output_path)
    return output_path


def detect_printed_structure_touch(mask: np.ndarray, rois: list[dict], tol: int = 3) -> list[dict]:
    """Return ROIs whose mask pixels touch the ROI bbox (likely printed structure painted)."""
    warnings = []
    for roi in rois:
        b = roi["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        # pixels near border of roi (within tol)
        near = np.zeros_like(mask, dtype=bool)
        # left/right strips
        near[:, max(0, x - tol) : x] = True
        near[:, x + w : min(mask.shape[1], x + w + tol)] = True
        # top/bottom strips
        near[max(0, y - tol) : y, :] = True
        near[y + h : min(mask.shape[0], y + h + tol), :] = True
        touched = int(np.sum(mask[near] > 0))
        if touched > 0:
            warnings.append(
                {
                    "question_id": roi["question_id"],
                    "option_id": roi["option_id"],
                    "touched_px": touched,
                }
            )
    return warnings


def generate_auto_mask_for_roi(
    ref_crop_bgr: np.ndarray,
    is_checkbox: bool = False,
) -> np.ndarray:
    """Return single-channel mask for one ROI crop. Color ink -> 255, background/border -> 0."""
    # Convert to HSV and look for colored ink (high saturation)
    hsv = cv2.cvtColor(ref_crop_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    # colored ink: saturation > 40 and value not too dark/bright
    _, s_mask = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)
    # For checkbox interior, erode border: keep only interior 60% to avoid printed border
    h, w = s_mask.shape
    if is_checkbox:
        # keep interior, shave 4px border
        margin = 4
        border_mask = np.zeros_like(s_mask)
        if h > 2 * margin and w > 2 * margin:
            border_mask[margin : h - margin, margin : w - margin] = 255
            s_mask = cv2.bitwise_and(s_mask, border_mask)
        # filter small noise < 10px
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(s_mask, connectivity=8)
        filtered = np.zeros_like(s_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 10:
                filtered[labels == i] = 255
        return filtered
    else:
        # circle ROI: keep components that look like strokes inside/bubble, but preserve printed circle  # noqa: E501
        # For minimal slice, same as above but without interior shave, just size filter
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(s_mask, connectivity=8)
        filtered = np.zeros_like(s_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # keep small/medium colored components (10 to 500 px) typical for ink residue
            if 10 <= area <= 800:
                filtered[labels == i] = 255
        return filtered


def generate_auto_mask(
    reference_path: Path,
    rois: list[dict],
    is_checkbox_fn=None,
) -> np.ndarray:
    """Generate full-image suspected-ink mask by processing each ROI.
    rois: list of {question_id, option_id, bbox{x,y,w,h}}
    Returns uint8 mask same size as reference.
    """
    ref_bgr = load_image_bgr(reference_path)
    h, w = ref_bgr.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)
    for roi in rois:
        bbox = roi["bbox"] if "bbox" in roi else roi
        x, y, rw, rh = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
        # small padding to catch ink slightly outside bbox
        pad = 3
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + rw + pad)
        y2 = min(h, y + rh + pad)
        crop = ref_bgr[y1:y2, x1:x2]
        is_cb = False
        if is_checkbox_fn:
            is_cb = is_checkbox_fn(roi)
        else:
            # heuristic: Q14 is checkbox
            qid = roi.get("question_id", "")
            is_cb = qid == "Q14"
        roi_mask = generate_auto_mask_for_roi(crop, is_checkbox=is_cb)
        # paste back
        full_mask[y1:y2, x1:x2] = cv2.bitwise_or(full_mask[y1:y2, x1:x2], roi_mask)
    return full_mask


def mask_pixels_by_roi(mask: np.ndarray, rois: list[dict]) -> list[dict]:
    """Count mask pixels per ROI. Returns list of {question_id, option_id, mask_pixels, bbox}."""
    result = []
    for roi in rois:
        bbox = roi["bbox"]
        x, y, rw, rh = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
        h, w = mask.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + rw)
        y2 = min(h, y + rh)
        crop = mask[y1:y2, x1:x2]
        cnt = int(np.count_nonzero(crop))
        result.append(
            {
                "question_id": roi["question_id"],
                "option_id": roi["option_id"],
                "mask_pixels": cnt,
                "bbox": bbox,
            }
        )
    return result


def build_clean_manifest(
    source_reference: Path,
    output_dir: Path,
    image_size: dict,
    inpaint: dict,
    auto_mask: dict,
    manual_edits: dict,
    roi_summary: list[dict],
    profile_dir: str,
    page_number: int,
    save_basename: str = "cleaned_reference",
) -> dict:
    return {
        "schema_version": 1,
        "source_reference": str(source_reference),
        "cleaned_reference": f"{save_basename}.png",
        "clean_mask": "clean_mask.png",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool_version": "reference_cleaner_v1",
        "profile_dir": str(profile_dir),
        "page_number": page_number,
        "image_size": image_size,
        "inpaint": inpaint,
        "auto_mask": auto_mask,
        "manual_edits": manual_edits,
        "roi_summary": roi_summary,
    }


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure 8-bit
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    cv2.imwrite(str(path), mask)


def load_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return m

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from matera.vision.contracts import MarkScore, MarkScoringConfig, ROIFeature

if TYPE_CHECKING:
    from matera.core.layout import PageLayout
    from matera.core.profile import FormProfile
    from matera.vision.contracts import AlignedPage


def create_mark_map(
    source_patch: Image.Image,
    reference_patch: Image.Image,
    config: MarkScoringConfig | None = None,
) -> Image.Image:
    """
    Creates a binary mask isolating handwritten marks by subtracting the reference template.

    Args:
        source_patch: Crop of the ROI from the aligned source page.
        reference_patch: Crop of the ROI from the pristine reference page.

    Returns:
        A binary Image (mode "L") where 255 represents potential handwritten marks
        and 0 represents background or printed text.
    """
    if config is None:
        config = MarkScoringConfig()

    if source_patch.size != reference_patch.size:
        raise ValueError("Source and reference patches must have the same dimensions.")

    # Convert to grayscale NumPy arrays
    src_gray = cv2.cvtColor(np.array(source_patch), cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.cvtColor(np.array(reference_patch), cv2.COLOR_RGB2GRAY)

    # Calculate absolute difference
    diff = cv2.absdiff(src_gray, ref_gray)

    # Threshold the difference to create a binary mask.
    _, binary_mask = cv2.threshold(diff, config.diff_threshold, 255, cv2.THRESH_BINARY)

    # Apply morphology to clean up minor alignment noise around printed text edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.morph_kernel_size)

    # 1. Opening: remove small isolated noise dots (false positives from slight misalignment)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # 2. Closing: fill small holes inside actual pen strokes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(cleaned, mode="L")


def calculate_features(source_patch: Image.Image, mark_map: Image.Image) -> ROIFeature:
    """
    Computes deterministic geometric and pixel measurements from the source crop and mark map.
    """

    src_gray = cv2.cvtColor(np.array(source_patch), cv2.COLOR_RGB2GRAY)
    mask = np.array(mark_map)

    total_pixels = mask.size
    if total_pixels == 0:
        return ROIFeature(0.0, 0.0, 0, 0.0, 0.0)

    # dark_pixel_ratio: Percentage of raw pixels that are very dark (e.g., < 100 on 0-255 scale)
    # This is a fallback feature that doesn't rely on the reference template
    dark_pixels = np.count_nonzero(src_gray < 100)
    dark_pixel_ratio = float(dark_pixels / total_pixels)

    # foreground_area_ratio: Percentage of pixels identified as handwriting by the mark map
    foreground_pixels = np.count_nonzero(mask)
    foreground_area_ratio = float(foreground_pixels / total_pixels)

    # Find contours in the binary mark map
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    largest_component_ratio = 0.0
    bbox_fill_ratio = 0.0

    if contour_count > 0:
        # largest_component_ratio: Area of the largest connected handwriting stroke
        max_contour_area = max(cv2.contourArea(c) for c in contours)
        largest_component_ratio = float(max_contour_area / total_pixels)

        # bbox_fill_ratio: Density of the marks within their own bounding box
        # Find the bounding box that encompasses all contours
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        bbox_area = w * h
        if bbox_area > 0:
            bbox_fill_ratio = float(foreground_pixels / bbox_area)

    return ROIFeature(
        dark_pixel_ratio=dark_pixel_ratio,
        foreground_area_ratio=foreground_area_ratio,
        contour_count=contour_count,
        largest_component_ratio=largest_component_ratio,
        bbox_fill_ratio=bbox_fill_ratio,
    )


def normalize_score(
    features: ROIFeature, strategy: str, config: MarkScoringConfig | None = None
) -> float:
    """
    Maps the raw ROI features to a 0.0 - 1.0 confidence score based on the strategy.
    """
    if config is None:
        config = MarkScoringConfig()

    if strategy not in ("circle", "checkbox", "rating"):
        raise ValueError(f"Unsupported mark strategy: {strategy}")

    # Currently MVP shares the same scoring logic for all mark types, relying purely on template diff
    # Future iterations will branch on `strategy` for custom feature weighting.
    base_score = features.foreground_area_ratio * config.area_score_multiplier

    # Cap between 0.0 and 1.0
    return min(1.0, max(0.0, base_score))


def extract_mark_scores(
    aligned_page: "AlignedPage",
    profile: "FormProfile",
    layout: "PageLayout",
    reference_image: Image.Image,
    debug_dir: str | None = None,
) -> list[MarkScore]:
    """
    Extracts features and normalizes a score for every ROI.
    Resolves the mark strategy using both the FormProfile and PageLayout.
    """
    import pathlib

    scores = []

    # Pre-compute a strategy lookup from FormProfile
    strategy_map = {}
    for q in profile.questions:
        strategy_map[q.question_id] = q.mark_strategy

    if debug_dir:
        debug_path = pathlib.Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

    if aligned_page.image.size != reference_image.size:
        raise ValueError("Aligned page and reference image dimensions must match.")

    for roi in layout.rois:
        # Resolve strategy: Layout override > Profile default
        strategy = roi.mark_strategy_override or strategy_map.get(roi.question_id)
        if not strategy:
            raise ValueError(f"Cannot resolve mark strategy for ROI question_id={roi.question_id}")

        # Crop the patches
        bbox = (roi.bbox.x, roi.bbox.y, roi.bbox.x + roi.bbox.w, roi.bbox.y + roi.bbox.h)
        src_crop = aligned_page.image.crop(bbox)
        ref_crop = reference_image.crop(bbox)

        # Compute map and features
        mark_map = create_mark_map(src_crop, ref_crop)
        features = calculate_features(src_crop, mark_map)

        # Normalize score
        score = normalize_score(features, strategy)

        evidence_path = None
        if debug_dir:
            # Save the crop and mask side-by-side for debugging
            combo = Image.new("RGB", (roi.bbox.w * 2, roi.bbox.h))
            combo.paste(src_crop, (0, 0))
            # Convert mask back to RGB for pasting
            mask_rgb = mark_map.convert("RGB")
            combo.paste(mask_rgb, (roi.bbox.w, 0))

            evidence_file = (
                debug_path
                / f"page_{aligned_page.page_number}_{roi.question_id}_{roi.option_id}.png"
            )
            combo.save(evidence_file)
            evidence_path = evidence_file

        scores.append(
            MarkScore(
                question_id=roi.question_id,
                option_id=roi.option_id,
                score=score,
                strategy=strategy,
                method="template_difference",
                features=features,
                evidence_path=evidence_path,
            )
        )

    return scores

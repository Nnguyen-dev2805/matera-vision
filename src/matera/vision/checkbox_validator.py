from dataclasses import dataclass

import cv2
import numpy as np

from matera.core.layout import BoundingBox
from matera.vision.evidence.models import (
    CheckboxComponentEvidence,
    CheckboxDecision,
    CheckboxStrokeEvidence,
)

CHECKBOX_BORDER_DILATE_PX = 3
CHECKBOX_INTERIOR_MARGIN_PX = 6
CHECKBOX_MIN_INTERIOR_PIXELS = 15


@dataclass
class CheckboxMasks:
    safe_mask: np.ndarray
    final_hsv_mask: np.ndarray
    diff_mask: np.ndarray
    color_mask: np.ndarray
    reference_border_mask: np.ndarray
    border_residue_mask: np.ndarray
    border_suppressed_mask: np.ndarray
    interior_ink_mask: np.ndarray
    interior_region_mask: np.ndarray
    labels: np.ndarray


def compute_checkbox_masks(
    target_bgr: np.ndarray,
    ref_crop_bgr: np.ndarray,
    safe_mask: np.ndarray,
    final_hsv_mask: np.ndarray,
    diff_mask: np.ndarray,
    color_mask: np.ndarray,
) -> CheckboxMasks:
    ref_gray = cv2.cvtColor(ref_crop_bgr, cv2.COLOR_BGR2GRAY)
    _, ref_thresh = cv2.threshold(ref_gray, 200, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ref_thresh, connectivity=8
    )
    h, w = ref_thresh.shape

    # We want to suppress ALL ink present in the reference template,
    # primarily the printed checkbox itself. Since crops can be loose,
    # the printed checkbox doesn't always touch the edge.
    reference_border_mask = ref_thresh.copy()

    kernel = np.ones(
        (CHECKBOX_BORDER_DILATE_PX * 2 + 1, CHECKBOX_BORDER_DILATE_PX * 2 + 1), np.uint8
    )
    reference_border_mask_dilated = cv2.dilate(reference_border_mask, kernel, iterations=1)

    border_residue_mask = cv2.bitwise_and(safe_mask, reference_border_mask_dilated)
    border_suppressed_mask = cv2.bitwise_and(
        safe_mask, cv2.bitwise_not(reference_border_mask_dilated)
    )

    interior_region_mask = np.zeros_like(safe_mask)
    if h > 2 * CHECKBOX_INTERIOR_MARGIN_PX and w > 2 * CHECKBOX_INTERIOR_MARGIN_PX:
        interior_region_mask[
            CHECKBOX_INTERIOR_MARGIN_PX : h - CHECKBOX_INTERIOR_MARGIN_PX,
            CHECKBOX_INTERIOR_MARGIN_PX : w - CHECKBOX_INTERIOR_MARGIN_PX,
        ] = 255

    interior_ink_mask = cv2.bitwise_and(border_suppressed_mask, interior_region_mask)

    _, labels = cv2.connectedComponents(interior_ink_mask, connectivity=8)

    return CheckboxMasks(
        safe_mask=safe_mask,
        final_hsv_mask=final_hsv_mask,
        diff_mask=diff_mask,
        color_mask=color_mask,
        reference_border_mask=reference_border_mask_dilated,
        border_residue_mask=border_residue_mask,
        border_suppressed_mask=border_suppressed_mask,
        interior_ink_mask=interior_ink_mask,
        interior_region_mask=interior_region_mask,
        labels=labels,
    )


def compute_checkbox_stroke_evidence(masks: CheckboxMasks) -> CheckboxStrokeEvidence:
    h, w = masks.safe_mask.shape

    # We re-run connectedComponentsWithStats to get stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        masks.interior_ink_mask, connectivity=8
    )

    components = []
    largest_component = None
    max_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        comp_mask = (labels == i).astype(np.uint8) * 255

        touches_left = x <= CHECKBOX_INTERIOR_MARGIN_PX
        touches_top = y <= CHECKBOX_INTERIOR_MARGIN_PX
        touches_right = (x + comp_w) >= w - CHECKBOX_INTERIOR_MARGIN_PX
        touches_bottom = (y + comp_h) >= h - CHECKBOX_INTERIOR_MARGIN_PX
        touches_border = bool(np.any(cv2.bitwise_and(comp_mask, masks.reference_border_mask) > 0))

        width = int(comp_w)
        height = int(comp_h)
        orientation_deg = 0.0

        # Fit minAreaRect if enough pixels
        pts = cv2.findNonZero(comp_mask)
        if pts is not None and len(pts) > 4:
            rect = cv2.minAreaRect(pts)
            orientation_deg = float(rect[2])

        interior_pixel_count = int(
            np.sum(cv2.bitwise_and(comp_mask, masks.interior_region_mask) > 0)
        )
        border_pixel_count = int(
            np.sum(cv2.bitwise_and(comp_mask, masks.reference_border_mask) > 0)
        )

        comp = CheckboxComponentEvidence(
            component_id=i,
            area=int(area),
            bbox=BoundingBox(x=int(x), y=int(y), w=int(comp_w), h=int(comp_h)),
            centroid=(float(cx), float(cy)),
            touches_border=touches_border,
            touches_left=bool(touches_left),
            touches_right=bool(touches_right),
            touches_top=bool(touches_top),
            touches_bottom=bool(touches_bottom),
            aspect_ratio=float(comp_w) / float(comp_h) if comp_h > 0 else 0.0,
            diagonal_span=float(np.sqrt(comp_w**2 + comp_h**2)),
            horizontal_span=int(comp_w),
            vertical_span=int(comp_h),
            width=width,
            height=height,
            orientation_deg=orientation_deg,
            interior_pixel_count=interior_pixel_count,
            border_pixel_count=border_pixel_count,
        )
        components.append(comp)

        if area > max_area:
            max_area = int(area)
            largest_component = comp

    safe_mask_pixels = int(np.sum(masks.safe_mask > 0))

    evidence = CheckboxStrokeEvidence(
        crop_width=w,
        crop_height=h,
        safe_mask_pixels=safe_mask_pixels,
        final_hsv_pixels=int(np.sum(masks.final_hsv_mask > 0)),
        diff_mask_pixels=int(np.sum(masks.diff_mask > 0)),
        color_mask_pixels=int(np.sum(masks.color_mask > 0)),
        reference_border_pixels=int(np.sum(masks.reference_border_mask > 0)),
        border_residue_pixels=int(np.sum(masks.border_residue_mask > 0)),
        border_suppressed_pixels=int(np.sum(masks.border_suppressed_mask > 0)),
        interior_ink_pixels=int(np.sum(masks.interior_ink_mask > 0)),
        border_touch_ratio=(float(np.sum(masks.border_residue_mask > 0)) / float(safe_mask_pixels))
        if safe_mask_pixels > 0
        else None,
        interior_ink_ratio=(float(np.sum(masks.interior_ink_mask > 0)) / float(safe_mask_pixels))
        if safe_mask_pixels > 0
        else None,
        largest_component=largest_component,
        components=tuple(components),
        validator_features={
            "largest_area": largest_component.area if largest_component else 0,
            "interior_ink_pixels": int(np.sum(masks.interior_ink_mask > 0)),
        },
        suspicion_notes=tuple(),
    )

    return evidence


def resolve_checkbox_state(
    evidence: CheckboxStrokeEvidence,
    *,
    hsv_decision: str,
    hsv_method: str,
    classifier_probability: float | None,
) -> CheckboxDecision:

    if evidence.reference_border_pixels < 20:
        return CheckboxDecision(
            decision=hsv_decision,
            confidence=0.5,
            reason_code="insufficient_reference_border",
            reason="Reference border is too weak or missing, defaulting to HSV",
            evidence=evidence,
        )

    if evidence.interior_ink_pixels < CHECKBOX_MIN_INTERIOR_PIXELS:
        # Classifier guard: must not override strong BLANK
        return CheckboxDecision(
            decision="BLANK",
            confidence=1.0,
            reason_code="low_interior_ink",
            reason=f"Interior ink ({evidence.interior_ink_pixels}) < MIN_INTERIOR",
            evidence=evidence,
        )

    is_border_dominant = bool(evidence.border_touch_ratio and evidence.border_touch_ratio > 0.8)
    weak_interior = not evidence.largest_component or evidence.largest_component.area < 20

    if is_border_dominant and weak_interior:
        return CheckboxDecision(
            decision="BLANK",
            confidence=0.9,
            reason_code="border_residue_dominant",
            reason=f"Border touch ratio {evidence.border_touch_ratio:.2f} > 0.8 with weak interior",
            evidence=evidence,
        )

    if (
        evidence.largest_component
        and evidence.largest_component.area > 30
        and evidence.largest_component.diagonal_span > 10
    ):
        comp = evidence.largest_component
        comp_w = comp.bbox.w
        comp_h = comp.bbox.h
        min_dim = min(comp_w, comp_h)
        max_dim = max(comp_w, comp_h)
        aspect = max_dim / min_dim if min_dim > 0 else 999.0

        # Thin solid lines (min_dim ≤ 4, aspect ≥ 4) are border bleed artifacts
        # that survived dilation, not real checkmark strokes.
        if min_dim <= 4 and aspect >= 4.0:
            return CheckboxDecision(
                decision="NEED_REVIEW",
                confidence=0.6,
                reason_code="thin_edge_artifact",
                reason=f"Thin edge artifact (min_dim={min_dim}, aspect={aspect:.1f})",
                evidence=evidence,
            )

        return CheckboxDecision(
            decision="MARKED",
            confidence=0.9,
            reason_code="strong_interior_stroke",
            reason=f"Strong interior stroke (area {comp.area})",
            evidence=evidence,
        )

    # Classifier can tip the scale if not border-dominated
    if classifier_probability is not None and not is_border_dominant:
        if classifier_probability > 0.8:
            return CheckboxDecision(
                decision="MARKED",
                confidence=0.7,
                reason_code="classifier_supported_stroke",
                reason=f"Classifier supports MARKED (prob {classifier_probability:.2f}) on ambig",
                evidence=evidence,
            )
        elif classifier_probability < 0.2:
            return CheckboxDecision(
                decision="BLANK",
                confidence=0.7,
                reason_code="classifier_supported_stroke",
                reason=f"Classifier supports BLANK (prob {classifier_probability:.2f}) on ambig",
                evidence=evidence,
            )

    return CheckboxDecision(
        decision="NEED_REVIEW",
        confidence=0.5,
        reason_code="ambiguous_overlap_band",
        reason="Mixed evidence, requires review",
        evidence=evidence,
    )

"""Tests for resolve_checkbox_state decision logic."""

from matera.core.layout import BoundingBox
from matera.vision.checkbox_validator import resolve_checkbox_state
from matera.vision.evidence import (
    CheckboxComponentEvidence,
    CheckboxStrokeEvidence,
)


def _make_component(
    *,
    area: int = 100,
    w: int = 20,
    h: int = 20,
    touches_border: bool = False,
) -> CheckboxComponentEvidence:
    """Helper to create a minimal CheckboxComponentEvidence."""
    import math

    return CheckboxComponentEvidence(
        component_id=1,
        area=area,
        bbox=BoundingBox(x=10, y=10, w=w, h=h),
        centroid=(10 + w / 2.0, 10 + h / 2.0),
        touches_border=touches_border,
        touches_left=False,
        touches_right=False,
        touches_top=False,
        touches_bottom=False,
        aspect_ratio=float(w) / float(h) if h > 0 else 0.0,
        diagonal_span=math.sqrt(w**2 + h**2),
        horizontal_span=w,
        vertical_span=h,
        width=w,
        height=h,
        orientation_deg=0.0,
        interior_pixel_count=area,
        border_pixel_count=0,
    )


def _make_evidence(
    *,
    interior_ink_pixels: int = 100,
    reference_border_pixels: int = 500,
    border_touch_ratio: float | None = 0.5,
    interior_ink_ratio: float | None = 0.3,
    largest_component: CheckboxComponentEvidence | None = None,
) -> CheckboxStrokeEvidence:
    """Helper to create a minimal CheckboxStrokeEvidence."""
    return CheckboxStrokeEvidence(
        crop_width=50,
        crop_height=50,
        safe_mask_pixels=300,
        final_hsv_pixels=400,
        diff_mask_pixels=350,
        color_mask_pixels=200,
        reference_border_pixels=reference_border_pixels,
        border_residue_pixels=150,
        border_suppressed_pixels=150,
        interior_ink_pixels=interior_ink_pixels,
        border_touch_ratio=border_touch_ratio,
        interior_ink_ratio=interior_ink_ratio,
        largest_component=largest_component,
        components=(largest_component,) if largest_component else (),
        validator_features={
            "largest_area": largest_component.area if largest_component else 0,
            "interior_ink_pixels": interior_ink_pixels,
        },
        suspicion_notes=(),
    )


class TestResolveCheckboxState:
    """Tests for the resolve_checkbox_state decision function."""

    def test_low_interior_ink_returns_blank(self):
        """Interior ink below MIN_INTERIOR_PIXELS -> BLANK."""
        evidence = _make_evidence(interior_ink_pixels=5)
        result = resolve_checkbox_state(
            evidence,
            hsv_decision="MARKED",
            hsv_method="HSV_AI_L2",
            classifier_probability=0.9,
        )
        assert result.decision == "BLANK"
        assert result.reason_code == "low_interior_ink"

    def test_classifier_cannot_override_strong_blank(self):
        """Even a high classifier probability must not override low interior ink."""
        evidence = _make_evidence(interior_ink_pixels=10)
        result = resolve_checkbox_state(
            evidence,
            hsv_decision="MARKED",
            hsv_method="HSV_AI_L2",
            classifier_probability=0.99,
        )
        assert result.decision == "BLANK"
        assert result.reason_code == "low_interior_ink"

    def test_insufficient_reference_border_falls_back_to_hsv(self):
        """Weak reference border -> fallback to legacy HSV decision."""
        evidence = _make_evidence(reference_border_pixels=10)
        result = resolve_checkbox_state(
            evidence,
            hsv_decision="MARKED",
            hsv_method="HSV_AI_L2",
            classifier_probability=None,
        )
        assert result.decision == "MARKED"
        assert result.reason_code == "insufficient_reference_border"

    def test_insufficient_reference_border_preserves_blank(self):
        """Weak reference border -> fallback preserves BLANK too."""
        evidence = _make_evidence(reference_border_pixels=5)
        result = resolve_checkbox_state(
            evidence,
            hsv_decision="BLANK",
            hsv_method="HSV_AI_L2",
            classifier_probability=None,
        )
        assert result.decision == "BLANK"
        assert result.reason_code == "insufficient_reference_border"

    def test_strong_interior_stroke_returns_marked(self):
        """Large interior component with sufficient span -> MARKED."""
        comp = _make_component(area=150, w=20, h=25)
        evidence = _make_evidence(
            interior_ink_pixels=200,
            largest_component=comp,
        )
        result = resolve_checkbox_state(
            evidence,
            hsv_decision="MARKED",
            hsv_method="HSV_AI_L2",
            classifier_probability=None,
        )
        assert result.decision == "MARKED"
        assert result.reason_code == "strong_interior_stroke"

    def test_thin_edge_artifact_returns_need_review(self):
        """Thin solid line (min_dim <= 4, aspect >= 4) -> NEED_REVIEW."""
        # 3 pixels wide, 32 pixels tall -> classic border bleed artifact
        comp = _make_component(area=85, w=3, h=32)
        evidence = _make_evidence(
            interior_ink_pixels=100,
            largest_component=comp,
        )
        result = resolve_checkbox_state(
            evidence,
            hsv_decision="MARKED",
            hsv_method="HSV_AI_L2",
            classifier_probability=None,
        )
        assert result.decision == "NEED_REVIEW"
        assert result.reason_code == "thin_edge_artifact"

    def test_thin_horizontal_artifact_returns_need_review(self):
        """Thin horizontal line also caught as artifact."""
        comp = _make_component(area=48, w=16, h=3)
        evidence = _make_evidence(
            interior_ink_pixels=77,
            largest_component=comp,
        )
        result = resolve_checkbox_state(
            evidence,
            hsv_decision="MARKED",
            hsv_method="HSV_AI_L2",
            classifier_probability=None,
        )
        assert result.decision == "NEED_REVIEW"
        assert result.reason_code == "thin_edge_artifact"

    def test_border_dominant_weak_interior_returns_blank(self):
        """High border touch ratio + weak interior -> BLANK."""
        comp = _make_component(area=15, w=5, h=5)
        evidence = _make_evidence(
            interior_ink_pixels=20,
            border_touch_ratio=0.85,
            largest_component=comp,
        )
        result = resolve_checkbox_state(
            evidence,
            hsv_decision="MARKED",
            hsv_method="HSV_AI_L2",
            classifier_probability=None,
        )
        assert result.decision == "BLANK"
        assert result.reason_code == "border_residue_dominant"

    def test_ambiguous_overlap_band_returns_need_review(self):
        """No strong signal in any direction -> NEED_REVIEW."""
        comp = _make_component(area=25, w=8, h=8)
        evidence = _make_evidence(
            interior_ink_pixels=30,
            border_touch_ratio=0.5,
            largest_component=comp,
        )
        result = resolve_checkbox_state(
            evidence,
            hsv_decision="AMBIGUOUS",
            hsv_method="HSV_AI_L2",
            classifier_probability=0.5,
        )
        assert result.decision == "NEED_REVIEW"
        assert result.reason_code == "ambiguous_overlap_band"

    def test_classifier_supports_marked_on_ambiguous(self):
        """High classifier probability on ambiguous evidence -> MARKED."""
        comp = _make_component(area=25, w=8, h=8)
        evidence = _make_evidence(
            interior_ink_pixels=30,
            border_touch_ratio=0.5,
            largest_component=comp,
        )
        result = resolve_checkbox_state(
            evidence,
            hsv_decision="AMBIGUOUS",
            hsv_method="HSV_AI_L2",
            classifier_probability=0.9,
        )
        assert result.decision == "MARKED"
        assert result.reason_code == "classifier_supported_stroke"

    def test_classifier_supports_blank_on_ambiguous(self):
        """Low classifier probability on ambiguous evidence -> BLANK."""
        comp = _make_component(area=25, w=8, h=8)
        evidence = _make_evidence(
            interior_ink_pixels=30,
            border_touch_ratio=0.5,
            largest_component=comp,
        )
        result = resolve_checkbox_state(
            evidence,
            hsv_decision="AMBIGUOUS",
            hsv_method="HSV_AI_L2",
            classifier_probability=0.1,
        )
        assert result.decision == "BLANK"
        assert result.reason_code == "classifier_supported_stroke"

    def test_genuine_stroke_not_caught_by_thin_filter(self):
        """A real checkmark stroke (7x26, area 111) must NOT be caught as thin artifact."""
        comp = _make_component(area=111, w=7, h=26)
        evidence = _make_evidence(
            interior_ink_pixels=222,
            largest_component=comp,
        )
        result = resolve_checkbox_state(
            evidence,
            hsv_decision="MARKED",
            hsv_method="HSV_AI_L2",
            classifier_probability=None,
        )
        assert result.decision == "MARKED"
        assert result.reason_code == "strong_interior_stroke"

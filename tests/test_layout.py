from pathlib import Path

import pytest

from matera.core.errors import ProfileValidationError
from matera.core.layout import (
    AnchorDef,
    BoundingBox,
    LayoutProfile,
    PageLayout,
    RoiDef,
    load_layout_profile,
)
from matera.core.profile import FormProfile, OptionDef, QuestionDef


def test_bbox_valid():
    bbox = BoundingBox(10, 20, 30, 40)
    assert bbox.x == 10


def test_bbox_invalid():
    with pytest.raises(ValueError, match="w and h must be positive ints"):
        BoundingBox(0, 0, 0, 10)
    with pytest.raises(ValueError, match="x and y must be ints"):
        BoundingBox(1.5, 0, 10, 10)  # type: ignore


def test_layout_profile_valid(tmp_path: Path):
    json_data = """{
        "form_id": "matera-pre",
        "form_version": "v1",
        "coordinate_space": "absolute_pixel",
        "reference_dpi": 200,
        "pages": [
            {
                "page_number": 1,
                "width_px": 1000,
                "height_px": 2000,
                "anchors": [
                    {"anchor_id": "a1", "anchor_type": "qr", "bbox": {"x": 10, "y": 10, "w": 50, "h": 50}}
                ],
                "rois": [
                    {"question_id": "Q1", "option_id": "opt1", "bbox": {"x": 100, "y": 100, "w": 20, "h": 20}}
                ]
            }
        ]
    }"""
    p = tmp_path / "layout.json"
    p.write_text(json_data, encoding="utf-8")

    semantic = FormProfile(
        form_id="matera-pre",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="Q1",
                response_type="single_select",
                mark_strategy="checkbox",
                options=(OptionDef("opt1"),),
                max_selections=1,
            ),
        ),
    )

    layout = load_layout_profile(p, semantic)
    assert layout.form_id == "matera-pre"
    assert layout.reference_dpi == 200
    assert len(layout.pages) == 1
    page1 = layout.pages[0]
    assert page1.width_px == 1000
    assert len(page1.anchors) == 1
    assert page1.anchors[0].anchor_id == "a1"
    assert len(page1.rois) == 1
    assert page1.rois[0].question_id == "Q1"


def test_layout_cross_validation_orphaned(tmp_path: Path):
    json_data = """{
        "form_id": "f1",
        "form_version": "v1",
        "coordinate_space": "absolute_pixel",
        "reference_dpi": 200,
        "pages": [
            {
                "page_number": 1,
                "width_px": 100,
                "height_px": 100,
                "anchors": [],
                "rois": []
            }
        ]
    }"""
    p = tmp_path / "layout.json"
    p.write_text(json_data, encoding="utf-8")

    semantic = FormProfile(
        form_id="f1",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="Q_DUMMY",
                response_type="single_select",
                mark_strategy="circle",
                options=(OptionDef("opt_dummy"),),
                max_selections=1,
            ),
            QuestionDef(
                question_id="Q1",
                response_type="single_select",
                mark_strategy="circle",
                options=(OptionDef("opt1"),),
                max_selections=1,
            ),
        ),
    )

    with pytest.raises(ProfileValidationError, match="ORPHANED_OPTION"):
        load_layout_profile(p, semantic)


def test_layout_cross_validation_missing_ref(tmp_path: Path):
    json_data = """{
        "form_id": "f1",
        "form_version": "v1",
        "coordinate_space": "absolute_pixel",
        "reference_dpi": 200,
        "pages": [
            {
                "page_number": 1,
                "width_px": 100,
                "height_px": 100,
                "anchors": [],
                "rois": [
                    {"question_id": "UNKNOWN", "option_id": "opt1", "bbox": {"x": 10, "y": 10, "w": 10, "h": 10}}
                ]
            }
        ]
    }"""
    p = tmp_path / "layout.json"
    p.write_text(json_data, encoding="utf-8")

    semantic = FormProfile(
        form_id="f1",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="Q_DUMMY",
                response_type="single_select",
                mark_strategy="circle",
                options=(OptionDef("opt_dummy"),),
                max_selections=1,
            ),
        ),
    )

    with pytest.raises(ProfileValidationError, match="MISSING_SEMANTIC_REF"):
        load_layout_profile(p, semantic)


def test_layout_validation_out_of_bounds(tmp_path: Path):
    json_data = """{
        "form_id": "f1",
        "form_version": "v1",
        "coordinate_space": "absolute_pixel",
        "reference_dpi": 200,
        "pages": [
            {
                "page_number": 1,
                "width_px": 100,
                "height_px": 100,
                "anchors": [],
                "rois": [
                    {"question_id": "Q1", "option_id": "opt1", "bbox": {"x": 90, "y": 90, "w": 20, "h": 20}}
                ]
            }
        ]
    }"""
    p = tmp_path / "layout.json"
    p.write_text(json_data, encoding="utf-8")

    with pytest.raises(ProfileValidationError) as exc:
        load_layout_profile(p)
    assert exc.value.error_code == "INVALID_FIELD"
    assert "exceeds page dimensions" in exc.value.reason


def test_layout_validation_duplicate_roi(tmp_path: Path):
    json_data = """{
        "form_id": "f1",
        "form_version": "v1",
        "coordinate_space": "absolute_pixel",
        "reference_dpi": 200,
        "pages": [
            {
                "page_number": 1,
                "width_px": 100,
                "height_px": 100,
                "anchors": [],
                "rois": [
                    {"question_id": "Q1", "option_id": "opt1", "bbox": {"x": 10, "y": 10, "w": 10, "h": 10}},
                    {"question_id": "Q1", "option_id": "opt1", "bbox": {"x": 20, "y": 20, "w": 10, "h": 10}}
                ]
            }
        ]
    }"""
    p = tmp_path / "layout.json"
    p.write_text(json_data, encoding="utf-8")

    with pytest.raises(ProfileValidationError, match="DUPLICATE_ROI"):
        load_layout_profile(p)


def test_layout_mismatched_semantic(tmp_path: Path):
    json_data = """{
        "form_id": "f2",
        "form_version": "v1",
        "coordinate_space": "absolute_pixel",
        "reference_dpi": 200,
        "pages": [
            {"page_number": 1, "width_px": 100, "height_px": 100, "anchors": [], "rois": []}
        ]
    }"""
    p = tmp_path / "layout.json"
    p.write_text(json_data, encoding="utf-8")

    semantic = FormProfile(
        "f1",
        "v1",
        (QuestionDef("Q1", "single_select", "circle", (OptionDef("O1"),), max_selections=1),),
    )
    with pytest.raises(ProfileValidationError, match="SEMANTIC_MISMATCH"):
        load_layout_profile(p, semantic)


def test_anchor_def_invalid():
    b = BoundingBox(0, 0, 10, 10)
    with pytest.raises(ValueError, match="anchor_id must be a non-empty string"):
        AnchorDef("", "qr", b)
    with pytest.raises(ValueError, match="anchor_type must be a non-empty string"):
        AnchorDef("a1", "", b)
    with pytest.raises(ValueError, match="bbox must be a BoundingBox"):
        AnchorDef("a1", "qr", "bbox")  # type: ignore


def test_roi_def_invalid():
    b = BoundingBox(0, 0, 10, 10)
    with pytest.raises(ValueError, match="question_id must be a non-empty string"):
        RoiDef("", "opt", b)
    with pytest.raises(ValueError, match="option_id must be a non-empty string"):
        RoiDef("Q1", "", b)
    with pytest.raises(ValueError, match="bbox must be a BoundingBox"):
        RoiDef("Q1", "opt", "bbox")  # type: ignore
    with pytest.raises(
        ValueError, match="mark_strategy_override must be one of: circle, checkbox, rating, or None"
    ):
        RoiDef("Q1", "opt", b, 1)  # type: ignore


def test_page_layout_invalid():
    a = AnchorDef("a1", "qr", BoundingBox(0, 0, 10, 10))
    r = RoiDef("Q1", "opt", BoundingBox(20, 20, 10, 10))

    with pytest.raises(ValueError, match="page_number must be a positive int"):
        PageLayout(0, 100, 100, (a,), (r,))
    with pytest.raises(ValueError, match="width_px must be a positive int"):
        PageLayout(1, -10, 100, (a,), (r,))
    with pytest.raises(ValueError, match="height_px must be a positive int"):
        PageLayout(1, 100, 0, (a,), (r,))
    with pytest.raises(ValueError, match="anchors must be a tuple"):
        PageLayout(1, 100, 100, [a], (r,))  # type: ignore
    with pytest.raises(ValueError, match="rois must be a tuple"):
        PageLayout(1, 100, 100, (a,), [r])  # type: ignore

    a2 = AnchorDef("a1", "qr", BoundingBox(95, 95, 10, 10))
    with pytest.raises(ValueError, match="anchor bbox exceeds page dimensions"):
        PageLayout(1, 100, 100, (a2,), (r,))

    a3 = AnchorDef("a1", "qr", BoundingBox(-5, 5, 10, 10))
    with pytest.raises(ValueError, match="anchor bbox must have non-negative x/y"):
        PageLayout(1, 100, 100, (a3,), (r,))

    r2 = RoiDef("Q1", "opt", BoundingBox(-5, 5, 10, 10))
    with pytest.raises(ValueError, match="roi bbox must have non-negative x/y"):
        PageLayout(1, 100, 100, (a,), (r2,))


def test_layout_profile_model_invalid():
    p = PageLayout(1, 100, 100, (), ())
    with pytest.raises(ValueError, match="form_id must be a non-empty string"):
        LayoutProfile("", "v1", "absolute_pixel", 200, (p,))
    with pytest.raises(ValueError, match="form_version must be a non-empty string"):
        LayoutProfile("f", "", "absolute_pixel", 200, (p,))
    with pytest.raises(ValueError, match="coordinate_space must be 'absolute_pixel'"):
        LayoutProfile("f", "v1", "", 200, (p,))
    with pytest.raises(ValueError, match="reference_dpi must be a positive int"):
        LayoutProfile("f", "v1", "absolute_pixel", 0, (p,))
    with pytest.raises(ValueError, match="pages must be a tuple"):
        LayoutProfile("f", "v1", "absolute_pixel", 200, [p])  # type: ignore
    with pytest.raises(ValueError, match="pages cannot be empty"):
        LayoutProfile("f", "v1", "absolute_pixel", 200, ())

    p2 = PageLayout(3, 100, 100, (), ())
    with pytest.raises(ValueError, match="page numbers must be sequential"):
        LayoutProfile("f", "v1", "absolute_pixel", 200, (p2,))


def test_load_layout_profile_invalid_fields(tmp_path: Path):
    # Test file not found
    with pytest.raises(ProfileValidationError, match="FILE_NOT_FOUND"):
        load_layout_profile(tmp_path / "nonexistent.json")

    p = tmp_path / "bad.json"
    p.write_text("invalid json")
    with pytest.raises(ProfileValidationError, match="INVALID_JSON"):
        load_layout_profile(p)

    p.write_text("[]")
    with pytest.raises(ProfileValidationError, match="INVALID_ROOT"):
        load_layout_profile(p)

    p.write_text('{"form_id": "f"}')
    with pytest.raises(ProfileValidationError, match="INVALID_TYPE"):  # missing pages
        load_layout_profile(p)

    p.write_text('{"form_id": "f", "pages": [{}]}')
    with pytest.raises(ProfileValidationError, match="INVALID_TYPE"):  # missing anchors
        load_layout_profile(p)

    p.write_text(
        '{"form_id": "f", "pages": [{"page_number": 1, "width_px": 100, "height_px": 100, "anchors": [{}]}]}'
    )
    with pytest.raises(ProfileValidationError, match="INVALID_FIELD"):  # invalid anchor
        load_layout_profile(p)

    p.write_text(
        '{"form_id": "f", "pages": [{"page_number": 1, "width_px": 100, "height_px": 100, "anchors": [], "rois": [{}]}]}'
    )
    with pytest.raises(ProfileValidationError, match="INVALID_FIELD"):  # invalid roi
        load_layout_profile(p)

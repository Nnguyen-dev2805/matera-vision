import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from matera.core.profile import FormProfile, OptionDef, QuestionDef
from scripts.dataset_tools.golden import (
    AnnotationError,
    generate_golden_dataset,
    load_profile_semantic,
    parse_ground_truth,
)


def test_load_profile_semantic_options():
    # Simple mock structure using dataclasses
    mock_semantic = FormProfile(
        form_id="test",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="Q1",
                response_type="multi_select",
                mark_strategy="circle",
                options=(OptionDef("a"), OptionDef("b"), OptionDef("c")),
            ),
            QuestionDef(
                question_id="Q2",
                response_type="rating",
                mark_strategy="rating",
                options=(OptionDef("0", value=0), OptionDef("1", value=1)),
                min_selections=1,
                max_selections=1,
            ),
        ),
    )

    # We should have a helper that maps semantic to a dict of question_id -> list of option_ids
    semantic_map = load_profile_semantic(mock_semantic)
    assert semantic_map["Q1"] == ["a", "b", "c"]
    assert semantic_map["Q2"] == ["0", "1"]


def test_parse_ground_truth_success():
    mock_semantic = {"Q1": ["a", "b", "c"], "Q13.1": ["0", "1", "2"]}
    mock_gt = {"page_1": {"expected": {"Q1": ["0", "2"], "Q13_1": ["1"]}}}

    parsed = parse_ground_truth(mock_gt, mock_semantic, expected_pages=1)

    assert "page_1" in parsed
    page_1 = parsed["page_1"]

    # Q1 index 0 -> a, index 2 -> c
    assert page_1["Q1"] == ["a", "c"]
    # Q13_1 should map to Q13.1, index 1 -> 1
    assert page_1["Q13.1"] == ["1"]


def test_parse_ground_truth_invalid_index():
    mock_semantic = {"Q1": ["a", "b"]}
    mock_gt = {
        "page_1": {
            "expected": {
                "Q1": ["2"]  # Out of bounds!
            }
        }
    }

    with pytest.raises(AnnotationError, match="Index out of bounds"):
        parse_ground_truth(mock_gt, mock_semantic, expected_pages=1)


def test_parse_ground_truth_missing_question_in_semantic():
    mock_semantic = {"Q1": ["a", "b"]}
    mock_gt = {
        "page_1": {
            "expected": {
                "Q2": ["0"]  # Not in semantic!
            }
        }
    }

    with pytest.raises(AnnotationError, match="Question Q2 not found in semantic profile"):
        parse_ground_truth(mock_gt, mock_semantic, expected_pages=1)


def test_parse_ground_truth_missing_question_in_gt():
    mock_semantic = {"Q1": ["a", "b"], "Q2": ["c", "d"]}
    mock_gt = {
        "page_1": {
            "expected": {
                "Q1": ["0"]  # Q2 is missing!
            }
        }
    }

    with pytest.raises(AnnotationError, match="Question Q2 is missing from page_1 annotations"):
        parse_ground_truth(mock_gt, mock_semantic, expected_pages=1)


def test_parse_ground_truth_missing_page_in_gt():
    mock_semantic = {"Q1": ["a", "b"]}
    mock_gt = {
        "page_1": {"expected": {"Q1": ["0"]}}
        # missing page_2!
    }

    with pytest.raises(AnnotationError, match="Missing annotations for page_2 in ground truth"):
        parse_ground_truth(mock_gt, mock_semantic, expected_pages=2)


def test_generate_golden_dataset(tmp_path: Path):
    """
    Acceptance Test:
    Must run on the REAL profile and mock ground truth to assert exactly 770 rows
    and 10 pages processed.
    """
    # Load real profiles
    from matera.core.layout import load_layout_profile
    from matera.core.profile import load_semantic_profile

    semantic_path = Path("profiles/semantic.json")
    layout_path = Path("profiles/layout.json")

    semantic_profile = load_semantic_profile(semantic_path)
    layout_profile = load_layout_profile(layout_path, semantic=semantic_profile)

    # Mock ground truth for all 10 pages, marking just the first option of Q1 as selected
    mock_gt = {}
    for i in range(1, 11):
        # We must provide annotations for ALL questions defined in semantic_profile
        page_gt = {}
        for q in semantic_profile.questions:
            page_gt[q.question_id] = ["0"]  # Mark index 0 for everything

        # The Q13.1 alias mapping trick in our mock
        if "Q13.1" in page_gt:
            page_gt["Q13_1"] = page_gt.pop("Q13.1")
        if "Q13.2" in page_gt:
            page_gt["Q13_2"] = page_gt.pop("Q13.2")

        mock_gt[f"page_{i}"] = {"expected": page_gt}

    # Create fake 1x1 images
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    for i in range(1, 11):
        fake_img = pages_dir / f"page_{i}.png"
        fake_img.touch()

    output_dir = tmp_path / "golden"

    # Mock Pillow's Image.open to return a fake image that can be "cropped"
    mock_image_instance = MagicMock()
    mock_image_instance.crop.return_value = MagicMock()

    with patch("scripts.dataset_tools.golden.Image.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_image_instance

        generate_golden_dataset(
            pages_dir=pages_dir,
            semantic_profile=semantic_profile,
            layout_profile=layout_profile,
            ground_truth_json=mock_gt,
            output_dir=output_dir,
        )

    # Verify outputs
    csv_path = output_dir / "labels.csv"
    assert csv_path.exists()

    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # 10 pages * 77 ROIs = 770 rows
    assert len(reader) == 770

    # Verify exact counts per split
    train_rows = [r for r in reader if r["split"] == "train"]
    dev_rows = [r for r in reader if r["split"] == "dev"]
    assert len(train_rows) == 5 * 77  # 385
    assert len(dev_rows) == 5 * 77  # 385

    # Verify expected_mark logic
    q1_a_marked = [
        r
        for r in reader
        if r["question_id"] == "Q1" and r["option_id"] == "a" and r["expected_mark"] == "1"
    ]
    assert len(q1_a_marked) == 10  # 1 per page

    # Verify some audit metadata exists
    first_row = reader[0]
    assert first_row["form_id"] == "matera-pre"
    assert first_row["annotation_source"] == "human_ground_truth.json"
    assert first_row["mark_strategy"] in ["circle", "checkbox"]

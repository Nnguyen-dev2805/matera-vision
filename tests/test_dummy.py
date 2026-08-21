import pytest

from matera.core.contracts import NormalizedAnswer


def test_normalized_answer_creation():
    """Test that the NormalizedAnswer contract can be instantiated."""
    answer = NormalizedAnswer(
        form_id="matera-pre",
        form_version="v1",
        page_number=1,
        question_id="Q1",
        option_id="a",
        decision="selected",
        confidence=0.95,
        evidence_path="debug/Q1-a.png",
        model_version="v1.0",
    )
    assert answer.form_id == "matera-pre"
    assert answer.decision == "selected"
    assert answer.confidence == 0.95
    assert answer.evidence_path == "debug/Q1-a.png"
    assert answer.model_version == "v1.0"


def test_normalized_answer_boundary_decisions():
    """Test valid boundary decisions."""
    ans1 = NormalizedAnswer(
        form_id="m",
        form_version="1",
        page_number=1,
        question_id="Q1",
        option_id="a",
        decision="unselected",
    )
    ans2 = NormalizedAnswer(
        form_id="m",
        form_version="1",
        page_number=1,
        question_id="Q1",
        option_id="a",
        decision="review",
    )
    assert ans1.decision == "unselected"
    assert ans2.decision == "review"


def test_normalized_answer_boundary_confidence():
    """Test confidence boundaries."""
    ans1 = NormalizedAnswer(
        form_id="m",
        form_version="1",
        page_number=1,
        question_id="Q1",
        option_id="a",
        decision="review",
        confidence=0.0,
    )
    ans2 = NormalizedAnswer(
        form_id="m",
        form_version="1",
        page_number=1,
        question_id="Q1",
        option_id="a",
        decision="review",
        confidence=1.0,
    )
    ans3 = NormalizedAnswer(
        form_id="m",
        form_version="1",
        page_number=1,
        question_id="Q1",
        option_id="a",
        decision="review",
        confidence=None,
    )
    assert ans1.confidence == 0.0
    assert ans2.confidence == 1.0
    assert ans3.confidence is None


def test_normalized_answer_invalid_decision():
    """Test that invalid decision raises ValueError."""
    with pytest.raises(ValueError, match="decision must be"):
        NormalizedAnswer(
            form_id="matera",
            form_version="v1",
            page_number=1,
            question_id="Q1",
            option_id="a",
            decision="maybe",  # type: ignore
        )


def test_normalized_answer_invalid_page():
    """Test that invalid page_number raises ValueError."""
    with pytest.raises(ValueError, match="page_number must be > 0"):
        NormalizedAnswer(
            form_id="matera",
            form_version="v1",
            page_number=0,
            question_id="Q1",
            option_id="a",
            decision="selected",
        )
    with pytest.raises(ValueError, match="page_number must be > 0"):
        NormalizedAnswer(
            form_id="matera",
            form_version="v1",
            page_number=-1,
            question_id="Q1",
            option_id="a",
            decision="selected",
        )


def test_normalized_answer_invalid_confidence():
    """Test that invalid confidence raises ValueError."""
    with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
        NormalizedAnswer(
            form_id="matera",
            form_version="v1",
            page_number=1,
            question_id="Q1",
            option_id="a",
            decision="selected",
            confidence=1.5,
        )

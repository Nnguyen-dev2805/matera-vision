import pytest

from matera.core.contracts import (
    AnswerKey,
    NormalizedAnswer,
    NormalizedPageResult,
    ReviewTask,
)


def test_answer_key_valid():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    assert key.page_number == 1


def test_answer_key_invalid_page():
    with pytest.raises(ValueError, match="page_number must be > 0"):
        AnswerKey("form1", "v1", 0, "Q1", "A")


def test_answer_key_invalid_types():
    with pytest.raises(ValueError, match="form_id must be a string"):
        AnswerKey(1, "v1", 1, "Q1", "A")  # type: ignore
    with pytest.raises(ValueError, match="form_version must be a string"):
        AnswerKey("form1", 1, 1, "Q1", "A")  # type: ignore
    with pytest.raises(ValueError, match="page_number must be an int"):
        AnswerKey("form1", "v1", "1", "Q1", "A")  # type: ignore
    with pytest.raises(ValueError, match="question_id must be a string"):
        AnswerKey("form1", "v1", 1, 1, "A")  # type: ignore
    with pytest.raises(ValueError, match="option_id must be a string"):
        AnswerKey("form1", "v1", 1, "Q1", 1)  # type: ignore


def test_answer_key_empty_identifiers():
    with pytest.raises(ValueError, match="form_id cannot be empty"):
        AnswerKey("", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="form_version cannot be empty"):
        AnswerKey("form", " ", 1, "Q1", "A")
    with pytest.raises(ValueError, match="question_id cannot be empty"):
        AnswerKey("form", "v1", 1, "\t", "A")
    with pytest.raises(ValueError, match="option_id cannot be empty"):
        AnswerKey("form", "v1", 1, "Q1", "")


def test_normalized_answer_valid_resolved():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    ans = NormalizedAnswer(
        answer_key=key,
        selected=True,
        resolution_status="resolved",
        decision_source="deterministic",
        confidence=0.8,
        deterministic_score=1.0,
    )
    assert ans.selected is True


def test_normalized_answer_valid_needs_review():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    ans = NormalizedAnswer(
        answer_key=key,
        selected=None,
        resolution_status="needs_review",
        decision_source="classifier",
    )
    assert ans.resolution_status == "needs_review"


def test_normalized_answer_invalid_selected_type():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="selected must be a boolean or None"):
        NormalizedAnswer(
            answer_key=key,
            selected=1,  # type: ignore
            resolution_status="resolved",
            decision_source="deterministic",
        )


def test_normalized_answer_invalid_needs_review():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="selected must be None"):
        NormalizedAnswer(
            answer_key=key,
            selected=False,
            resolution_status="needs_review",
            decision_source="classifier",
        )


def test_normalized_answer_invalid_resolved():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="selected must be a bool"):
        NormalizedAnswer(
            answer_key=key,
            selected=None,
            resolution_status="resolved",
            decision_source="deterministic",
        )


def test_normalized_answer_invalid_human():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="Human decisions must have resolution_status='resolved'"):
        NormalizedAnswer(
            answer_key=key,
            selected=None,
            resolution_status="needs_review",
            decision_source="human",
        )


def test_normalized_answer_invalid_resolution_status():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="Invalid resolution_status unknown"):
        NormalizedAnswer(
            answer_key=key,
            selected=True,
            resolution_status="unknown",  # type: ignore
            decision_source="deterministic",
        )


def test_normalized_answer_invalid_decision_source():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="Invalid decision_source unknown"):
        NormalizedAnswer(
            answer_key=key,
            selected=True,
            resolution_status="resolved",
            decision_source="unknown",  # type: ignore
        )


def test_normalized_answer_invalid_scores():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
        NormalizedAnswer(
            answer_key=key,
            selected=True,
            resolution_status="resolved",
            decision_source="deterministic",
            confidence=1.5,
        )
    with pytest.raises(ValueError, match="deterministic_score must be between 0.0 and 1.0"):
        NormalizedAnswer(
            answer_key=key,
            selected=True,
            resolution_status="resolved",
            decision_source="deterministic",
            deterministic_score=-0.1,
        )


def test_review_task_valid():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    task = ReviewTask("t1", key, "ambiguous", "pending")
    assert task.status == "pending"


def test_review_task_invalid_resolved():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="reviewer_decision must be a bool"):
        ReviewTask("t1", key, "ambiguous", "resolved")


def test_review_task_invalid_decision_type():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="reviewer_decision must be a boolean or None"):
        ReviewTask("t1", key, "ambiguous", "resolved", reviewer_decision=1)  # type: ignore


def test_review_task_invalid_pending():
    ak = AnswerKey("form1", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="reviewer_decision must be None"):
        ReviewTask("t1", ak, "ambiguous", "pending", reviewer_decision=True)


def test_review_task_invalid_types():
    ak = AnswerKey("form1", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="task_id must be a string"):
        ReviewTask(1, ak, "ambiguous", "pending")  # type: ignore
    with pytest.raises(ValueError, match="answer_key must be an instance of AnswerKey"):
        ReviewTask("t1", "not_an_ak", "ambiguous", "pending")  # type: ignore


def test_review_task_empty_identifiers():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="task_id cannot be empty"):
        ReviewTask("  ", key, "ambiguous", "pending")
    with pytest.raises(ValueError, match="reason cannot be empty"):
        ReviewTask("t1", key, "", "pending")


def test_review_task_invalid_status():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    with pytest.raises(ValueError, match="Invalid status unknown"):
        ReviewTask("t1", key, "ambiguous", "unknown")  # type: ignore


def test_normalized_page_result_resolved():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    ans = NormalizedAnswer(key, True, "resolved", "deterministic")
    page = NormalizedPageResult("form", "v1", 1, answers=(ans,))
    assert page.page_status == "resolved"


def test_normalized_page_result_review_required():
    key1 = AnswerKey("form", "v1", 1, "Q1", "A")
    key2 = AnswerKey("form", "v1", 1, "Q1", "B")
    ans1 = NormalizedAnswer(key1, True, "resolved", "deterministic")
    ans2 = NormalizedAnswer(key2, None, "needs_review", "classifier")

    # Must provide a matching review task for needs_review
    rt = ReviewTask("t1", key2, "low_confidence", "pending")

    page = NormalizedPageResult("form", "v1", 1, answers=(ans1, ans2), review_tasks=(rt,))
    assert page.page_status == "review_required"


def test_normalized_page_result_missing_pending_review_task():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    ans = NormalizedAnswer(key, None, "needs_review", "classifier")

    with pytest.raises(
        ValueError, match="needs_review but has no corresponding pending ReviewTask"
    ):
        NormalizedPageResult("form", "v1", 1, answers=(ans,))


def test_normalized_page_result_empty_identifiers():
    with pytest.raises(ValueError, match="form_id cannot be empty"):
        NormalizedPageResult("", "v1", 1, answers=())
    with pytest.raises(ValueError, match="form_version cannot be empty"):
        NormalizedPageResult("form", " ", 1, answers=())


def test_normalized_page_result_invalid_page_number():
    with pytest.raises(ValueError, match="page_number must be > 0"):
        NormalizedPageResult("form1", "v1", 0, (), ())


def test_normalized_page_result_invalid_types():
    with pytest.raises(ValueError, match="form_id must be a string"):
        NormalizedPageResult(1, "v1", 1, (), ())  # type: ignore
    with pytest.raises(ValueError, match="form_version must be a string"):
        NormalizedPageResult("form1", 1, 1, (), ())  # type: ignore
    with pytest.raises(ValueError, match="page_number must be an int"):
        NormalizedPageResult("form1", "v1", "1", (), ())  # type: ignore
    with pytest.raises(
        ValueError, match="All elements in answers must be instances of NormalizedAnswer"
    ):
        NormalizedPageResult("form1", "v1", 1, ("not_an_answer",), ())  # type: ignore
    with pytest.raises(
        ValueError, match="All elements in review_tasks must be instances of ReviewTask"
    ):
        NormalizedPageResult("form1", "v1", 1, (), ("not_a_task",))  # type: ignore


def test_normalized_page_result_mismatched_page():
    key = AnswerKey("form", "v1", 2, "Q1", "A")  # belongs to page 2
    ans = NormalizedAnswer(key, True, "resolved", "deterministic")

    with pytest.raises(
        ValueError, match="All answers must match the page's form_id, form_version, and page_number"
    ):
        NormalizedPageResult("form", "v1", 1, answers=(ans,))

    # Same for review tasks
    rt = ReviewTask("t1", key, "low_confidence", "pending")
    with pytest.raises(ValueError, match="All review_tasks must match"):
        NormalizedPageResult("form", "v1", 1, answers=(), review_tasks=(rt,))


def test_normalized_page_result_duplicate_answers():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    ans1 = NormalizedAnswer(key, True, "resolved", "deterministic")
    ans2 = NormalizedAnswer(key, False, "resolved", "deterministic")

    with pytest.raises(ValueError, match="Duplicate answer_key found"):
        NormalizedPageResult("form", "v1", 1, answers=(ans1, ans2))


def test_normalized_page_result_task_references_missing_answer():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    rt = ReviewTask("t1", key, "ambiguous", "pending")
    with pytest.raises(ValueError, match="ReviewTask references missing answer"):
        NormalizedPageResult("form", "v1", 1, answers=(), review_tasks=(rt,))


def test_normalized_page_result_pending_task_references_resolved_answer():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    ans = NormalizedAnswer(key, True, "resolved", "deterministic")
    rt = ReviewTask("t1", key, "ambiguous", "pending")
    with pytest.raises(ValueError, match="Pending ReviewTask references resolved answer"):
        NormalizedPageResult("form", "v1", 1, answers=(ans,), review_tasks=(rt,))


def test_normalized_page_result_multiple_pending_tasks():
    key = AnswerKey("form", "v1", 1, "Q1", "A")
    ans = NormalizedAnswer(key, None, "needs_review", "classifier")
    rt1 = ReviewTask("t1", key, "ambiguous", "pending")
    rt2 = ReviewTask("t2", key, "still_ambiguous", "pending")
    with pytest.raises(ValueError, match="Multiple pending ReviewTasks for answer"):
        NormalizedPageResult("form", "v1", 1, answers=(ans,), review_tasks=(rt1, rt2))


def test_immutability():
    from dataclasses import FrozenInstanceError

    key = AnswerKey("form", "v1", 1, "Q1", "A")
    ans = NormalizedAnswer(key, True, "resolved", "deterministic")
    with pytest.raises(FrozenInstanceError):
        ans.selected = False  # type: ignore

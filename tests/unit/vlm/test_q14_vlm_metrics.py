from matera.vlm.metrics import calculate_q14_vlm_metrics
from matera.vlm.models import Q14VlmOptionResult


def _make_result(page, opt, exp, vlm, parse_err=None):
    return Q14VlmOptionResult(
        page_number=page,
        option_id=opt,
        expected_state=exp,
        vlm_state=vlm,
        raw_vlm_state=vlm,
        reason=None,
        parse_error=parse_err,
        image_path="",
    )


def test_need_review_increments_review_count_not_fp_fn():
    results = [
        _make_result(1, "1", "BLANK", "NEED_REVIEW"),
        _make_result(1, "2", "MARKED", "NEED_REVIEW"),
    ]
    metrics = calculate_q14_vlm_metrics(results)

    assert metrics["needs_review"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tp"] == 0
    assert metrics["tn"] == 0


def test_auto_accuracy_excludes_need_review():
    results = [
        _make_result(1, "1", "BLANK", "BLANK"),  # tn
        _make_result(1, "2", "MARKED", "MARKED"),  # tp
        _make_result(1, "3", "BLANK", "MARKED"),  # fp
        _make_result(1, "4", "BLANK", "NEED_REVIEW"),  # review
    ]
    metrics = calculate_q14_vlm_metrics(results)

    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["needs_review"] == 1

    # 2 correct out of 3 auto-decided
    assert metrics["auto_accuracy"] == 2 / 3


def test_review_rate_uses_total_options():
    results = [
        _make_result(1, "1", "BLANK", "BLANK"),
        _make_result(1, "2", "MARKED", "MARKED"),
        _make_result(1, "3", "BLANK", "MARKED"),
        _make_result(1, "4", "BLANK", "NEED_REVIEW"),
    ]
    metrics = calculate_q14_vlm_metrics(results)

    assert metrics["options_total"] == 4
    assert metrics["review_rate"] == 1 / 4


def test_pages_exact_match_auto_only():
    results = [
        # Page 1: perfect
        _make_result(1, "1", "BLANK", "BLANK"),
        _make_result(1, "2", "MARKED", "MARKED"),
        # Page 2: has review
        _make_result(2, "1", "BLANK", "BLANK"),
        _make_result(2, "2", "MARKED", "NEED_REVIEW"),
        # Page 3: has mistake
        _make_result(3, "1", "BLANK", "MARKED"),
        _make_result(3, "2", "MARKED", "MARKED"),
    ]
    metrics = calculate_q14_vlm_metrics(results)
    assert metrics["pages_total"] == 3
    assert metrics["pages_exact_match_auto_only"] == 1
    assert (
        metrics["pages_with_any_error"] == 1
    )  # Page 3 is error. NEED_REVIEW is not a mistake/parse error.


def test_pages_with_any_error():
    results = [
        # Page 1: parse error
        _make_result(1, "1", "BLANK", "NEED_REVIEW", parse_err="Bad"),
        # Page 2: fp mistake
        _make_result(2, "1", "BLANK", "MARKED"),
    ]
    metrics = calculate_q14_vlm_metrics(results)
    assert metrics["pages_with_any_error"] == 2

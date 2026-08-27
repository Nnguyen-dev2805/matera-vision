def test_ground_truth_report_q14(tmp_path):
    import json

    from matera.evaluation.ground_truth_metrics import (
        OptionEvaluationResult,
        PageEvaluationResult,
        QuestionEvaluationResult,
    )
    from matera.evaluation.ground_truth_report import GroundTruthReporter

    o_res = OptionEvaluationResult(
        question_id="Q14",
        option_id="a",
        expected_state="BLANK",
        actual_state="MARKED",
        score=0.9,
        is_correct=False,
        is_review=False,
        is_skipped=False,
        error_type="FP",
        failure_taxonomy="q14_false_positive",
        debug_report_path="report.html",
        evidence_path="evidence.png",
        method="HSV_AI",
        q14_diagnostic_path="q14.png",
    )

    q_res = QuestionEvaluationResult(
        question_id="Q14",
        expected_marked=[],
        actual_marked=["a"],
        unknown=False,
        has_review=False,
        has_error=False,
        exact_match_auto=False,
        option_error_count=0,
        option_review_count=0,
        debug_report_path="report.html",
        option_results=[o_res],
    )

    p_res = PageEvaluationResult(
        item_id="item1",
        pdf="test.pdf",
        page=1,
        exact_match_auto=False,
        review_required=False,
        question_results=[q_res],
        q14_diagnostics=[
            {"expected_state": "BLANK", "actual_state": "MARKED", "safe_mask_pixels": 250}
        ],
    )

    reporter = GroundTruthReporter(tmp_path)
    reporter.write_report("test_run", {}, [p_res])

    # Check CSV
    details = (tmp_path / "details.csv").read_text(encoding="utf-8")
    assert "q14_diagnostic_path" in details
    assert "q14.png" in details

    # Check JSON
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "q14" in summary
    assert summary["q14"]["fp"] == 1
    assert summary["q14"]["safe_mask_pixel_distribution"]["max"] == 250

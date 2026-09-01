import csv
import json
from pathlib import Path

from matera.evaluation.ground_truth_metrics import (
    OptionEvaluationResult,
    PageEvaluationResult,
    QuestionEvaluationResult,
)
from matera.evaluation.ground_truth_report import GroundTruthReporter


def test_reporter_output_shape(tmp_path: Path):
    reporter = GroundTruthReporter(tmp_path)

    # Create mock result
    o_res = OptionEvaluationResult(
        question_id="Q1",
        option_id="O1",
        expected_state="MARKED",
        actual_state="MARKED",
        is_correct=True,
        is_review=False,
        is_skipped=False,
        error_type=None,
        failure_taxonomy="none",
        method="mock",
        score=0.99,
        debug_report_path="debug.html",
        evidence_path="evidence.json",
        q14_diagnostic_path=None,
        notes=None,
        local_realignment=None,
    )

    q_res = QuestionEvaluationResult(
        question_id="Q1",
        expected_marked=["O1"],
        actual_marked=["O1"],
        unknown=[],
        has_review=False,
        has_error=False,
        exact_match_auto=True,
        option_error_count=0,
        option_review_count=0,
        option_results=[o_res],
        debug_report_path=None,
        notes=None,
    )

    p_res = PageEvaluationResult(
        item_id="item1",
        pdf="mock.pdf",
        page=1,
        question_results=[q_res],
        exact_match_auto=True,
        review_required=False,
        q14_diagnostics=[],
        global_shape_diagnostics=[],
    )

    reporter.write_report("run1", {"test": True}, [p_res])

    # Assert summary.json
    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
        assert "run_config" in summary_data
        assert "metrics" in summary_data
        assert "taxonomy" in summary_data
        assert "q14" in summary_data
        assert "global_shape_validator" in summary_data
        assert "local_realignment" in summary_data

        assert "checkbox_locator" in summary_data

    # Assert details.csv
    details_path = tmp_path / "details.csv"
    assert details_path.exists()
    with open(details_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        expected_details = [
            "run_id",
            "item_id",
            "pdf",
            "page",
            "question_id",
            "option_id",
            "expected_state",
            "actual_state",
            "is_correct",
            "is_review",
            "is_skipped",
            "error_type",
            "failure_taxonomy",
            "method",
            "score",
            "debug_report_path",
            "evidence_path",
            "q14_diagnostic_path",
            "notes",
            "vlm_state",
            "vlm_raw_state",
            "vlm_reason",
            "vlm_parse_error",
            "vlm_provider_error",
            "vlm_crop_path",
            "vlm_latency_ms",
        ]
        assert headers == expected_details

    # Assert questions.csv
    questions_path = tmp_path / "questions.csv"
    assert questions_path.exists()
    with open(questions_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        expected_questions = [
            "run_id",
            "item_id",
            "pdf",
            "page",
            "question_id",
            "expected_marked",
            "actual_marked",
            "unknown",
            "has_review",
            "has_error",
            "exact_match_auto",
            "option_error_count",
            "option_review_count",
            "debug_report_path",
            "notes",
        ]
        assert headers == expected_questions

    # Assert mistakes.md
    mistakes_path = tmp_path / "mistakes.md"
    assert mistakes_path.exists()
    with open(mistakes_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "# Ground Truth Evaluation Mistakes" in content
        assert "## False Positives" in content
        assert "## False Negatives" in content
        assert "## Needs Review" in content
        assert "## Errors / Missing" in content

from matera.evaluation.metrics import ConfusionMatrix
from matera.evaluation.report import EvaluationReport


def test_confusion_matrix_metrics():
    # 100 automatic decisions, 20 reviews.
    # Of automatic decisions: 80 TP, 15 TN, 5 FP, 0 FN.
    cm = ConfusionMatrix(tp=80, tn=15, fp=5, fn=0, needs_review=20)

    assert cm.automatic_decisions == 100
    assert cm.total_options == 120
    assert cm.automatic_errors == 5

    # Precision = TP / (TP + FP) = 80 / 85
    assert abs(cm.precision - (80 / 85)) < 1e-6
    # Recall = TP / (TP + FN) = 80 / 80 = 1.0
    assert cm.recall == 1.0
    # F1 = 2 * (P * R) / (P + R)
    assert cm.f1_score > 0.0

    # Coverage = automatic / total = 100 / 120
    assert abs(cm.coverage - (100 / 120)) < 1e-6
    # Risk = errors / automatic = 5 / 100
    assert cm.risk == 0.05
    # Review rate = review / total = 20 / 120
    assert abs(cm.review_rate - (20 / 120)) < 1e-6


def test_confusion_matrix_div_by_zero():
    cm = ConfusionMatrix(tp=0, tn=0, fp=0, fn=0, needs_review=0)

    assert cm.precision == 0.0
    assert cm.recall == 0.0
    assert cm.f1_score == 0.0
    assert cm.fpr == 0.0
    assert cm.fnr == 0.0
    assert cm.coverage == 0.0
    assert cm.risk == 0.0
    assert cm.review_rate == 0.0


def test_confusion_matrix_addition():
    cm1 = ConfusionMatrix(tp=10, fp=2)
    cm2 = ConfusionMatrix(tp=5, fn=1, needs_review=5)

    result = cm1.add(cm2)
    assert result.tp == 15
    assert result.fp == 2
    assert result.fn == 1
    assert result.tn == 0
    assert result.needs_review == 5


def test_evaluation_report_page_match_rate():
    report = EvaluationReport(total_pages=10, exact_match_pages=9)
    assert report.page_exact_match_rate == 0.9


def test_evaluation_report_zero_pages():
    report = EvaluationReport(total_pages=0, exact_match_pages=0)
    assert report.page_exact_match_rate == 0.0

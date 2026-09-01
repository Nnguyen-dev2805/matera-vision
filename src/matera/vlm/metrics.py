from typing import Any

from matera.vlm.models import Q14VlmOptionResult


def calculate_q14_vlm_metrics(results: list[Q14VlmOptionResult]) -> dict[str, Any]:
    pages_total = len(set(r.page_number for r in results))
    options_total = len(results)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    needs_review = 0

    # Pre-calculate page-level tracking
    page_errors = set()
    page_parse_errors = set()
    page_has_mistake = set()  # Pages that have a wrong prediction (excluding NEED_REVIEW)
    page_has_review = set()  # Pages that have at least one NEED_REVIEW

    for r in results:
        if r.parse_error:
            page_parse_errors.add(r.page_number)
            page_errors.add(r.page_number)

        if r.vlm_state == "NEED_REVIEW":
            needs_review += 1
            page_has_review.add(r.page_number)
            continue

        if r.expected_state == "MARKED" and r.vlm_state == "MARKED":
            tp += 1
        elif r.expected_state == "BLANK" and r.vlm_state == "BLANK":
            tn += 1
        elif r.expected_state == "BLANK" and r.vlm_state == "MARKED":
            fp += 1
            page_has_mistake.add(r.page_number)
        elif r.expected_state == "MARKED" and r.vlm_state == "BLANK":
            fn += 1
            page_has_mistake.add(r.page_number)

    total_auto = tp + tn + fp + fn
    auto_accuracy = (tp + tn) / total_auto if total_auto > 0 else None
    review_rate = needs_review / options_total if options_total > 0 else None
    coverage = total_auto / options_total if options_total > 0 else None

    precision_marked = tp / (tp + fp) if (tp + fp) > 0 else None
    recall_marked = tp / (tp + fn) if (tp + fn) > 0 else None

    actual_blanks = tn + fp
    actual_marked = tp + fn

    false_positive_rate = fp / actual_blanks if actual_blanks > 0 else None
    false_negative_rate = fn / actual_marked if actual_marked > 0 else None

    # Calculate pages exact match auto only
    # To be exactly matched auto only: page must have NO mistakes and NO reviews
    pages_exact_match_auto_only = 0
    for page in set(r.page_number for r in results):
        if page not in page_has_mistake and page not in page_has_review:
            pages_exact_match_auto_only += 1

    # pages_with_any_error means either a parse error or a provider error or a mistake
    # Wait, the spec says "page exact match ignoring NEED_REVIEW is not enough; include pages_exact_match_auto_only and pages_with_any_error".  # noqa: E501
    # I will define pages_with_any_error as pages with parse error, or a wrong prediction.
    pages_with_any_error = len(page_errors | page_has_mistake)

    return {
        "pages_total": pages_total,
        "options_total": options_total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "needs_review": needs_review,
        "auto_accuracy": auto_accuracy,
        "review_rate": review_rate,
        "coverage": coverage,
        "precision_marked": precision_marked,
        "recall_marked": recall_marked,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "parse_error_pages": len(page_parse_errors),
        "provider_error_pages": 0,  # To be filled by the outer script if needed
        "mean_latency_ms": None,  # To be filled by the outer script
        "pages_exact_match_auto_only": pages_exact_match_auto_only,
        "pages_with_any_error": pages_with_any_error,
    }

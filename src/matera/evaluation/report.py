from dataclasses import dataclass, field

from matera.evaluation.metrics import ConfusionMatrix


@dataclass
class EvaluationReport:
    """Stores the aggregated results of the evaluation harness."""

    overall_metrics: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    by_response_type: dict[str, ConfusionMatrix] = field(default_factory=dict)
    total_pages: int = 0
    exact_match_pages: int = 0

    @property
    def page_exact_match_rate(self) -> float:
        """Percentage of pages with 0 automatic errors and 0 needs_review."""
        if self.total_pages == 0:
            return 0.0
        return self.exact_match_pages / self.total_pages

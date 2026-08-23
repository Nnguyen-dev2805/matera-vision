from dataclasses import dataclass


@dataclass
class ConfusionMatrix:
    """Tracks outcomes for deterministic rules and review logic."""

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    needs_review: int = 0

    @property
    def automatic_decisions(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def automatic_errors(self) -> int:
        return self.fp + self.fn

    @property
    def total_options(self) -> int:
        return self.automatic_decisions + self.needs_review

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def f1_score(self) -> float:
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def fpr(self) -> float:
        if self.tn + self.fp == 0:
            return 0.0
        return self.fp / (self.tn + self.fp)

    @property
    def fnr(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.fn / (self.tp + self.fn)

    @property
    def coverage(self) -> float:
        """Percentage of total options that were decided automatically."""
        if self.total_options == 0:
            return 0.0
        return self.automatic_decisions / self.total_options

    @property
    def risk(self) -> float:
        """Percentage of automatic decisions that were incorrect."""
        if self.automatic_decisions == 0:
            return 0.0
        return self.automatic_errors / self.automatic_decisions

    @property
    def review_rate(self) -> float:
        """Percentage of total options routed to manual review."""
        if self.total_options == 0:
            return 0.0
        return self.needs_review / self.total_options

    def add(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        """Adds two confusion matrices together."""
        return ConfusionMatrix(
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            needs_review=self.needs_review + other.needs_review,
        )

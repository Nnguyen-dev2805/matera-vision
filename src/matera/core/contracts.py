from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class NormalizedAnswer:
    """Represents a single parsed answer from a form."""

    form_id: str
    form_version: str
    page_number: int
    question_id: str
    option_id: str
    decision: Literal["selected", "unselected", "review"]
    confidence: Optional[float] = None
    evidence_path: Optional[str] = None
    model_version: Optional[str] = None

    def __post_init__(self) -> None:
        if self.page_number <= 0:
            raise ValueError("page_number must be > 0")
        if self.decision not in ("selected", "unselected", "review"):
            raise ValueError("decision must be 'selected', 'unselected', or 'review'")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

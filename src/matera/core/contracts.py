from dataclasses import dataclass
from typing import Literal


def _is_empty_or_whitespace(s: str) -> bool:
    return not s or not s.strip()


@dataclass(frozen=True)
class AnswerKey:
    form_id: str
    form_version: str
    page_number: int
    question_id: str
    option_id: str

    def __post_init__(self) -> None:
        if type(self.form_id) is not str:
            raise ValueError("form_id must be a string")
        if type(self.form_version) is not str:
            raise ValueError("form_version must be a string")
        if type(self.page_number) is not int:
            raise ValueError("page_number must be an int")
        if type(self.question_id) is not str:
            raise ValueError("question_id must be a string")
        if type(self.option_id) is not str:
            raise ValueError("option_id must be a string")

        if self.page_number <= 0:
            raise ValueError(f"page_number must be > 0, got {self.page_number}")

        if _is_empty_or_whitespace(self.form_id):
            raise ValueError("form_id cannot be empty")
        if _is_empty_or_whitespace(self.form_version):
            raise ValueError("form_version cannot be empty")
        if _is_empty_or_whitespace(self.question_id):
            raise ValueError("question_id cannot be empty")
        if _is_empty_or_whitespace(self.option_id):
            raise ValueError("option_id cannot be empty")


@dataclass(frozen=True)
class NormalizedAnswer:
    answer_key: AnswerKey
    selected: bool | None
    resolution_status: Literal["resolved", "needs_review"]
    decision_source: Literal["deterministic", "classifier", "human"]
    confidence: float | None = None
    evidence_path: str | None = None
    deterministic_score: float | None = None
    model_version: str | None = None

    def __post_init__(self) -> None:
        if self.selected is not None and not isinstance(self.selected, bool):
            raise ValueError("selected must be a boolean or None")

        if self.resolution_status == "needs_review" and self.selected is not None:
            raise ValueError("selected must be None when resolution_status is needs_review")

        if self.resolution_status == "resolved" and self.selected is None:
            raise ValueError("selected must be a bool when resolution_status is resolved")

        if self.decision_source == "human" and self.resolution_status != "resolved":
            raise ValueError("Human decisions must have resolution_status='resolved'")

        valid_statuses = ("resolved", "needs_review")
        if self.resolution_status not in valid_statuses:
            raise ValueError(f"Invalid resolution_status {self.resolution_status}")

        valid_sources = ("deterministic", "classifier", "human")
        if self.decision_source not in valid_sources:
            raise ValueError(f"Invalid decision_source {self.decision_source}")

        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise ValueError("confidence must be a float between 0.0 and 1.0")
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

        if self.deterministic_score is not None:
            if isinstance(self.deterministic_score, bool) or not isinstance(
                self.deterministic_score, (int, float)
            ):
                raise ValueError("deterministic_score must be a float between 0.0 and 1.0")
            if not (0.0 <= self.deterministic_score <= 1.0):
                raise ValueError(
                    "deterministic_score must be between 0.0 and 1.0, "
                    f"got {self.deterministic_score}"
                )


@dataclass(frozen=True)
class ReviewTask:
    task_id: str
    answer_key: AnswerKey
    reason: str
    status: Literal["pending", "resolved"]
    reviewer_decision: bool | None = None
    reviewer_note: str | None = None

    def __post_init__(self) -> None:
        if type(self.task_id) is not str:
            raise ValueError("task_id must be a string")
        if not isinstance(self.answer_key, AnswerKey):
            raise ValueError("answer_key must be an instance of AnswerKey")
        if not self.task_id or not self.task_id.strip():
            raise ValueError("task_id cannot be empty")
        if _is_empty_or_whitespace(self.reason):
            raise ValueError("reason cannot be empty")

        valid_statuses = ("pending", "resolved")
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status {self.status}")

        if self.status == "resolved" and self.reviewer_decision is None:
            raise ValueError("reviewer_decision must be a bool when status is resolved")

        if self.status == "pending" and self.reviewer_decision is not None:
            raise ValueError("reviewer_decision must be None when status is pending")

        if self.reviewer_decision is not None and not isinstance(self.reviewer_decision, bool):
            raise ValueError("reviewer_decision must be a boolean or None")


@dataclass(frozen=True)
class NormalizedPageResult:
    form_id: str
    form_version: str
    page_number: int
    answers: tuple[NormalizedAnswer, ...]
    review_tasks: tuple[ReviewTask, ...] = ()

    def __post_init__(self) -> None:
        if type(self.form_id) is not str:
            raise ValueError("form_id must be a string")
        if type(self.form_version) is not str:
            raise ValueError("form_version must be a string")
        if type(self.page_number) is not int:
            raise ValueError("page_number must be an int")
        if self.page_number <= 0:
            raise ValueError("page_number must be > 0")
        if _is_empty_or_whitespace(self.form_id):
            raise ValueError("form_id cannot be empty")
        if _is_empty_or_whitespace(self.form_version):
            raise ValueError("form_version cannot be empty")

        if not isinstance(self.answers, tuple):
            raise ValueError("answers must be a tuple")
        for ans in self.answers:
            if not isinstance(ans, NormalizedAnswer):
                raise ValueError("All elements in answers must be instances of NormalizedAnswer")

        if not isinstance(self.review_tasks, tuple):
            raise ValueError("review_tasks must be a tuple")
        for rt in self.review_tasks:
            if not isinstance(rt, ReviewTask):
                raise ValueError("All elements in review_tasks must be instances of ReviewTask")

        answer_map = {}
        for ans in self.answers:
            if (
                ans.answer_key.form_id != self.form_id
                or ans.answer_key.form_version != self.form_version
                or ans.answer_key.page_number != self.page_number
            ):
                raise ValueError(
                    "All answers must match the page's form_id, form_version, and page_number"
                )

            if ans.answer_key in answer_map:
                raise ValueError(f"Duplicate answer_key found: {ans.answer_key}")
            answer_map[ans.answer_key] = ans

        pending_tasks = {}
        for task in self.review_tasks:
            if (
                task.answer_key.form_id != self.form_id
                or task.answer_key.form_version != self.form_version
                or task.answer_key.page_number != self.page_number
            ):
                raise ValueError(
                    "All review_tasks must match the page's form_id, form_version, and page_number"
                )

            if task.answer_key not in answer_map:
                raise ValueError(f"ReviewTask references missing answer: {task.answer_key}")

            if task.status == "pending":
                if task.answer_key in pending_tasks:
                    raise ValueError(f"Multiple pending ReviewTasks for answer: {task.answer_key}")
                pending_tasks[task.answer_key] = task

                if answer_map[task.answer_key].resolution_status != "needs_review":
                    raise ValueError(
                        f"Pending ReviewTask references resolved answer: {task.answer_key}"
                    )

        for ans in self.answers:
            if ans.resolution_status == "needs_review" and ans.answer_key not in pending_tasks:
                raise ValueError(
                    f"Answer {ans.answer_key} needs_review but has no corresponding "
                    "pending ReviewTask"
                )

    @property
    def page_status(self) -> Literal["resolved", "review_required"]:
        if any(ans.resolution_status == "needs_review" for ans in self.answers):
            return "review_required"
        return "resolved"

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OptionDef:
    option_id: str
    value: int | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if type(self.option_id) is not str:
            raise ValueError("option_id must be a string")
        if not self.option_id or not self.option_id.strip():
            raise ValueError("option_id cannot be empty")
        if self.value is not None and type(self.value) is not int:
            raise ValueError("value must be an int or None")


@dataclass(frozen=True)
class QuestionDef:
    question_id: str
    response_type: Literal["single_select", "multi_select", "rating"]
    mark_strategy: Literal["circle", "checkbox", "rating"]
    options: tuple[OptionDef, ...]
    min_selections: int = 0
    max_selections: int | None = None

    def __post_init__(self) -> None:
        if type(self.question_id) is not str:
            raise ValueError("question_id must be a string")
        if not self.question_id or not self.question_id.strip():
            raise ValueError("question_id cannot be empty")

        valid_response_types = ("single_select", "multi_select", "rating")
        if self.response_type not in valid_response_types:
            raise ValueError(f"Invalid response_type {self.response_type}")

        valid_mark_strategies = ("circle", "checkbox", "rating")
        if self.mark_strategy not in valid_mark_strategies:
            raise ValueError(f"Invalid mark_strategy {self.mark_strategy}")

        if not isinstance(self.options, tuple):
            raise ValueError("options must be a tuple")

        for opt in self.options:
            if not isinstance(opt, OptionDef):
                raise ValueError("All elements in options must be instances of OptionDef")

        if not self.options:
            raise ValueError("options cannot be empty")

        seen_ids = set()
        for opt in self.options:
            if opt.option_id in seen_ids:
                raise ValueError(f"Duplicate option_id found: {opt.option_id}")
            seen_ids.add(opt.option_id)

        if type(self.min_selections) is not int:
            raise ValueError("min_selections must be an int")
        if self.min_selections < 0:
            raise ValueError("min_selections cannot be negative")

        if self.min_selections > len(self.options):
            raise ValueError("min_selections cannot exceed the number of options")

        if self.max_selections is not None:
            if type(self.max_selections) is not int:
                raise ValueError("max_selections must be an int or None")
            if self.max_selections < self.min_selections:
                raise ValueError("max_selections cannot be less than min_selections")
            if self.max_selections <= 0:
                raise ValueError("max_selections must be > 0")
            if self.max_selections > len(self.options):
                raise ValueError("max_selections cannot exceed the number of options")

        if self.response_type == "single_select":
            if self.max_selections != 1:
                raise ValueError("single_select must have max_selections=1")
        elif self.response_type == "rating":
            if self.max_selections != 1 or self.min_selections != 1:
                raise ValueError("rating must have min_selections=1 and max_selections=1")

            seen_values = set()
            for opt in self.options:
                if type(opt.value) is not int:
                    raise ValueError(f"Rating options must have an integer value, got {opt.value}")
                if opt.value in seen_values:
                    raise ValueError(
                        f"Rating options must have unique values, duplicate {opt.value}"
                    )
                seen_values.add(opt.value)


@dataclass(frozen=True)
class FormProfile:
    form_id: str
    form_version: str
    questions: tuple[QuestionDef, ...]

    def __post_init__(self) -> None:
        if type(self.form_id) is not str:
            raise ValueError("form_id must be a string")
        if type(self.form_version) is not str:
            raise ValueError("form_version must be a string")

        if not self.form_id or not self.form_id.strip():
            raise ValueError("form_id cannot be empty")
        if not self.form_version or not self.form_version.strip():
            raise ValueError("form_version cannot be empty")

        if not isinstance(self.questions, tuple):
            raise ValueError("questions must be a tuple")

        for q in self.questions:
            if not isinstance(q, QuestionDef):
                raise ValueError("All elements in questions must be instances of QuestionDef")

        if not self.questions:
            raise ValueError("questions cannot be empty")

        seen_ids = set()
        for q in self.questions:
            if q.question_id in seen_ids:
                raise ValueError(f"Duplicate question_id found: {q.question_id}")
            seen_ids.add(q.question_id)

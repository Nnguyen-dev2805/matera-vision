from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from matera.core.errors import ProfileValidationError


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


def _raise_error(path: str, field: str, code: str, reason: str) -> None:
    raise ProfileValidationError(path, field, code, reason)


def load_semantic_profile(path: Path) -> FormProfile:
    """Loads and strictly validates a semantic profile JSON."""
    import json

    path_str = str(path)
    if not path.exists():
        _raise_error(path_str, "", "FILE_NOT_FOUND", "Profile file does not exist")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _raise_error(path_str, "", "INVALID_JSON", str(e))

    if not isinstance(data, dict):
        _raise_error(path_str, "", "INVALID_ROOT", "Root must be a JSON object")

    form_id = data.get("form_id")
    if not isinstance(form_id, str) or not form_id:
        _raise_error(path_str, "form_id", "INVALID_TYPE", "must be a non-empty string")

    form_version = data.get("form_version")
    if not isinstance(form_version, str) or not form_version:
        _raise_error(path_str, "form_version", "INVALID_TYPE", "must be a non-empty string")

    raw_questions = data.get("questions")

    if not isinstance(raw_questions, list):
        _raise_error(path_str, "questions", "INVALID_TYPE", "must be a list")

    if not raw_questions:
        _raise_error(path_str, "questions", "EMPTY_LIST", "Questions cannot be empty")

    seen_q_ids: set[str] = set()
    questions = []
    for i, q in enumerate(raw_questions):
        field_q = f"questions[{i}]"
        if not isinstance(q, dict):
            _raise_error(path_str, field_q, "INVALID_TYPE", "must be an object")

        q_id = q.get("question_id")
        if not isinstance(q_id, str) or not q_id:
            _raise_error(
                path_str, f"{field_q}.question_id", "INVALID_TYPE", "must be a non-empty string"
            )

        if q_id in seen_q_ids:
            _raise_error(
                path_str, f"{field_q}.question_id", "DUPLICATE_ID", "Question ID must be unique"
            )
        seen_q_ids.add(q_id)

        resp_type = q.get("response_type")
        if not isinstance(resp_type, str) or not resp_type:
            _raise_error(
                path_str, f"{field_q}.response_type", "INVALID_TYPE", "must be a non-empty string"
            )

        mark_strat = q.get("mark_strategy")
        if not isinstance(mark_strat, str) or not mark_strat:
            _raise_error(
                path_str, f"{field_q}.mark_strategy", "INVALID_TYPE", "must be a non-empty string"
            )

        raw_options = q.get("options")
        if not isinstance(raw_options, list):
            _raise_error(path_str, f"{field_q}.options", "INVALID_TYPE", "must be a list")

        if not raw_options:
            _raise_error(path_str, f"{field_q}.options", "EMPTY_LIST", "Options cannot be empty")

        options = []
        seen_opt_ids: set[str] = set()
        for j, opt in enumerate(raw_options):
            if isinstance(opt, str):
                o_id = opt
                try:
                    options.append(OptionDef(option_id=o_id))
                except ValueError as e:
                    _raise_error(path_str, f"{field_q}.options[{j}]", "INVALID_FIELD", str(e))
            elif isinstance(opt, dict):
                o_id = opt.get("option_id")
                try:
                    options.append(
                        OptionDef(
                            option_id=o_id,
                            value=opt.get("value"),
                            label=opt.get("label"),
                        )
                    )
                except ValueError as e:
                    _raise_error(path_str, f"{field_q}.options[{j}]", "INVALID_FIELD", str(e))
            else:
                _raise_error(
                    path_str,
                    f"{field_q}.options[{j}]",
                    "INVALID_TYPE",
                    "must be a string or object",
                )

            if o_id in seen_opt_ids:
                _raise_error(
                    path_str,
                    f"{field_q}.options[{j}]",
                    "DUPLICATE_ID",
                    "Option ID must be unique within question",
                )
            seen_opt_ids.add(o_id)
        try:
            kwargs = {
                "question_id": q_id,
                "response_type": resp_type,
                "mark_strategy": mark_strat,
                "options": tuple(options),
            }
            if "min_selections" in q:
                kwargs["min_selections"] = q["min_selections"]
            if "max_selections" in q:
                kwargs["max_selections"] = q["max_selections"]
            elif resp_type == "single_select":
                kwargs["max_selections"] = 1
            elif resp_type == "rating":
                kwargs["max_selections"] = 1
                if "min_selections" not in q:
                    kwargs["min_selections"] = 1

            questions.append(QuestionDef(**kwargs))
        except ValueError as e:
            _raise_error(path_str, field_q, "INVALID_FIELD", str(e))

    try:
        return FormProfile(form_id=form_id, form_version=form_version, questions=tuple(questions))
    except ValueError as e:
        _raise_error(path_str, "", "INVALID_FIELD", str(e))

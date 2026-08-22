from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from matera.core.profile import FormProfile


@dataclass(frozen=True)
class ColumnDef:
    column_name: str
    question_id: str
    option_id: str
    value_encoding: Literal["binary"] = "binary"

    def __post_init__(self) -> None:
        if type(self.column_name) is not str:
            raise ValueError("column_name must be a string")
        if type(self.question_id) is not str:
            raise ValueError("question_id must be a string")
        if type(self.option_id) is not str:
            raise ValueError("option_id must be a string")

        if self.value_encoding != "binary":
            raise ValueError(f"Invalid value_encoding {self.value_encoding}")

        if not self.column_name or not self.column_name.strip():
            raise ValueError("column_name cannot be empty")
        if not self.question_id or not self.question_id.strip():
            raise ValueError("question_id cannot be empty")
        if not self.option_id or not self.option_id.strip():
            raise ValueError("option_id cannot be empty")


@dataclass(frozen=True)
class ExcelSchema:
    form_id: str
    form_version: str
    columns: tuple[ColumnDef, ...]

    def __post_init__(self) -> None:
        if type(self.form_id) is not str:
            raise ValueError("form_id must be a string")
        if type(self.form_version) is not str:
            raise ValueError("form_version must be a string")

        if not self.form_id or not self.form_id.strip():
            raise ValueError("form_id cannot be empty")
        if not self.form_version or not self.form_version.strip():
            raise ValueError("form_version cannot be empty")

        if not isinstance(self.columns, tuple):
            raise ValueError("columns must be a tuple")

        if not self.columns:
            raise ValueError("columns cannot be empty")

        seen_names = set()
        seen_targets = set()

        for col in self.columns:
            if not isinstance(col, ColumnDef):
                raise ValueError("All elements in columns must be instances of ColumnDef")

            if col.column_name in seen_names:
                raise ValueError(f"Duplicate column_name found: {col.column_name}")
            seen_names.add(col.column_name)

            target = (col.question_id, col.option_id)
            if target in seen_targets:
                raise ValueError(
                    f"Duplicate mapping for "
                    f"question_id={col.question_id}, "
                    f"option_id={col.option_id}"
                )
            seen_targets.add(target)

    @classmethod
    def from_profile(cls, profile: "FormProfile") -> "ExcelSchema":
        """
        Creates an ExcelSchema that strictly matches the given FormProfile's questions and options.
        """
        columns = []
        for q in profile.questions:
            for opt in q.options:
                col_name = f"{q.question_id}_{opt.option_id}"
                columns.append(ColumnDef(col_name, q.question_id, opt.option_id))
        return cls(profile.form_id, profile.form_version, tuple(columns))

    def validate_against(self, profile: "FormProfile") -> None:
        """Validates that this schema maps exactly to the given FormProfile."""
        if self.form_id != profile.form_id or self.form_version != profile.form_version:
            raise ValueError("Form ID or version mismatch between schema and profile")

        expected_mapping = []
        for q in profile.questions:
            for opt in q.options:
                expected_mapping.append((q.question_id, opt.option_id))

        actual_mapping = [(c.question_id, c.option_id) for c in self.columns]
        if expected_mapping != actual_mapping:
            raise ValueError("Columns do not strictly match profile options in mapping or order")

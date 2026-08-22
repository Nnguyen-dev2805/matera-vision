import pytest

from matera.core.contracts import AnswerKey, NormalizedAnswer, NormalizedPageResult
from matera.core.profile import FormProfile, OptionDef, QuestionDef
from matera.export.schema import ColumnDef, ExcelSchema


def test_integration_flow():
    # 1. Define a minimal profile for Matera v1
    opt_a = OptionDef("A", 1)
    opt_b = OptionDef("B", 2)
    q1 = QuestionDef(
        "Q1", "single_select", "circle", (opt_a, opt_b), min_selections=1, max_selections=1
    )

    r0 = OptionDef("0", 0)
    r1 = OptionDef("1", 1)
    q2 = QuestionDef("Q2", "rating", "rating", (r0, r1), min_selections=1, max_selections=1)

    profile = FormProfile("matera-pre", "v1", (q1, q2))

    # 2. Generate and validate the Excel Schema
    schema = ExcelSchema.from_profile(profile)
    schema.validate_against(profile)

    # Ordering and column check
    assert len(schema.columns) == 4
    assert schema.columns[0].column_name == "Q1_A"
    assert schema.columns[1].column_name == "Q1_B"
    assert schema.columns[2].column_name == "Q2_0"
    assert schema.columns[3].column_name == "Q2_1"

    # 3. Represent an Answer mapped against the schema
    ans_q1_a = NormalizedAnswer(
        answer_key=AnswerKey("matera-pre", "v1", 1, "Q1", "A"),
        selected=True,
        resolution_status="resolved",
        decision_source="deterministic",
    )
    ans_q1_b = NormalizedAnswer(
        answer_key=AnswerKey("matera-pre", "v1", 1, "Q1", "B"),
        selected=False,
        resolution_status="resolved",
        decision_source="deterministic",
    )

    page = NormalizedPageResult("matera-pre", "v1", 1, (ans_q1_a, ans_q1_b), ())
    assert page.page_status == "resolved"


def test_schema_validate_against_mismatch():
    opt_a = OptionDef("A", 1)
    q1 = QuestionDef("Q1", "single_select", "circle", (opt_a,), min_selections=1, max_selections=1)
    profile = FormProfile("matera-pre", "v1", (q1,))

    # Wrong form_id
    schema_wrong_id = ExcelSchema("matera-wrong", "v1", ExcelSchema.from_profile(profile).columns)
    with pytest.raises(ValueError, match="Form ID or version mismatch"):
        schema_wrong_id.validate_against(profile)

    # Missing column (mismatched schema)
    col = ColumnDef("WRONG_COL", "WRONG", "COL")
    schema_missing = ExcelSchema("matera-pre", "v1", (col,))
    with pytest.raises(ValueError, match="Columns do not strictly match profile options in mapping or order"):
        schema_missing.validate_against(profile)

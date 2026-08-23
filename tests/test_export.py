from pathlib import Path

from matera.core.profile import FormProfile, OptionDef, QuestionDef
from matera.export.excel import export_to_excel


def test_export_to_excel_skeleton(tmp_path: Path):
    profile = FormProfile(
        form_id="test",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="q1",
                response_type="single_select",
                mark_strategy="circle",
                options=(OptionDef("o1"),),
                max_selections=1,
            ),
        ),
    )

    export_to_excel([], profile, tmp_path / "out.xlsx")
    # Will fail until implemented, but confirms import works.

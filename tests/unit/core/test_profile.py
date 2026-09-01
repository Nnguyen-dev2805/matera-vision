from pathlib import Path

import pytest

from matera.core.errors import ProfileValidationError
from matera.core.profile import FormProfile, OptionDef, QuestionDef, load_semantic_profile


def test_option_def_valid():
    opt = OptionDef("A", 1, "Label A")
    assert opt.option_id == "A"


def test_option_def_empty_id():
    with pytest.raises(ValueError, match="option_id cannot be empty"):
        OptionDef("  ", 1)


def test_option_def_invalid_types():
    with pytest.raises(ValueError, match="option_id must be a string"):
        OptionDef(1, 1)  # type: ignore


def test_option_def_invalid_value_type():
    with pytest.raises(ValueError, match="value must be an int or None"):
        OptionDef("A", True)  # type: ignore
    with pytest.raises(ValueError, match="value must be an int or None"):
        OptionDef("A", "1")  # type: ignore


def test_question_def_valid_single_select():
    opt1 = OptionDef("A", 1)
    q = QuestionDef(
        question_id="Q1",
        response_type="single_select",
        mark_strategy="circle",
        options=(opt1,),
        min_selections=0,
        max_selections=1,
    )
    assert q.question_id == "Q1"


def test_question_def_invalid_response_type():
    opt1 = OptionDef("A", 1)
    with pytest.raises(ValueError, match="Invalid response_type"):
        QuestionDef("Q1", "unknown", "circle", (opt1,))  # type: ignore


def test_question_def_invalid_types():
    opt1 = OptionDef("A", 1)
    with pytest.raises(ValueError, match="question_id must be a string"):
        QuestionDef(1, "single_select", "circle", (opt1,))  # type: ignore
    with pytest.raises(ValueError, match="All elements in options must be instances of OptionDef"):
        QuestionDef("Q1", "single_select", "circle", ("not_an_option",))  # type: ignore


def test_question_def_invalid_mark_strategy():
    opt1 = OptionDef("A", 1)
    with pytest.raises(ValueError, match="Invalid mark_strategy unknown"):
        QuestionDef(
            question_id="Q1",
            response_type="single_select",
            mark_strategy="unknown",  # type: ignore
            options=(opt1,),
            max_selections=1,
        )


def test_question_def_empty_options():
    with pytest.raises(ValueError, match="options cannot be empty"):
        QuestionDef(
            question_id="Q1",
            response_type="single_select",
            mark_strategy="circle",
            options=(),
            max_selections=1,
        )


def test_question_def_duplicate_option_id():
    opt1 = OptionDef("A", 1)
    opt2 = OptionDef("A", 2)
    with pytest.raises(ValueError, match="Duplicate option_id found"):
        QuestionDef(
            question_id="Q1",
            response_type="multi_select",
            mark_strategy="checkbox",
            options=(opt1, opt2),
        )


def test_question_def_invalid_selections():
    opt1 = OptionDef("A", 1)
    with pytest.raises(ValueError, match="min_selections cannot be negative"):
        QuestionDef(
            question_id="Q1",
            response_type="multi_select",
            mark_strategy="checkbox",
            options=(opt1,),
            min_selections=-1,
        )
    opt2 = OptionDef("B", 2)
    opt3 = OptionDef("C", 3)
    with pytest.raises(ValueError, match="max_selections cannot be less than min_selections"):
        QuestionDef(
            question_id="Q1",
            response_type="multi_select",
            mark_strategy="checkbox",
            options=(opt1, opt2, opt3),
            min_selections=2,
            max_selections=1,
        )


def test_question_def_single_select_validation():
    opt1 = OptionDef("A", 1)
    opt2 = OptionDef("B", 2)
    with pytest.raises(ValueError, match="single_select must have max_selections=1"):
        QuestionDef(
            question_id="Q1",
            response_type="single_select",
            mark_strategy="circle",
            options=(opt1, opt2),
            max_selections=2,
        )


def test_question_def_rating_validation():
    opt1 = OptionDef("1", 1)
    with pytest.raises(ValueError, match="rating must have min_selections=1 and max_selections=1"):
        QuestionDef(
            question_id="Q1",
            response_type="rating",
            mark_strategy="rating",
            options=(opt1,),
            min_selections=0,
            max_selections=1,
        )


def test_form_profile_valid():
    opt1 = OptionDef("A", 1)
    q = QuestionDef(
        question_id="Q1",
        response_type="single_select",
        mark_strategy="circle",
        options=(opt1,),
        max_selections=1,
    )
    profile = FormProfile("form1", "v1", (q,))
    assert profile.form_id == "form1"


def test_form_profile_empty_identifiers():
    opt = OptionDef("A", 1)
    q = QuestionDef("Q1", "single_select", "circle", (opt,), max_selections=1)
    with pytest.raises(ValueError, match="form_id cannot be empty"):
        FormProfile(" ", "v1", (q,))
    with pytest.raises(ValueError, match="form_version cannot be empty"):
        FormProfile("form1", " ", (q,))


def test_form_profile_invalid_types():
    opt = OptionDef("A", 1)
    q = QuestionDef("Q1", "single_select", "circle", (opt,), max_selections=1)
    with pytest.raises(ValueError, match="form_id must be a string"):
        FormProfile(1, "v1", (q,))  # type: ignore
    with pytest.raises(ValueError, match="form_version must be a string"):
        FormProfile("form1", 1, (q,))  # type: ignore
    with pytest.raises(
        ValueError, match="All elements in questions must be instances of QuestionDef"
    ):
        FormProfile("form1", "v1", ("not_a_question",))  # type: ignore


def test_form_profile_duplicate_question_id():
    opt1 = OptionDef("A", 1)
    q1 = QuestionDef(
        question_id="Q1",
        response_type="single_select",
        mark_strategy="circle",
        options=(opt1,),
        max_selections=1,
    )
    q2 = QuestionDef(
        question_id="Q1",
        response_type="multi_select",
        mark_strategy="checkbox",
        options=(opt1,),
    )
    with pytest.raises(ValueError, match="Duplicate question_id found"):
        FormProfile("form1", "v1", (q1, q2))


def test_question_def_valid_multi_select():
    opt1 = OptionDef("A", 1)
    opt2 = OptionDef("B", 2)
    q = QuestionDef(
        question_id="Q1",
        response_type="multi_select",
        mark_strategy="checkbox",
        options=(opt1, opt2),
        min_selections=0,
        max_selections=2,
    )
    assert q.response_type == "multi_select"


def test_question_def_valid_rating():
    opt1 = OptionDef("R1", 1)
    opt2 = OptionDef("R2", 2)
    q = QuestionDef(
        question_id="Q1",
        response_type="rating",
        mark_strategy="rating",
        options=(opt1, opt2),
        min_selections=1,
        max_selections=1,
    )
    assert q.response_type == "rating"


def test_question_def_selections_exceed_options():
    opt1 = OptionDef("A", None)
    with pytest.raises(ValueError, match="min_selections cannot exceed the number of options"):
        QuestionDef(
            question_id="Q1",
            response_type="multi_select",
            mark_strategy="checkbox",
            options=(opt1,),
            min_selections=2,
        )
    with pytest.raises(ValueError, match="max_selections cannot exceed the number of options"):
        QuestionDef(
            question_id="Q1",
            response_type="multi_select",
            mark_strategy="checkbox",
            options=(opt1,),
            min_selections=0,
            max_selections=2,
        )


def test_question_def_rating_invalid_values():
    opt1 = OptionDef("R1", 1)
    opt2 = OptionDef("R2", 1)
    with pytest.raises(ValueError, match="Rating options must have unique values"):
        QuestionDef(
            question_id="Q1",
            response_type="rating",
            mark_strategy="rating",
            options=(opt1, opt2),
            min_selections=1,
            max_selections=1,
        )

    opt_none = OptionDef("R3", None)
    with pytest.raises(ValueError, match="Rating options must have an integer value"):
        QuestionDef(
            question_id="Q2",
            response_type="rating",
            mark_strategy="rating",
            options=(opt1, opt_none),
            min_selections=1,
            max_selections=1,
        )


def test_form_profile_empty_questions():
    with pytest.raises(ValueError, match="questions cannot be empty"):
        FormProfile("form1", "v1", ())


def test_question_def_empty_question_id():
    opt1 = OptionDef("A", 1)
    with pytest.raises(ValueError, match="question_id cannot be empty"):
        QuestionDef(
            question_id="  ",
            response_type="single_select",
            mark_strategy="circle",
            options=(opt1,),
        )


def test_question_def_max_selections_invalid():
    opt1 = OptionDef("A", 1)
    with pytest.raises(ValueError, match="max_selections must be > 0"):
        QuestionDef(
            question_id="Q1",
            response_type="multi_select",
            mark_strategy="checkbox",
            options=(opt1,),
            max_selections=0,
        )


def test_immutability():
    from dataclasses import FrozenInstanceError

    opt = OptionDef("A", 1)
    with pytest.raises(FrozenInstanceError):
        opt.value = 2  # type: ignore


def test_question_def_invalid_selection_types():
    opt1 = OptionDef("A", 1)
    with pytest.raises(ValueError, match="min_selections must be an int"):
        QuestionDef(
            question_id="Q1",
            response_type="multi_select",
            mark_strategy="checkbox",
            options=(opt1,),
            min_selections=0.5,  # type: ignore
        )
    with pytest.raises(ValueError, match="max_selections must be an int or None"):
        QuestionDef(
            question_id="Q1",
            response_type="multi_select",
            mark_strategy="checkbox",
            options=(opt1,),
            max_selections=True,  # type: ignore
        )


def test_question_def_mutable_options_rejected():
    opt1 = OptionDef("A", 1)
    options_list = [opt1]
    with pytest.raises(ValueError, match="options must be a tuple"):
        QuestionDef(
            question_id="Q1",
            response_type="single_select",
            mark_strategy="circle",
            options=options_list,  # type: ignore
        )


def test_form_profile_mutable_questions_rejected():
    opt1 = OptionDef("A", 1)
    q1 = QuestionDef(
        question_id="Q1",
        response_type="single_select",
        mark_strategy="circle",
        options=(opt1,),
        max_selections=1,
    )
    questions_list = [q1]
    with pytest.raises(ValueError, match="questions must be a tuple"):
        FormProfile("form1", "v1", questions_list)  # type: ignore


def test_load_semantic_profile_valid(tmp_path: Path):
    json_data = """{
        "form_id": "matera-pre",
        "form_version": "v1",
        "questions": [
            {
                "question_id": "Q1",
                "response_type": "single_select",
                "mark_strategy": "checkbox",
                "options": ["a", "b"]
            },
            {
                "question_id": "Q2",
                "response_type": "rating",
                "mark_strategy": "rating",
                "options": [
                    {"option_id": "r1", "value": 1, "label": "Poor"},
                    {"option_id": "r2", "value": 2, "label": "Good"}
                ],
                "min_selections": 1,
                "max_selections": 1
            }
        ]
    }"""
    p = tmp_path / "valid.json"
    p.write_text(json_data, encoding="utf-8")

    profile = load_semantic_profile(p)
    assert profile.form_id == "matera-pre"
    assert profile.form_version == "v1"
    assert len(profile.questions) == 2

    q1 = profile.questions[0]
    assert q1.question_id == "Q1"
    assert len(q1.options) == 2
    assert q1.options[0].option_id == "a"
    assert q1.options[0].value is None

    q2 = profile.questions[1]
    assert q2.question_id == "Q2"
    assert len(q2.options) == 2
    assert q2.options[0].value == 1


def test_load_semantic_profile_missing_file(tmp_path: Path):
    with pytest.raises(ProfileValidationError) as exc:
        load_semantic_profile(tmp_path / "nonexistent.json")
    assert exc.value.error_code == "FILE_NOT_FOUND"


def test_load_semantic_profile_invalid_json(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{bad json")
    with pytest.raises(ProfileValidationError) as exc:
        load_semantic_profile(p)
    assert exc.value.error_code == "INVALID_JSON"


def test_load_semantic_profile_invalid_root(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("[]")
    with pytest.raises(ProfileValidationError) as exc:
        load_semantic_profile(p)
    assert exc.value.error_code == "INVALID_ROOT"


def test_load_semantic_profile_missing_fields(tmp_path: Path):
    p = tmp_path / "bad.json"

    # Missing form_id
    p.write_text('{"form_version": "v1", "questions": []}')
    with pytest.raises(ProfileValidationError) as exc:
        load_semantic_profile(p)
    assert exc.value.field_path == "form_id"
    assert exc.value.error_code == "INVALID_TYPE"

    # Missing questions
    p.write_text('{"form_id": "f", "form_version": "v1"}')
    with pytest.raises(ProfileValidationError) as exc:
        load_semantic_profile(p)
    assert exc.value.field_path == "questions"
    assert exc.value.error_code == "INVALID_TYPE"

    # Empty questions
    p.write_text('{"form_id": "f", "form_version": "v1", "questions": []}')
    with pytest.raises(ProfileValidationError) as exc:
        load_semantic_profile(p)
    assert exc.value.field_path == "questions"
    assert exc.value.error_code == "EMPTY_LIST"


def test_load_semantic_profile_invalid_question(tmp_path: Path):
    p = tmp_path / "bad.json"

    # Missing question_id
    p.write_text('{"form_id": "f", "form_version": "v1", "questions": [{}]}')
    with pytest.raises(ProfileValidationError) as exc:
        load_semantic_profile(p)
    assert exc.value.field_path == "questions[0].question_id"

    # Duplicate question_id
    p.write_text("""{
        "form_id": "f", "form_version": "v1", 
        "questions": [
            {
                "question_id": "Q1", "response_type": "single_select", 
                "mark_strategy": "checkbox", "options": ["a"]
            },
            {
                "question_id": "Q1", "response_type": "single_select", 
                "mark_strategy": "checkbox", "options": ["a"]
            }
        ]
    }""")
    with pytest.raises(ProfileValidationError) as exc:
        load_semantic_profile(p)
    assert exc.value.field_path == "questions[1].question_id"
    assert exc.value.error_code == "DUPLICATE_ID"

    # Empty options
    p.write_text("""{
        "form_id": "f", "form_version": "v1", 
        "questions": [
            {
                "question_id": "Q1", "response_type": "single_select", 
                "mark_strategy": "checkbox", "options": []
            }
        ]
    }""")
    with pytest.raises(ProfileValidationError) as exc:
        load_semantic_profile(p)
    assert exc.value.field_path == "questions[0].options"
    assert exc.value.error_code == "EMPTY_LIST"

    # Duplicate options
    p.write_text("""{
        "form_id": "f", "form_version": "v1", 
        "questions": [
            {
                "question_id": "Q1", "response_type": "single_select", 
                "mark_strategy": "checkbox", "options": ["a", "a"]
            }
        ]
    }""")
    with pytest.raises(ProfileValidationError) as exc:
        load_semantic_profile(p)
    assert exc.value.field_path == "questions[0].options[1]"
    assert exc.value.error_code == "DUPLICATE_ID"

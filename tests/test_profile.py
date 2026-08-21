import pytest

from matera.core.profile import FormProfile, OptionDef, QuestionDef


def test_option_def_valid():
    opt = OptionDef("A", 1, "Label A")
    assert opt.option_id == "A"


def test_option_def_empty_id():
    with pytest.raises(ValueError, match="option_id cannot be empty"):
        OptionDef("  ", 1)


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
    with pytest.raises(ValueError, match="Invalid response_type unknown"):
        QuestionDef(
            question_id="Q1",
            response_type="unknown",  # type: ignore
            mark_strategy="circle",
            options=(opt1,),
        )


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
    with pytest.raises(ValueError, match="form_id cannot be empty"):
        FormProfile(" ", "v1", ())
    with pytest.raises(ValueError, match="form_version cannot be empty"):
        FormProfile("form1", "", ())


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

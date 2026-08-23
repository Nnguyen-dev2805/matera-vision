import pytest

from matera.core.profile import FormProfile, OptionDef, QuestionDef
from matera.vision.contracts import RoutingConfig
from matera.vision.routing import route_page


def test_routing_config_validation():
    # Valid
    config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)
    assert config.low_threshold == 0.2
    assert config.high_threshold == 0.6

    # Invalid: low >= high
    with pytest.raises(ValueError, match="Thresholds must satisfy"):
        RoutingConfig(low_threshold=0.6, high_threshold=0.4)

    # Invalid: out of bounds
    with pytest.raises(ValueError, match="Thresholds must satisfy"):
        RoutingConfig(low_threshold=-0.1, high_threshold=0.6)
    with pytest.raises(ValueError, match="Thresholds must satisfy"):
        RoutingConfig(low_threshold=0.2, high_threshold=1.1)


def test_route_page_skeleton():
    profile = FormProfile(
        form_id="test",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="q1",
                response_type="single_select",
                mark_strategy="circle",
                options=(OptionDef(option_id="o1", value=1),),
                min_selections=1,
                max_selections=1,
            ),
        ),
    )
    from pathlib import Path

    from matera.vision.contracts import MarkScore, ROIFeature

    mock_feature = ROIFeature(0.0, 0.0, 0, 0.0, 0.0)
    scores = [
        MarkScore("q1", "o1", 0.1, "circle", "diff", mock_feature, Path("dummy.png")),
    ]
    result = route_page(mark_scores=scores, profile=profile, page_number=1)

    assert result.form_id == "test"
    assert result.form_version == "v1"
    assert result.page_number == 1
    assert len(result.answers) == 1


def test_route_page_option_level_scoring():
    profile = FormProfile(
        form_id="test",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="q1",
                response_type="single_select",
                mark_strategy="circle",
                options=(
                    OptionDef(option_id="o1", value=1),
                    OptionDef(option_id="o2", value=2),
                    OptionDef(option_id="o3", value=3),
                ),
                min_selections=1,
                max_selections=1,
            ),
        ),
    )

    from matera.vision.contracts import MarkScore, ROIFeature

    mock_feature = ROIFeature(
        dark_pixel_ratio=0.0,
        foreground_area_ratio=0.0,
        contour_count=0,
        largest_component_ratio=0.0,
        bbox_fill_ratio=0.0,
    )

    # 1. low (< 0.2)
    # 2. high (>= 0.6)
    # 3. ambiguous (0.2 <= score < 0.6)
    from pathlib import Path

    scores = [
        MarkScore("q1", "o1", 0.1, "circle", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o2", 0.8, "circle", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o3", 0.4, "circle", "diff", mock_feature, Path("dummy.png")),
    ]

    config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)
    result = route_page(mark_scores=scores, profile=profile, page_number=1, config=config)

    assert len(result.answers) == 3

    a1 = next(a for a in result.answers if a.answer_key.option_id == "o1")
    assert a1.selected is False
    assert a1.resolution_status == "resolved"
    assert a1.deterministic_score == 0.1

    a2 = next(a for a in result.answers if a.answer_key.option_id == "o2")
    assert a2.selected is True
    assert a2.resolution_status == "resolved"
    assert a2.deterministic_score == 0.8

    a3 = next(a for a in result.answers if a.answer_key.option_id == "o3")
    assert a3.selected is None
    assert a3.resolution_status == "needs_review"
    assert a3.deterministic_score == 0.4

    # Check ReviewTasks
    assert len(result.review_tasks) == 1
    rt = result.review_tasks[0]
    assert rt.answer_key.option_id == "o3"
    assert rt.status == "pending"
    assert "0.4" in rt.reason

    assert result.page_status == "review_required"


def test_route_page_over_selection():
    profile = FormProfile(
        form_id="test",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="q1",
                response_type="single_select",
                mark_strategy="circle",
                options=(
                    OptionDef(option_id="o1", value=1),
                    OptionDef(option_id="o2", value=2),
                ),
                min_selections=1,
                max_selections=1,
            ),
        ),
    )

    from matera.vision.contracts import MarkScore, ROIFeature

    mock_feature = ROIFeature(0.0, 0.0, 0, 0.0, 0.0)

    from pathlib import Path

    scores = [
        MarkScore("q1", "o1", 0.8, "circle", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o2", 0.9, "circle", "diff", mock_feature, Path("dummy.png")),
    ]

    config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)
    result = route_page(mark_scores=scores, profile=profile, page_number=1, config=config)

    assert len(result.answers) == 2
    for ans in result.answers:
        assert ans.selected is None
        assert ans.resolution_status == "needs_review"

    assert len(result.review_tasks) == 2
    for rt in result.review_tasks:
        assert rt.status == "pending"
        assert "Over-selection" in rt.reason

    assert result.page_status == "review_required"


def test_route_page_under_selection():
    profile = FormProfile(
        form_id="test",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="q1",
                response_type="single_select",
                mark_strategy="circle",
                options=(
                    OptionDef(option_id="o1", value=1),
                    OptionDef(option_id="o2", value=2),
                ),
                min_selections=1,
                max_selections=1,
            ),
        ),
    )

    from matera.vision.contracts import MarkScore, ROIFeature

    mock_feature = ROIFeature(0.0, 0.0, 0, 0.0, 0.0)

    # 0 options > 0.6 -> under-selection (min_selections=1)
    from pathlib import Path

    scores = [
        MarkScore("q1", "o1", 0.1, "circle", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o2", 0.15, "circle", "diff", mock_feature, Path("dummy.png")),
    ]

    config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)
    result = route_page(mark_scores=scores, profile=profile, page_number=1, config=config)

    assert len(result.answers) == 2

    a1 = next(a for a in result.answers if a.answer_key.option_id == "o1")
    assert a1.selected is False
    assert a1.resolution_status == "resolved"

    a2 = next(a for a in result.answers if a.answer_key.option_id == "o2")
    assert a2.selected is None
    assert a2.resolution_status == "needs_review"

    assert len(result.review_tasks) == 1
    rt = result.review_tasks[0]
    assert rt.answer_key.option_id == "o2"
    assert "Under-selection" in rt.reason
    assert "0.150" in rt.reason

    assert result.page_status == "review_required"


def test_route_page_fail_fast():
    profile = FormProfile(
        form_id="test",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="q1",
                response_type="single_select",
                mark_strategy="circle",
                options=(
                    OptionDef(option_id="o1", value=1),
                    OptionDef(option_id="o2", value=2),
                ),
                min_selections=1,
                max_selections=1,
            ),
        ),
    )

    from pathlib import Path

    from matera.vision.contracts import MarkScore, ROIFeature

    mock_feature = ROIFeature(0.0, 0.0, 0, 0.0, 0.0)

    # 1. Missing MarkScore
    scores_missing = [
        MarkScore("q1", "o1", 0.1, "circle", "diff", mock_feature, Path("dummy.png")),
    ]
    with pytest.raises(ValueError, match="Missing MarkScore for options: q1.o2"):
        route_page(mark_scores=scores_missing, profile=profile, page_number=1)

    # 2. Duplicate MarkScore
    scores_duplicate = [
        MarkScore("q1", "o1", 0.1, "circle", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o2", 0.1, "circle", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o1", 0.1, "circle", "diff", mock_feature, Path("dummy.png")),
    ]
    with pytest.raises(ValueError, match="Duplicate MarkScore found for q1.o1"):
        route_page(mark_scores=scores_duplicate, profile=profile, page_number=1)

    # 3. Unknown MarkScore
    scores_unknown = [
        MarkScore("q1", "o1", 0.1, "circle", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o2", 0.1, "circle", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o3", 0.1, "circle", "diff", mock_feature, Path("dummy.png")),
    ]
    with pytest.raises(ValueError, match="Unknown or extra MarkScore found for q1.o3"):
        route_page(mark_scores=scores_unknown, profile=profile, page_number=1)

    # 4. Missing evidence path
    scores_no_evidence = [
        MarkScore("q1", "o1", 0.1, "circle", "diff", mock_feature, None),
        MarkScore("q1", "o2", 0.1, "circle", "diff", mock_feature, Path("dummy.png")),
    ]
    with pytest.raises(ValueError, match="Missing evidence_path in MarkScore for q1.o1"):
        route_page(mark_scores=scores_no_evidence, profile=profile, page_number=1)

    # 5. Mark strategy mismatch
    scores_mismatch = [
        MarkScore("q1", "o1", 0.1, "checkbox", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o2", 0.1, "circle", "diff", mock_feature, Path("dummy.png")),
    ]
    with pytest.raises(
        ValueError, match="Mark strategy mismatch for q1.o1: expected circle, got checkbox"
    ):
        route_page(mark_scores=scores_mismatch, profile=profile, page_number=1)


def test_route_page_exact_boundaries():
    profile = FormProfile(
        form_id="test",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="q1",
                response_type="single_select",
                mark_strategy="circle",
                options=(
                    OptionDef(option_id="o1", value=1),
                    OptionDef(option_id="o2", value=2),
                ),
                min_selections=1,
                max_selections=1,
            ),
        ),
    )

    from pathlib import Path

    from matera.vision.contracts import MarkScore, ROIFeature

    mock_feature = ROIFeature(0.0, 0.0, 0, 0.0, 0.0)

    scores = [
        MarkScore("q1", "o1", 0.2, "circle", "diff", mock_feature, Path("dummy.png")),  # exact low
        MarkScore("q1", "o2", 0.6, "circle", "diff", mock_feature, Path("dummy.png")),  # exact high
    ]

    config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)
    result = route_page(mark_scores=scores, profile=profile, page_number=1, config=config)

    a1 = next(a for a in result.answers if a.answer_key.option_id == "o1")
    # 0.2 is >= low_threshold (0.2). Wait, rule: < low -> false, >= high -> true.
    # So 0.2 is ambiguous!
    assert a1.selected is None
    assert a1.resolution_status == "needs_review"

    a2 = next(a for a in result.answers if a.answer_key.option_id == "o2")
    assert a2.selected is True
    assert a2.resolution_status == "resolved"


def test_route_page_rating_question():
    profile = FormProfile(
        form_id="test",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="q1",
                response_type="rating",
                mark_strategy="rating",
                options=(
                    OptionDef(option_id="o1", value=1),
                    OptionDef(option_id="o2", value=2),
                    OptionDef(option_id="o3", value=3),
                ),
                min_selections=1,
                max_selections=1,
            ),
        ),
    )

    from pathlib import Path

    from matera.vision.contracts import MarkScore, ROIFeature

    mock_feature = ROIFeature(0.0, 0.0, 0, 0.0, 0.0)

    # normal case
    scores = [
        MarkScore("q1", "o1", 0.1, "rating", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o2", 0.8, "rating", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o3", 0.1, "rating", "diff", mock_feature, Path("dummy.png")),
    ]

    config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)
    result = route_page(mark_scores=scores, profile=profile, page_number=1, config=config)

    assert result.page_status == "resolved"
    a2 = next(a for a in result.answers if a.answer_key.option_id == "o2")
    assert a2.selected is True


def test_route_page_checkbox_multi_select():
    profile = FormProfile(
        form_id="test",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="q1",
                response_type="multi_select",
                mark_strategy="checkbox",
                options=(
                    OptionDef(option_id="o1", value=1),
                    OptionDef(option_id="o2", value=2),
                    OptionDef(option_id="o3", value=3),
                ),
                min_selections=0,
                max_selections=3,
            ),
        ),
    )

    from pathlib import Path

    from matera.vision.contracts import MarkScore, ROIFeature

    mock_feature = ROIFeature(0.0, 0.0, 0, 0.0, 0.0)

    # 3 options selected -> should be fine (max=3)
    scores = [
        MarkScore("q1", "o1", 0.9, "checkbox", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o2", 0.8, "checkbox", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o3", 0.9, "checkbox", "diff", mock_feature, Path("dummy.png")),
    ]

    config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)
    result = route_page(mark_scores=scores, profile=profile, page_number=1, config=config)

    assert result.page_status == "resolved"
    for a in result.answers:
        assert a.selected is True

    # 0 options selected -> should be fine (min=0)
    scores_zero = [
        MarkScore("q1", "o1", 0.1, "checkbox", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o2", 0.1, "checkbox", "diff", mock_feature, Path("dummy.png")),
        MarkScore("q1", "o3", 0.1, "checkbox", "diff", mock_feature, Path("dummy.png")),
    ]
    result_zero = route_page(mark_scores=scores_zero, profile=profile, page_number=1, config=config)
    assert result_zero.page_status == "resolved"
    for a in result_zero.answers:
        assert a.selected is False

import pytest
from matera.vision.contracts import RoutingConfig
from matera.vision.routing import route_page
from matera.core.profile import FormProfile, QuestionDef, OptionDef
from matera.core.layout import PageLayout


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
                max_selections=1
            ),
        )
    )
    result = route_page(mark_scores=[], profile=profile, page_number=1)
    
    assert result.form_id == "test"
    assert result.form_version == "v1"
    assert result.page_number == 1
    assert result.answers == ()
    assert result.review_tasks == ()

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
                max_selections=1
            ),
        )
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
    scores = [
        MarkScore("q1", "o1", 0.1, "circle", "diff", mock_feature, None),
        MarkScore("q1", "o2", 0.8, "circle", "diff", mock_feature, None),
        MarkScore("q1", "o3", 0.4, "circle", "diff", mock_feature, None),
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

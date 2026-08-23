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

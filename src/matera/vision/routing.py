from matera.core.contracts import NormalizedPageResult
from matera.core.profile import FormProfile
from matera.vision.contracts import MarkScore, RoutingConfig


def route_page(
    mark_scores: list[MarkScore],
    profile: FormProfile,
    page_number: int,
    config: RoutingConfig | None = None,
) -> NormalizedPageResult:
    """
    Evaluates mark scores against deterministic thresholds and form constraints
    to produce a NormalizedPageResult.
    """
    config = config or RoutingConfig()

    # Skeleton implementation
    # TODO: Implement Task 8.2 and 8.3

    return NormalizedPageResult(
        form_id=profile.form_id,
        form_version=profile.form_version,
        page_number=page_number,
        answers=(),
        review_tasks=(),
    )

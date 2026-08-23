from matera.core.contracts import NormalizedPageResult
from matera.core.profile import FormProfile
from matera.vision.contracts import MarkScore, RoutingConfig
import uuid
from matera.core.contracts import AnswerKey, NormalizedAnswer, NormalizedPageResult, ReviewTask


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

    answers: list[NormalizedAnswer] = []
    review_tasks: list[ReviewTask] = []

    for mark_score in mark_scores:
        key = AnswerKey(
            form_id=profile.form_id,
            form_version=profile.form_version,
            page_number=page_number,
            question_id=mark_score.question_id,
            option_id=mark_score.option_id,
        )

        selected: bool | None = None
        resolution_status: str = "resolved"

        if mark_score.score < config.low_threshold:
            selected = False
        elif mark_score.score >= config.high_threshold:
            selected = True
        else:
            selected = None
            resolution_status = "needs_review"

        answers.append(
            NormalizedAnswer(
                answer_key=key,
                selected=selected,
                resolution_status=resolution_status,
                decision_source="deterministic",
                deterministic_score=mark_score.score,
                evidence_path=str(mark_score.evidence_path) if mark_score.evidence_path else None,
            )
        )

        if resolution_status == "needs_review":
            review_tasks.append(
                ReviewTask(
                    task_id=str(uuid.uuid4()),
                    answer_key=key,
                    reason=f"Score {mark_score.score:.3f} is ambiguous (thresholds: {config.low_threshold}-{config.high_threshold})",
                    status="pending",
                )
            )

    return NormalizedPageResult(
        form_id=profile.form_id,
        form_version=profile.form_version,
        page_number=page_number,
        answers=tuple(answers),
        review_tasks=tuple(review_tasks),
    )

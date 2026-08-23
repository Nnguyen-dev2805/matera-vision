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

    score_map: dict[str, list[MarkScore]] = {}
    for ms in mark_scores:
        score_map.setdefault(ms.question_id, []).append(ms)

    for q_def in profile.questions:
        q_scores = score_map.get(q_def.question_id, [])
        
        original_selections = {}
        final_selections = {}
        
        for ms in q_scores:
            if ms.score < config.low_threshold:
                original_selections[ms.option_id] = False
            elif ms.score >= config.high_threshold:
                original_selections[ms.option_id] = True
            else:
                original_selections[ms.option_id] = None
            final_selections[ms.option_id] = original_selections[ms.option_id]

        num_selected = sum(1 for v in original_selections.values() if v is True)

        if q_def.max_selections is not None and num_selected > q_def.max_selections:
            for opt_id, is_sel in original_selections.items():
                if is_sel is True:
                    final_selections[opt_id] = None
        
        elif num_selected < q_def.min_selections and q_scores:
            max_score = max(ms.score for ms in q_scores)
            for ms in q_scores:
                if ms.score == max_score:
                    final_selections[ms.option_id] = None

        for ms in q_scores:
            key = AnswerKey(
                form_id=profile.form_id,
                form_version=profile.form_version,
                page_number=page_number,
                question_id=ms.question_id,
                option_id=ms.option_id,
            )
            
            selected = final_selections[ms.option_id]
            resolution_status = "resolved" if selected is not None else "needs_review"
            
            answers.append(
                NormalizedAnswer(
                    answer_key=key,
                    selected=selected,
                    resolution_status=resolution_status,
                    decision_source="deterministic",
                    deterministic_score=ms.score,
                    confidence=ms.score,
                    evidence_path=str(ms.evidence_path) if ms.evidence_path else None,
                )
            )

            if resolution_status == "needs_review":
                orig_sel = original_selections[ms.option_id]
                if orig_sel is True:
                    reason = f"Over-selection: {num_selected} options selected, max is {q_def.max_selections}"
                elif orig_sel is False:
                    reason = f"Under-selection: {num_selected} options selected, min is {q_def.min_selections}. Highest score was {ms.score:.3f}"
                else:
                    if num_selected < q_def.min_selections and ms.score == max_score:
                        reason = f"Under-selection (and ambiguous): {num_selected} options selected, min is {q_def.min_selections}. Highest score was {ms.score:.3f}"
                    else:
                        reason = f"Score {ms.score:.3f} is ambiguous (thresholds: {config.low_threshold}-{config.high_threshold})"

                review_tasks.append(
                    ReviewTask(
                        task_id=str(uuid.uuid4()),
                        answer_key=key,
                        reason=reason,
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

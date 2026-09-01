from matera.core.contracts import (
    AnswerKey,
    NormalizedAnswer,
    NormalizedPageResult,
    ReviewTask,
)
from matera.core.profile import FormProfile
from matera.vision.contracts import MarkScore, RoutingConfig


def route_page(
    mark_scores: list[MarkScore],
    profile: FormProfile,
    page_number: int,
    config: RoutingConfig | None = None,
) -> NormalizedPageResult:
    """
    Evaluates mark scores against deterministic thresholds, ML classifier, and form constraints
    to produce a NormalizedPageResult.
    """
    config = config or RoutingConfig()

    # Fail-fast validation
    profile_options = set()
    question_map = {}
    for q in profile.questions:
        question_map[q.question_id] = q
        for opt in q.options:
            profile_options.add((q.question_id, opt.option_id))

    provided_options = set()
    score_by_key = {}
    for ms in mark_scores:
        key = (ms.question_id, ms.option_id)
        if key not in profile_options:
            raise ValueError(
                f"Unknown or extra MarkScore found for {ms.question_id}.{ms.option_id}"
            )
        if key in provided_options:
            raise ValueError(f"Duplicate MarkScore found for {ms.question_id}.{ms.option_id}")

        q_def = question_map[ms.question_id]
        if ms.strategy != q_def.mark_strategy:
            raise ValueError(
                f"Mark strategy mismatch for {ms.question_id}.{ms.option_id}: "
                f"expected {q_def.mark_strategy}, got {ms.strategy}"
            )

        provided_options.add(key)
        score_by_key[key] = ms

    missing_options = profile_options - provided_options
    if missing_options:
        missing_str = ", ".join(f"{q}.{o}" for q, o in sorted(missing_options))
        raise ValueError(f"Missing MarkScore for options: {missing_str}")

    answers: list[NormalizedAnswer] = []
    review_tasks: list[ReviewTask] = []

    for q_def in profile.questions:
        q_scores = [score_by_key[(q_def.question_id, opt.option_id)] for opt in q_def.options]

        original_selections = {}
        final_selections = {}

        for ms in q_scores:
            if ms.score >= config.high_threshold:
                original_selections[ms.option_id] = True
            elif ms.score < config.low_threshold:
                original_selections[ms.option_id] = False
            else:
                original_selections[ms.option_id] = None  # Needs Review

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

        # Build answers in the exact order of FormProfile options
        for opt in q_def.options:
            ms = score_by_key[(q_def.question_id, opt.option_id)]

            key = AnswerKey(
                form_id=profile.form_id,
                form_version=profile.form_version,
                page_number=page_number,
                question_id=ms.question_id,
                option_id=ms.option_id,
            )

            selected = final_selections[ms.option_id]
            resolution_status = "resolved" if selected is not None else "needs_review"

            if selected is True:
                confidence = float(ms.score)
            elif selected is False:
                confidence = float(1.0 - ms.score)
            else:
                confidence = None

            answers.append(
                NormalizedAnswer(
                    answer_key=key,
                    selected=selected,
                    resolution_status=resolution_status,
                    decision_source="deterministic",
                    deterministic_score=ms.score,
                    confidence=confidence,
                    evidence_path=str(ms.evidence_path) if ms.evidence_path else None,
                )
            )

            if resolution_status == "needs_review":
                orig_sel = original_selections[ms.option_id]
                if orig_sel is True:
                    reason = (
                        f"Over-selection: {num_selected} options selected, "
                        f"max is {q_def.max_selections}"
                    )
                elif orig_sel is False:
                    reason = (
                        f"Under-selection: {num_selected} options selected, "
                        f"min is {q_def.min_selections}. Highest score was {ms.score:.3f}"
                    )
                else:
                    if num_selected < q_def.min_selections and ms.score == max_score:
                        reason = (
                            f"Under-selection (and ambiguous): {num_selected} options selected, "
                            f"min is {q_def.min_selections}. Highest score was {ms.score:.3f}"
                        )
                    else:
                        reason = (
                            f"Score {ms.score:.3f} is ambiguous "
                            f"(thresholds: {config.low_threshold}-{config.high_threshold})"
                        )

                task_id = (
                    f"review:{key.form_id}:{key.form_version}:{key.page_number}:"
                    f"{key.question_id}:{key.option_id}"
                )

                review_tasks.append(
                    ReviewTask(
                        task_id=task_id,
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

from dataclasses import dataclass, field
from typing import Any

from matera.core.contracts import NormalizedAnswer, NormalizedPageResult
from matera.evaluation.ground_truth import ExpectedOption
from matera.vision.evidence import OptionMarkEvidence, PageMarkEvidence


@dataclass
class OptionEvaluationResult:
    question_id: str
    option_id: str
    expected_state: str  # MARKED, BLANK, UNKNOWN
    actual_state: str  # MARKED, BLANK, NEED_REVIEW, MISSING, ERROR
    score: float | None
    is_correct: bool
    is_review: bool
    is_skipped: bool
    error_type: str | None  # FP, FN, MISSING, ERROR
    failure_taxonomy: str
    debug_report_path: str | None
    evidence_path: str | None
    method: str | None
    q14_diagnostic_path: str | None = None
    notes: str = ""
    vlm_state: str | None = None
    vlm_raw_state: str | None = None
    vlm_reason: str | None = None
    vlm_parse_error: str | None = None
    vlm_provider_error: str | None = None
    vlm_crop_path: str | None = None
    vlm_latency_ms: float | None = None
    local_realignment: Any | None = None


@dataclass
class QuestionEvaluationResult:
    question_id: str
    expected_marked: list[str]
    actual_marked: list[str]
    unknown: bool
    has_review: bool
    has_error: bool
    exact_match_auto: bool
    option_error_count: int
    option_review_count: int
    debug_report_path: str | None
    notes: str = ""
    option_results: list[OptionEvaluationResult] = field(default_factory=list)


@dataclass
class PageEvaluationResult:
    item_id: str
    pdf: str
    page: int
    exact_match_auto: bool
    review_required: bool
    question_results: list[QuestionEvaluationResult] = field(default_factory=list)
    q14_diagnostics: list[dict] = field(default_factory=list)
    global_shape_diagnostics: list[dict] = field(default_factory=list)


def assign_taxonomy(
    expected_state: str,
    actual_state: str,
    method: str | None,
    opt_evidence: OptionMarkEvidence | None,
) -> str:
    if expected_state == "UNKNOWN":
        return "ground_truth_uncertain"
    if actual_state == "NEED_REVIEW":
        return "routing_review_required"
    if actual_state == "MISSING":
        return "missing_answer"
    if actual_state == "ERROR":
        return "pipeline_error"

    if expected_state == actual_state:
        return ""  # Correct

    # It's an error (FP or FN)
    is_fn = expected_state == "MARKED" and actual_state == "BLANK"
    _ = expected_state == "BLANK" and actual_state == "MARKED"

    # Use option evidence if available, otherwise fallback to method string
    used_method = method or ""
    if opt_evidence:
        used_method = opt_evidence.legacy_method or used_method

    if "GLOBAL_TOPOLOGY" in used_method or "GLOBAL_HULL" in used_method:
        return "global_under_select" if is_fn else "global_over_select"
    elif "LOCAL_RADIAL" in used_method:
        return "local_radial_blank_nham" if is_fn else "local_radial_marked_nham"
    elif "HSV" in used_method or "FALLBACK" in used_method:
        return "hsv_fallback_sai"

    return "unclassified"


def evaluate_page(
    item_id: str,
    pdf: str,
    page: int,
    expected_options: list[ExpectedOption],
    normalized_result: NormalizedPageResult | None,
    page_evidence: PageMarkEvidence | None,
    error_msg: str | None = None,
) -> PageEvaluationResult:

    if error_msg:
        # The entire page crashed
        q_results = []
        # Group expected by question
        q_map = {}
        for exp in expected_options:
            q_map.setdefault(exp.question_id, []).append(exp)

        for q_id, opts in q_map.items():
            o_results = []
            for exp in opts:
                o_results.append(
                    OptionEvaluationResult(
                        question_id=q_id,
                        option_id=exp.option_id,
                        expected_state=exp.expected_state,
                        actual_state="ERROR",
                        score=None,
                        is_correct=False,
                        is_review=False,
                        is_skipped=(exp.expected_state == "UNKNOWN"),
                        error_type="ERROR",
                        failure_taxonomy="pipeline_error",
                        debug_report_path=None,
                        evidence_path=None,
                        method=None,
                        notes=error_msg,
                    )
                )
            q_results.append(
                QuestionEvaluationResult(
                    question_id=q_id,
                    expected_marked=[o.option_id for o in opts if o.expected_state == "MARKED"],
                    actual_marked=[],
                    unknown=all(o.expected_state == "UNKNOWN" for o in opts),
                    has_review=False,
                    has_error=True,
                    exact_match_auto=False,
                    option_error_count=len(opts),
                    option_review_count=0,
                    debug_report_path=None,
                    notes=error_msg,
                    option_results=o_results,
                )
            )
        return PageEvaluationResult(
            item_id=item_id,
            pdf=pdf,
            page=page,
            exact_match_auto=False,
            review_required=False,
            question_results=q_results,
        )

    # Build maps for fast lookup
    actual_map: dict[tuple[str, str], NormalizedAnswer] = {}
    if normalized_result:
        for ans in normalized_result.answers:
            actual_map[(ans.answer_key.question_id, ans.answer_key.option_id)] = ans

    evidence_map: dict[tuple[str, str], OptionMarkEvidence] = {}
    if page_evidence:
        for q_ev in page_evidence.questions:
            for opt_ev in q_ev.option_evidence:
                evidence_map[(q_ev.question_id, opt_ev.option_id)] = opt_ev

    q_expected_map = {}
    for exp in expected_options:
        q_expected_map.setdefault(exp.question_id, []).append(exp)

    question_results = []
    page_exact_match = True
    page_has_review = False

    for q_id, opts in q_expected_map.items():
        o_results = []
        q_expected_marked = []
        q_actual_marked = []
        q_unknown = True
        q_has_review = False
        q_has_error = False

        for exp in opts:
            if exp.expected_state != "UNKNOWN":
                q_unknown = False
            if exp.expected_state == "MARKED":
                q_expected_marked.append(exp.option_id)

            ans = actual_map.get((q_id, exp.option_id))
            ev = evidence_map.get((q_id, exp.option_id))

            method = ev.legacy_method if ev else None
            score = ans.deterministic_score if ans else None

            if not ans:
                actual_state = "MISSING"
            elif ans.resolution_status == "needs_review":
                actual_state = "NEED_REVIEW"
                q_has_review = True
                page_has_review = True
            elif ans.selected:
                actual_state = "MARKED"
                q_actual_marked.append(exp.option_id)
            else:
                actual_state = "BLANK"

            is_skipped = exp.expected_state == "UNKNOWN"
            is_review = actual_state == "NEED_REVIEW"

            is_correct = False
            error_type = None
            if not is_skipped and not is_review:
                if exp.expected_state == actual_state:
                    is_correct = True
                else:
                    q_has_error = True
                    error_type = (
                        "FP"
                        if actual_state == "MARKED"
                        else "FN"
                        if actual_state == "BLANK"
                        else actual_state
                    )

            taxonomy = assign_taxonomy(exp.expected_state, actual_state, method, ev)

            vlm_state = ev.vlm.decision if ev and ev.vlm else None
            vlm_raw_state = ev.vlm.raw_state if ev and ev.vlm else None
            vlm_reason = ev.vlm.reason if ev and ev.vlm else None
            vlm_parse_error = ev.vlm.parse_error if ev and ev.vlm else None
            vlm_provider_error = None
            vlm_crop_path = None
            vlm_latency_ms = None
            
            if page_evidence:
                q_ev = next((q for q in page_evidence.questions if q.question_id == q_id), None)
                if q_ev and q_ev.vlm:
                    vlm_provider_error = q_ev.vlm.provider_error
                    vlm_crop_path = q_ev.vlm.crop_path
                    vlm_latency_ms = q_ev.vlm.latency_ms

            lr = getattr(ev, "local_realignment", None) if ev else None

            # In this context we don't know the full debug path yet, that's added by runner
            o_results.append(
                OptionEvaluationResult(
                    question_id=q_id,
                    option_id=exp.option_id,
                    expected_state=exp.expected_state,
                    actual_state=actual_state,
                    score=score,
                    is_correct=is_correct,
                    is_review=is_review,
                    is_skipped=is_skipped,
                    error_type=error_type,
                    failure_taxonomy=taxonomy,
                    debug_report_path=None,
                    evidence_path=None,  # TBD by runner if needed
                    method=method,
                    vlm_state=vlm_state,
                    vlm_raw_state=vlm_raw_state,
                    vlm_reason=vlm_reason,
                    vlm_parse_error=vlm_parse_error,
                    vlm_provider_error=vlm_provider_error,
                    vlm_crop_path=vlm_crop_path,
                    vlm_latency_ms=vlm_latency_ms,
                    local_realignment=lr,
                )
            )

        q_exact_match = not q_unknown and not q_has_error and not q_has_review
        if not q_exact_match and not q_unknown:
            page_exact_match = False

        question_results.append(
            QuestionEvaluationResult(
                question_id=q_id,
                expected_marked=q_expected_marked,
                actual_marked=q_actual_marked,
                unknown=q_unknown,
                has_review=q_has_review,
                has_error=q_has_error,
                exact_match_auto=q_exact_match,
                option_error_count=sum(1 for o in o_results if o.error_type),
                option_review_count=sum(1 for o in o_results if o.is_review),
                debug_report_path=None,
                option_results=o_results,
            )
        )

    return PageEvaluationResult(
        item_id=item_id,
        pdf=pdf,
        page=page,
        exact_match_auto=page_exact_match,
        review_required=page_has_review,
        question_results=question_results,
    )

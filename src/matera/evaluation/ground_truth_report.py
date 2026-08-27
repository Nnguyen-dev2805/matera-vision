import csv
import json
from collections import defaultdict
from pathlib import Path

from matera.evaluation.ground_truth_metrics import PageEvaluationResult


class GroundTruthReporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_report(self, run_id: str, run_config: dict, page_results: list[PageEvaluationResult]):
        self._write_run_config(run_config)
        self._write_details(run_id, page_results)
        self._write_questions(run_id, page_results)
        self._write_summary(page_results, run_config)
        self._write_mistakes(page_results)

    def _write_run_config(self, run_config: dict):
        path = self.output_dir / "run_config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

    def _write_details(self, run_id: str, page_results: list[PageEvaluationResult]):
        path = self.output_dir / "details.csv"
        headers = [
            "run_id",
            "item_id",
            "pdf",
            "page",
            "question_id",
            "option_id",
            "expected_state",
            "actual_state",
            "is_correct",
            "is_review",
            "is_skipped",
            "error_type",
            "failure_taxonomy",
            "method",
            "score",
            "debug_report_path",
            "evidence_path",
            "q14_diagnostic_path",
            "notes",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for page_res in page_results:
                for q_res in page_res.question_results:
                    for o_res in q_res.option_results:
                        writer.writerow(
                            {
                                "run_id": run_id,
                                "item_id": page_res.item_id,
                                "pdf": page_res.pdf,
                                "page": page_res.page,
                                "question_id": o_res.question_id,
                                "option_id": o_res.option_id,
                                "expected_state": o_res.expected_state,
                                "actual_state": o_res.actual_state,
                                "is_correct": o_res.is_correct,
                                "is_review": o_res.is_review,
                                "is_skipped": o_res.is_skipped,
                                "error_type": o_res.error_type or "",
                                "failure_taxonomy": o_res.failure_taxonomy,
                                "method": o_res.method or "",
                                "score": o_res.score if o_res.score is not None else "",
                                "debug_report_path": o_res.debug_report_path or "",
                                "evidence_path": o_res.evidence_path or "",
                                "q14_diagnostic_path": o_res.q14_diagnostic_path or "",
                                "notes": o_res.notes or "",
                            }
                        )

    def _write_questions(self, run_id: str, page_results: list[PageEvaluationResult]):
        path = self.output_dir / "questions.csv"
        headers = [
            "run_id",
            "item_id",
            "pdf",
            "page",
            "question_id",
            "expected_marked",
            "actual_marked",
            "unknown",
            "has_review",
            "has_error",
            "exact_match_auto",
            "option_error_count",
            "option_review_count",
            "debug_report_path",
            "notes",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for page_res in page_results:
                for q_res in page_res.question_results:
                    writer.writerow(
                        {
                            "run_id": run_id,
                            "item_id": page_res.item_id,
                            "pdf": page_res.pdf,
                            "page": page_res.page,
                            "question_id": q_res.question_id,
                            "expected_marked": json.dumps(q_res.expected_marked),
                            "actual_marked": json.dumps(q_res.actual_marked),
                            "unknown": q_res.unknown,
                            "has_review": q_res.has_review,
                            "has_error": q_res.has_error,
                            "exact_match_auto": q_res.exact_match_auto,
                            "option_error_count": q_res.option_error_count,
                            "option_review_count": q_res.option_review_count,
                            "debug_report_path": q_res.debug_report_path or "",
                            "notes": q_res.notes or "",
                        }
                    )

    def _write_summary(self, page_results: list[PageEvaluationResult], run_config: dict):
        path = self.output_dir / "summary.json"

        total_pages = len(page_results)
        total_questions = 0
        total_options = 0

        tp = tn = fp = fn = nr = skip = err = 0
        taxonomy_counts = defaultdict(int)

        page_exact = 0
        q_exact = 0

        for p in page_results:
            if p.exact_match_auto:
                page_exact += 1
            for q in p.question_results:
                total_questions += 1
                if q.exact_match_auto:
                    q_exact += 1
                for o in q.option_results:
                    total_options += 1
                    if o.is_skipped:
                        skip += 1
                    elif o.actual_state == "ERROR" or o.actual_state == "MISSING":
                        err += 1
                    elif o.is_review:
                        nr += 1
                    else:
                        if o.error_type == "FP":
                            fp += 1
                        elif o.error_type == "FN":
                            fn += 1
                        elif o.actual_state == "MARKED":
                            tp += 1
                        elif o.actual_state == "BLANK":
                            tn += 1

                    if o.failure_taxonomy:
                        taxonomy_counts[o.failure_taxonomy] += 1

        auto_decision_count = tp + tn + fp + fn
        known_count = auto_decision_count + nr

        q14_tp = q14_tn = q14_fp = q14_fn = q14_nr = 0
        q14_safe_mask_pixels = []
        for p in page_results:
            for diag in p.q14_diagnostics:
                expected = diag.get("expected_state")
                actual = diag.get("actual_state")
                if expected == "MARKED" and actual == "MARKED":
                    q14_tp += 1
                elif expected == "BLANK" and actual == "BLANK":
                    q14_tn += 1
                elif expected == "BLANK" and actual == "MARKED":
                    q14_fp += 1
                elif expected == "MARKED" and actual == "BLANK":
                    q14_fn += 1
                elif actual == "NEED_REVIEW":
                    q14_nr += 1
                if "safe_mask_pixels" in diag:
                    q14_safe_mask_pixels.append(diag["safe_mask_pixels"])

        summary = {
            "run_config": run_config,
            "metrics": {
                "pages_total": total_pages,
                "pages_exact_match": page_exact,
                "questions_total": total_questions,
                "questions_exact_match": q_exact,
                "options_total": total_options,
                "options_skipped": skip,
                "options_error": err,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "nr": nr,
                "auto_accuracy": (tp + tn) / auto_decision_count if auto_decision_count else None,
                "review_rate": nr / known_count if known_count else None,
                "coverage": auto_decision_count / known_count if known_count else None,
                "precision_marked": tp / (tp + fp) if (tp + fp) else None,
                "recall_marked": tp / (tp + fn) if (tp + fn) else None,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else None,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) else None,
            },
            "taxonomy": dict(taxonomy_counts),
            "q14": {
                "tp": q14_tp,
                "tn": q14_tn,
                "fp": q14_fp,
                "fn": q14_fn,
                "nr": q14_nr,
                "safe_mask_pixel_distribution": {
                    "min": min(q14_safe_mask_pixels) if q14_safe_mask_pixels else None,
                    "max": max(q14_safe_mask_pixels) if q14_safe_mask_pixels else None,
                    "mean": sum(q14_safe_mask_pixels) / len(q14_safe_mask_pixels)
                    if q14_safe_mask_pixels
                    else None,
                },
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _write_mistakes(self, page_results: list[PageEvaluationResult]):
        path = self.output_dir / "mistakes.md"

        fps = []
        fns = []
        nrs = []
        errs = []

        for p in page_results:
            for q in p.question_results:
                for o in q.option_results:
                    row = f"| {p.pdf} | {p.page} | {o.question_id} | {o.option_id} | {o.expected_state} | {o.actual_state} | {o.method or ''} | {o.score or ''} | {o.failure_taxonomy} | [Link]({o.debug_report_path or ''}) |"
                    if o.error_type == "FP":
                        fps.append(row)
                    elif o.error_type == "FN":
                        fns.append(row)
                    elif o.is_review:
                        nrs.append(row)
                    elif o.actual_state in ("ERROR", "MISSING"):
                        errs.append(row)

        header = "| PDF | Page | Question | Option | Expected | Actual | Method | Score | Taxonomy | Debug |\n|---|---|---|---|---|---|---|---|---|---|"

        lines = ["# Ground Truth Evaluation Mistakes\n"]

        lines.append("## False Positives")
        lines.append(header)
        lines.extend(fps)
        lines.append("")

        lines.append("## False Negatives")
        lines.append(header)
        lines.extend(fns)
        lines.append("")

        lines.append("## Needs Review")
        lines.append(header)
        lines.extend(nrs)
        lines.append("")

        lines.append("## Errors / Missing")
        lines.append(header)
        lines.extend(errs)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from matera.core.contracts import RoutingConfig
from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile
from matera.data.loader import load_page_image
from matera.evaluation.metrics import ConfusionMatrix
from matera.evaluation.report import EvaluationReport
from matera.vision.alignment import TemplateAligner
from matera.vision.mark import calculate_mark_scores
from matera.vision.routing import route_page


@dataclass
class ExpectedAnswer:
    question_id: str
    option_id: str
    expected_mark: int
    response_type: str


def load_golden_dataset(csv_path: Path) -> dict[str, list[ExpectedAnswer]]:
    """Loads labels.csv and groups expected answers by source_page_image."""
    dataset: dict[str, list[ExpectedAnswer]] = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["source_page_image"]
            ans = ExpectedAnswer(
                question_id=row["question_id"],
                option_id=row["option_id"],
                expected_mark=int(row["expected_mark"]),
                response_type=row["response_type"],
            )
            dataset[image_name].append(ans)

    return dataset


def run_evaluation(dataset_path: Path, output_path: Path) -> None:
    # 1. Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_golden_dataset(dataset_path)
    print(f"Loaded {len(dataset)} unique pages.")

    # We assume 'matera-pre' v1 profile for this baseline harness
    profile = load_semantic_profile(Path("profiles/matera-pre/v1/semantic.json"))
    layout_profile = load_layout_profile(Path("profiles/matera-pre/v1/layout.json"))
    print(f"Loaded semantic profile: {profile.form_id} {profile.form_version}")

    roi_map = {
        (mapping.question_id, mapping.option_id): mapping.roi for mapping in layout_profile.mappings
    }

    # Initialize pipeline
    aligner = TemplateAligner(reference_image_path=Path("data/pages/page_1.png"))
    routing_config = RoutingConfig(confidence_threshold=0.8, ambiguity_margin=0.2)

    report = EvaluationReport()

    # Create debug dir
    debug_dir = Path("data/debug/errors")
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 2. Pipeline loop
    for page_image_name, expected_answers in dataset.items():
        print(f"Processing {page_image_name}...")

        # Load image
        img_path = Path("data/golden/images") / page_image_name
        if not img_path.exists():
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue

        page_image = load_page_image(img_path)

        # Run Pipeline
        try:
            aligned_page = aligner.align(page_image)
            mark_scores = calculate_mark_scores(aligned_page, profile)
            normalized_result = route_page(mark_scores, profile, routing_config)
        except Exception as e:
            print(f"Pipeline error on {page_image_name}: {e}")
            continue

        # 3. Metric Comparison
        result_map = {
            (ans.answer_key.question_id, ans.answer_key.option_id): ans
            for ans in normalized_result.answers
        }

        page_cm = ConfusionMatrix()
        page_errors = 0
        page_reviews = 0

        draw_tasks = []

        for expected in expected_answers:
            key = (expected.question_id, expected.option_id)
            ans = result_map.get(key)
            roi = roi_map.get(key)

            resp_type = expected.response_type
            if resp_type not in report.by_response_type:
                report.by_response_type[resp_type] = ConfusionMatrix()

            if ans is None or ans.resolution_status == "needs_review":
                page_cm.needs_review += 1
                report.overall_metrics.needs_review += 1
                report.by_response_type[resp_type].needs_review += 1
                page_reviews += 1
                if roi:
                    # Yellow for Needs Review
                    draw_tasks.append((roi, (0, 255, 255)))
                continue

            # Calculate TP/TN/FP/FN
            selected = 1 if ans.selected else 0

            if expected.expected_mark == 1 and selected == 1:
                page_cm.tp += 1
                report.overall_metrics.tp += 1
                report.by_response_type[resp_type].tp += 1
            elif expected.expected_mark == 0 and selected == 0:
                page_cm.tn += 1
                report.overall_metrics.tn += 1
                report.by_response_type[resp_type].tn += 1
            elif expected.expected_mark == 0 and selected == 1:
                page_cm.fp += 1
                report.overall_metrics.fp += 1
                report.by_response_type[resp_type].fp += 1
                page_errors += 1
                if roi:
                    # Red for False Positive
                    draw_tasks.append((roi, (0, 0, 255)))
            elif expected.expected_mark == 1 and selected == 0:
                page_cm.fn += 1
                report.overall_metrics.fn += 1
                report.by_response_type[resp_type].fn += 1
                page_errors += 1
                if roi:
                    # Red for False Negative
                    draw_tasks.append((roi, (0, 0, 255)))

        # Check for exact match
        report.total_pages += 1
        if page_errors == 0 and page_reviews == 0:
            report.exact_match_pages += 1
        else:
            # Generate debug image
            if draw_tasks:
                import cv2

                debug_img = aligned_page.image.copy()
                for roi, color in draw_tasks:
                    cv2.rectangle(
                        debug_img, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), color, 2
                    )
                out_name = debug_dir / f"error_{page_image_name}"
                cv2.imwrite(str(out_name), debug_img)

    # Save JSON report
    report_dict = {
        "overall": {
            "tp": report.overall_metrics.tp,
            "tn": report.overall_metrics.tn,
            "fp": report.overall_metrics.fp,
            "fn": report.overall_metrics.fn,
            "needs_review": report.overall_metrics.needs_review,
            "f1_score": report.overall_metrics.f1_score,
            "fpr": report.overall_metrics.fpr,
            "fnr": report.overall_metrics.fnr,
            "coverage": report.overall_metrics.coverage,
            "risk": report.overall_metrics.risk,
            "review_rate": report.overall_metrics.review_rate,
        },
        "page_exact_match_rate": report.page_exact_match_rate,
        "by_response_type": {},
    }

    for rt, cm in report.by_response_type.items():
        report_dict["by_response_type"][rt] = {
            "tp": cm.tp,
            "tn": cm.tn,
            "fp": cm.fp,
            "fn": cm.fn,
            "needs_review": cm.needs_review,
            "f1_score": cm.f1_score,
            "fpr": cm.fpr,
            "fnr": cm.fnr,
            "coverage": cm.coverage,
            "risk": cm.risk,
            "review_rate": cm.review_rate,
        }

    # Pretty print some stuff
    print("\n--- Evaluation Results ---")
    print(f"Total Pages: {report.total_pages}")
    print(f"Page Exact Match (STP): {report.page_exact_match_rate:.2%}")
    print(f"Global F1 Score: {report.overall_metrics.f1_score:.2%}")
    print(f"Global Coverage: {report.overall_metrics.coverage:.2%}")
    print(f"Global Risk: {report.overall_metrics.risk:.2%}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nReport saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate End-to-End Baseline Pipeline")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to labels.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/baseline_report.json"),
        help="Output JSON report",
    )

    args = parser.parse_args()
    run_evaluation(args.dataset, args.output)


if __name__ == "__main__":
    main()

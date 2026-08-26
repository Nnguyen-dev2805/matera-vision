import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile
from matera.data.contracts import RenderedPage
from matera.evaluation.metrics import ConfusionMatrix
from matera.evaluation.report import EvaluationReport
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig, RoutingConfig
from matera.vision.mark import extract_mark_scores
from matera.vision.reference import generate_median_reference
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


def run_evaluation(dataset_path: Path, output_path: Path, reference_type: str) -> None:
    # 1. Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_golden_dataset(dataset_path)
    print(f"Loaded {len(dataset)} unique pages.")

    profile = load_semantic_profile(Path("profiles/semantic.json"))
    layout_profile = load_layout_profile(Path("profiles/layout.json"))
    print(f"Loaded semantic profile: {profile.form_id} {profile.form_version}")

    if reference_type == "hybrid":
        ref_path = Path("scratch/synthetic_median_reference.png")
    else:
        # Fallback to the first page in the dataset if the designated reference doesn't exist
        ref_path = Path("data/pages/page_1.png")
        if not ref_path.exists() and dataset:
            first_page = list(dataset.keys())[0]
            ref_path = Path("data/pages") / first_page
            
    if not ref_path.exists():
        if not dataset:
            raise RuntimeError(f"Cannot generate reference: no dataset found at {dataset_path} and {ref_path} is missing.")
            
        first_page = list(dataset.keys())[0]
        anchor_path = Path("data/pages") / first_page
        if not anchor_path.exists():
            raise RuntimeError(f"Cannot generate reference: anchor image {anchor_path} does not exist.")
            
        anchor_img = Image.open(anchor_path).convert("RGB")
        aligned_pages = []
        
        if reference_type == "hybrid":
            print(f"Hybrid reference {ref_path} not found. Generating on the fly...")
            
            # Align up to 10 pages against the anchor
            align_cfg = AlignmentConfig(algorithm="orb", transform_model="affine", inlier_threshold=0.05)
            for page_name in list(dataset.keys())[:10]:
                img = Image.open(Path("data/pages") / page_name).convert("RGB")
                src_page = RenderedPage(image=img, page_number=1, width_px=img.width, height_px=img.height, pdf_width_pt=float(img.width), pdf_height_pt=float(img.height))
                try:
                    result = align_page(src_page, anchor_img, align_cfg)
                    if result:
                        aligned_pages.append(result)
                except Exception as e:
                    print(f"Skipping page {page_name} for median reference generation due to alignment error: {e}")
                    continue
                
        if aligned_pages:
            reference_img = generate_median_reference(aligned_pages)
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            reference_img.save(ref_path)
            print(f"Saved generated reference to {ref_path}")
        else:
            print(f"Baseline reference missing, using anchor image: {anchor_path}")
            ref_path = anchor_path
            reference_img = Image.open(ref_path).convert("RGB")
    else:
        print(f"Using reference image: {ref_path}")
        reference_img = Image.open(ref_path).convert("RGB")

    # Initialize pipeline configs
    alignment_config = AlignmentConfig(
        algorithm="orb",
        transform_model="affine",
        inlier_threshold=0.05
    )
    routing_config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)

    report = EvaluationReport()

    # Create debug dir
    debug_dir = Path("data/debug/errors")
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 2. Pipeline loop
    for page_image_name, expected_answers in dataset.items():
        print(f"Processing {page_image_name}...")

        # Load image
        img_path = Path("data/pages") / page_image_name
        if not img_path.exists():
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue

        raw_img = Image.open(img_path).convert("RGB")
        
        # Try to parse page number from filename (e.g. "page_5.png")
        match = re.search(r"page_(\d+)", page_image_name)
        page_num = int(match.group(1)) if match else 0

        rendered = RenderedPage(
            page_number=page_num,
            width_px=raw_img.width,
            height_px=raw_img.height,
            pdf_width_pt=1.0,
            pdf_height_pt=1.0,
            image=raw_img
        )

        try:
            # matera-pre is a 1-page form, so all 10 PDFs are just page 1 filled out 10 times.
            layout = layout_profile.pages[0]
        except IndexError:
            print("Warning: No layout found in profile. Skipping.")
            continue

        # Count the page regardless of whether it crashes
        report.total_pages += 1

        # Run Pipeline
        try:
            aligned_page = align_page(rendered, reference_img, alignment_config)
            mark_scores = extract_mark_scores(
                aligned_page, profile, layout, reference_img, debug_dir=str(Path("data/debug/evidence"))
            )
            normalized_result = route_page(mark_scores, profile, page_num, routing_config)
        except Exception as e:
            import traceback
            traceback.print_exc()
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
            ans = result_map.get((expected.question_id, expected.option_id))
            roi = next((r for r in layout.rois if r.question_id == expected.question_id and r.option_id == expected.option_id), None)

            resp_type = expected.response_type
            if resp_type not in report.by_response_type:
                report.by_response_type[resp_type] = ConfusionMatrix()

            if ans is None:
                # System Error / Data drop - not a review!
                is_review = False
                selected = 0
            else:
                is_review = ans.resolution_status == "needs_review"
                selected = 1 if ans.selected else 0

            if is_review:
                page_cm.needs_review += 1
                report.overall_metrics.needs_review += 1
                report.by_response_type[resp_type].needs_review += 1
                page_reviews += 1
                if roi:
                    # Yellow for Needs Review
                    draw_tasks.append((roi, (0, 255, 255)))
                continue

            # Calculate TP/TN/FP/FN

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
        if page_errors == 0 and page_reviews == 0:
            report.exact_match_pages += 1
        else:
            # Generate debug image
            if draw_tasks:
                import cv2
                import numpy as np
                debug_img = np.array(aligned_page.image.convert("RGB"))
                for roi_def, color in draw_tasks:
                    bbox = roi_def.bbox
                    cv2.rectangle(debug_img, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), color, 2)
                
                # Save
                cv2.imwrite(str(Path("data/debug/errors") / page_image_name), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nReport saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate End-to-End Pipeline")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to labels.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/report.json"),
        help="Output JSON report",
    )
    parser.add_argument(
        "--reference-type",
        type=str,
        choices=["baseline", "hybrid"],
        default="hybrid",
        help="Which reference image to use (baseline or hybrid)",
    )

    args = parser.parse_args()
    run_evaluation(args.dataset, args.output, args.reference_type)


if __name__ == "__main__":
    main()

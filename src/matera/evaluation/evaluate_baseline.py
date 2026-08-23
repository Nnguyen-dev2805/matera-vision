import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from matera.core.profile import load_semantic_profile


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
    print(f"Loaded profile: {profile.form_id} {profile.form_version}")

    # 2. Pipeline loop
    # For Task 10.2, just print out the loop. Real logic in 10.3.
    for page_image, expected_answers in dataset.items():
        print(f"Processing {page_image} with {len(expected_answers)} expected answers...")
        # Stub for alignment, scoring, routing...

    print("Done.")


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

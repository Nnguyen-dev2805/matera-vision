import argparse
import csv
import sys
from pathlib import Path

from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile
from matera.vision.contracts import RoutingConfig
from matera.classifier.dataset import build_training_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare ambiguous ROIs training dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to labels.csv")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV")
    
    args = parser.parse_args()
    
    # 1. Load Profile
    profile = load_semantic_profile(Path("profiles/matera-pre/v1/semantic.json"))
    layout_profile = load_layout_profile(Path("profiles/matera-pre/v1/layout.json"))
    
    # In matera-pre we only have 1 page layout, so we can mock the FormProfile or just update build_training_dataset to take SemanticProfile and Layout
    print(f"Loaded semantic profile: {profile.form_id} v{profile.form_version}")
    
    # 2. Configure routing
    routing_config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)
    
    # 3. Extract dataset
    print(f"Extracting ambiguous ROIs from {args.input}...")
    dataset = build_training_dataset(args.input, profile, layout_profile.pages[0], routing_config)
    
    print(f"Extracted {len(dataset)} ambiguous ROIs.")
    
    if not dataset:
        print("No ambiguous ROIs found. Exiting.")
        sys.exit(0)
        
    # 4. Save to CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "page_number",
        "question_id",
        "option_id",
        "dark_pixel_ratio",
        "foreground_area_ratio",
        "contour_count",
        "largest_component_ratio",
        "bbox_fill_ratio",
        "deterministic_score",
        "expected_mark"
    ]
    
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
        
    print(f"Saved training dataset to {out_path}")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile
from matera.data.golden import AnnotationError, generate_golden_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Matera Vision golden dataset.")
    parser.add_argument(
        "--pages-dir",
        type=Path,
        required=True,
        help="Directory containing source PNG pages",
    )
    parser.add_argument(
        "--semantic",
        type=Path,
        required=True,
        help="Path to semantic.json profile",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        required=True,
        help="Path to layout.json profile",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to ground_truth.json annotations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for golden dataset",
    )

    args = parser.parse_args()

    try:
        semantic_profile = load_semantic_profile(args.semantic)
        layout_profile = load_layout_profile(args.layout, semantic=semantic_profile)

        import json

        with open(args.ground_truth, encoding="utf-8") as f:
            ground_truth_json = json.load(f)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading configuration files: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading JSON or configuration files: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        generate_golden_dataset(
            pages_dir=args.pages_dir,
            semantic_profile=semantic_profile,
            layout_profile=layout_profile,
            ground_truth_json=ground_truth_json,
            output_dir=args.output_dir,
        )
        print(f"Golden dataset successfully generated at {args.output_dir}")
    except AnnotationError as e:
        print(f"Annotation validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

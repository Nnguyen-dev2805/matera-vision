import argparse
import os
import sys
from pathlib import Path

from PIL import Image

from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile
from matera.core.q14_vlm_profile import load_q14_vlm_profile
from matera.data.extract import extract_pages
from matera.export.excel import export_to_excel
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig, RoutingConfig
from matera.vision.mark import extract_mark_scores
from matera.vision.q14_vlm_resolver import Q14VlmRuntime
from matera.vision.question_vlm_resolver import QuestionVlmRuntime
from matera.vision.routing import route_page
from matera.vlm.gemini_client import GeminiVlmClient


def process_pdf(pdf_path: Path, output_path: Path, profile_dir: Path) -> None:
    print(f"Starting Matera Vision Pipeline for: {pdf_path}")

    # 1. Load Profiles
    semantic_path = profile_dir / "semantic.json"
    layout_path = profile_dir / "layout.json"

    if not semantic_path.exists() or not layout_path.exists():
        print(f"Error: Profiles not found in {profile_dir}", file=sys.stderr)
        return

    semantic_profile = load_semantic_profile(semantic_path)
    layout_profile = load_layout_profile(layout_path)
    print(f"Loaded Form Profile: {semantic_profile.form_id} v{semantic_profile.form_version}")

    # 2. Extract PDF Pages
    pages = list(extract_pages(pdf_path))
    print(f"Extracted {len(pages)} pages from PDF.")

    # 3. Setup Configurations
    ref_img_path = profile_dir / "reference_template.png"
    if not ref_img_path.exists():
        raise RuntimeError(
            f"Reference image {ref_img_path} not found in profile directory."
        )
    reference_img = Image.open(ref_img_path).convert("RGB")
    alignment_config = AlignmentConfig(
        algorithm="orb", transform_model="affine", inlier_threshold=0.05
    )
    routing_config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)

    q14_profile_path = profile_dir / "q14_vlm.json"
    q14_runtime = None
    if q14_profile_path.exists():
        q14_profile = load_q14_vlm_profile(q14_profile_path)
        client = None
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            client = GeminiVlmClient(api_key=api_key)
        else:
            print(
                "Warning: GEMINI_API_KEY not set. Q14/escalated Q1-Q13 marked for review."
            )

        q14_runtime = Q14VlmRuntime(
            profile=q14_profile,
            client=client,
            model=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
            timeout_s=60.0,
        )
    else:
        print(
            f"Warning: Q14 profile not found at {q14_profile_path}.\nQ14 will be marked for review."
        )

    question_vlm_runtime = QuestionVlmRuntime(
        client=client
        if "client" in locals()
        else (
            GeminiVlmClient(api_key=os.environ["GEMINI_API_KEY"])
            if os.environ.get("GEMINI_API_KEY")
            else None
        ),
        model=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
        timeout_s=60.0,
    )

    # 4. Process loop
    results = []

    # matera-pre is a 1-page form, so we use layout.pages[0] for all pages.
    # If we support multi-page forms later, we need to match the page number to the layout.
    page_layout = layout_profile.pages[0]

    debug_dir = Path("data/debug/evidence")
    debug_dir.mkdir(parents=True, exist_ok=True)

    for page in pages:
        print(f"Processing page {page.page_number}...")
        try:
            # Align
            aligned_page = align_page(page, reference_img, alignment_config)

            # Detect
            mark_scores = extract_mark_scores(
                aligned_page,
                semantic_profile,
                page_layout,
                reference_img,
                debug_dir=str(debug_dir),
                q14_vlm_runtime=q14_runtime,
                question_vlm_runtime=question_vlm_runtime,
            )

            # Route
            result = route_page(mark_scores, semantic_profile, page.page_number, routing_config)

            results.append(result)
            print(
                f"  -> {len(result.answers)} answers extracted, {len(result.review_tasks)} needs review."  # noqa: E501
            )

        except Exception as e:
            print(f"  -> Error processing page {page.page_number}: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    # 5. Export
    if not results:
        print("No pages processed successfully.")
        return

    print(f"Exporting results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_excel(results, semantic_profile, str(output_path))
    print("Done!")


def main() -> int:
    parser = argparse.ArgumentParser(description="Matera Vision End-to-End Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Process Command
    process_parser = subparsers.add_parser("process", help="Process a PDF into Excel")
    process_parser.add_argument("--pdf", type=Path, required=True, help="Input PDF file")
    process_parser.add_argument("--output", type=Path, required=True, help="Output Excel file")
    process_parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("profiles"),
        help="Directory containing semantic.json and layout.json",
    )

    args = parser.parse_args()

    if args.command == "process":
        process_pdf(args.pdf, args.output, args.profile_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())

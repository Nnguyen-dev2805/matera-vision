import argparse
import sys
from pathlib import Path

from PIL import Image

from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile
from matera.data.extract import extract_pages
from matera.export.excel import export_to_excel
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig, RoutingConfig
from matera.vision.mark import extract_mark_scores
from matera.vision.routing import route_page


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
    ref_img_path = Path("scratch/synthetic_median_reference.png")
    if not ref_img_path.exists():
        raise RuntimeError(f"Reference image {ref_img_path} not found. Please run evaluate.py first to generate it.")
    reference_img = Image.open(ref_img_path).convert("RGB")
    alignment_config = AlignmentConfig(
        algorithm="orb",
        transform_model="affine",
        inlier_threshold=0.05
    )
    routing_config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)
    


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
                aligned_page, semantic_profile, page_layout, reference_img, debug_dir=str(debug_dir)
            )
            
            # Route
            result = route_page(
                mark_scores, semantic_profile, page.page_number, routing_config
            )
            
            results.append(result)
            print(f"  -> {len(result.answers)} answers extracted, {len(result.review_tasks)} needs review.")
            
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
        "--profile-dir", type=Path, default=Path("profiles"),
        help="Directory containing semantic.json and layout.json"
    )
    process_parser.add_argument(
        "--classifier", type=Path, default=Path("models/matera-pre-v1/classifier.joblib"),
        help="Path to the trained AmbiguityClassifier .joblib file"
    )
    
    args = parser.parse_args()
    
    if args.command == "process":
        process_pdf(args.pdf, args.output, args.profile_dir)
        
    return 0


if __name__ == "__main__":
    sys.exit(main())

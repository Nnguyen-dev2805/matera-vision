import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image

from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile
from matera.data.contracts import RenderedPage
from matera.data.io import save_image_lossless
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig, AlignmentError


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Matera Vision aligned dataset.")
    parser.add_argument(
        "--pages-dir", type=Path, required=True, help="Directory containing source PNG pages"
    )
    parser.add_argument(
        "--reference", type=Path, required=True, help="Path to reference_template.png"
    )
    parser.add_argument(
        "--semantic",
        type=Path,
        default=Path("profiles/semantic.json"),
        help="Path to semantic profile",
    )
    parser.add_argument(
        "--layout", type=Path, default=Path("profiles/layout.json"), help="Path to layout profile"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="Inlier threshold for alignment"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for aligned dataset"
    )

    args = parser.parse_args()

    # 1. Load profiles
    try:
        semantic_profile = load_semantic_profile(args.semantic)
        layout_profile = load_layout_profile(args.layout, semantic=semantic_profile)
    except Exception as e:
        print(f"Error loading configuration files: {e}", file=sys.stderr)
        sys.exit(1)

    if not layout_profile.pages:
        print("Error: No pages found in layout profile", file=sys.stderr)
        sys.exit(1)

    page_layout = layout_profile.pages[0]

    # 2. Load reference image
    try:
        reference_image = Image.open(args.reference)
        # Ensure RGB
        if reference_image.mode != "RGB":
            reference_image = reference_image.convert("RGB")
    except Exception as e:
        print(f"Error loading reference image: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Setup configuration
    config = AlignmentConfig(
        algorithm="orb",
        transform_model="affine",
        inlier_threshold=args.threshold,
        reference_dpi=layout_profile.reference_dpi,
        target_width=page_layout.width_px,
        target_height=page_layout.height_px,
    )

    num_expected_pages = 10
    source_paths = {}
    for page_num in range(1, num_expected_pages + 1):
        source_path = args.pages_dir / f"page_{page_num}.png"
        if not source_path.exists():
            print(f"Error: Source page image not found: {source_path}", file=sys.stderr)
            sys.exit(1)
        source_paths[page_num] = source_path

    # 4. Process each page safely
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="aligned_", dir=args.output_dir.parent))

    try:
        for page_num in range(1, num_expected_pages + 1):
            source_path = source_paths[page_num]

            with Image.open(source_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Mock RenderedPage since pdf metrics aren't used for alignment
                source_page = RenderedPage(
                    page_number=page_num,
                    width_px=img.width,
                    height_px=img.height,
                    pdf_width_pt=8.5 * 72,
                    pdf_height_pt=11.0 * 72,
                    image=img,
                )

                try:
                    aligned_page = align_page(
                        source_page=source_page,
                        reference_image=reference_image,
                        config=config,
                        profile_form_id=layout_profile.form_id,
                        profile_version=layout_profile.form_version,
                    )
                except AlignmentError as e:
                    print(f"Alignment failed for page {page_num}: {e}", file=sys.stderr)
                    sys.exit(1)

                # 4.1 Save clean aligned image
                clean_filename = f"aligned_page_{page_num:03d}.png"
                clean_path = tmp_dir / clean_filename
                save_image_lossless(aligned_page.image, clean_path)

                # 4.2 Save debug aligned image
                debug_filename = f"debug_aligned_page_{page_num:03d}.png"
                debug_path = tmp_dir / debug_filename

                # Create a copy for drawing debug info
                import cv2
                import numpy as np

                debug_img = np.array(aligned_page.image)
                # Draw layout rois for debug
                for p in layout_profile.pages:
                    for roi in p.rois:
                        bbox = roi.bbox
                        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
                        cv2.rectangle(
                            debug_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2
                        )

                from PIL import Image as PILImage

                save_image_lossless(PILImage.fromarray(debug_img), debug_path)

                print(f"Page {page_num}: score={aligned_page.alignment_score:.3f}")

        # 5. Atomic promotion
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        os.rename(tmp_dir, args.output_dir)
        print(f"Aligned dataset successfully generated at {args.output_dir}")

    except Exception as e:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        print(f"Error generating aligned dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

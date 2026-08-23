import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from matera.core.errors import ProfileValidationError
from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile


def draw_bounding_box(
    draw: ImageDraw.ImageDraw,
    bbox: object,
    label: str,
    outline_color: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> None:
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h  # type: ignore

    # Scale coordinates
    sx, sy = x * scale_x, y * scale_y
    sw, sh = w * scale_x, h * scale_y

    draw.rectangle([sx, sy, sx + sw, sy + sh], outline=outline_color, width=2)
    draw.text((sx + 2, sy - 12), label, fill=outline_color)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate debug overlays for a layout profile.")
    parser.add_argument("--semantic", required=True, type=Path, help="Path to semantic.json")
    parser.add_argument("--layout", required=True, type=Path, help="Path to layout.json")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to save the debug images"
    )
    parser.add_argument(
        "--page-number", type=int, default=None, help="Specific page number to debug (default: all)"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Optional directory containing background images (e.g. page_1.png)",
    )

    args = parser.parse_args()

    try:
        semantic = load_semantic_profile(args.semantic)
    except ProfileValidationError as e:
        print(f"Error loading semantic profile: {e}", file=sys.stderr)
        return 1

    try:
        layout = load_layout_profile(args.layout, semantic)
    except ProfileValidationError as e:
        print(f"Error loading layout profile: {e}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pages_to_draw = layout.pages
    if args.page_number is not None:
        pages_to_draw = [p for p in pages_to_draw if p.page_number == args.page_number]
        if not pages_to_draw:
            print(f"Error: Page {args.page_number} not found in layout profile.", file=sys.stderr)
            return 1

    for page in pages_to_draw:
        bg_images = []
        if args.images_dir:
            if len(pages_to_draw) == 1:
                # 1-page template: apply to all scan instances in the directory
                pngs = list(args.images_dir.glob("*.png"))
                jpgs = list(args.images_dir.glob("*.jpg"))
                bg_images = sorted(pngs + jpgs)
            else:
                # Multi-page template: find the specific page image
                bg_path = args.images_dir / f"page_{page.page_number}.png"
                if not bg_path.exists():
                    bg_path = args.images_dir / f"page_{page.page_number:03d}.png"
                if bg_path.exists():
                    bg_images = [bg_path]

            if not bg_images:
                print(
                    f"Error: Background image not found in {args.images_dir} "
                    f"for page {page.page_number}.",
                    file=sys.stderr,
                )
                return 1

        # If no images found, still generate a blank one
        if not bg_images:
            bg_images = [None]

        for img_idx, bg_path in enumerate(bg_images):
            scale_x = 1.0
            scale_y = 1.0
            img = None

            if bg_path:
                img = Image.open(bg_path)
                if img.width != page.width_px or img.height != page.height_px:
                    scale_x = img.width / page.width_px
                    scale_y = img.height / page.height_px
            else:
                # Create a blank image with original layout dimensions
                img = Image.new("RGB", (page.width_px, page.height_px), "white")

            draw = ImageDraw.Draw(img)

            # Draw page bounds (green)
            draw_bounding_box(
                draw,
                type("BBox", (), {"x": 0, "y": 0, "w": page.width_px, "h": page.height_px})(),
                f"page_{page.page_number:03d}",
                "green",
                scale_x,
                scale_y,
            )

            # Draw anchors (blue)
            for anchor in page.anchors:
                label = f"{anchor.anchor_type}:{anchor.anchor_id}"
                draw_bounding_box(draw, anchor.bbox, label, "blue", scale_x, scale_y)

            # Draw ROIs (red)
            for roi in page.rois:
                label = f"{roi.question_id}/{roi.option_id}"
                draw_bounding_box(draw, roi.bbox, label, "red", scale_x, scale_y)

            if bg_path:
                import re
                m = re.search(r'\d+', bg_path.stem)
                if m:
                    num = int(m.group())
                    output_name = f"overlay_page_{num:03d}.png"
                else:
                    output_name = f"overlay_{bg_path.stem}.png"
            else:
                output_name = f"overlay_page_{page.page_number:03d}.png"

            output_path = args.output_dir / output_name
            img.save(output_path)
            print(f"Generated {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

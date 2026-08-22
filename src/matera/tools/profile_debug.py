import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from matera.core.errors import ProfileValidationError
from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile


def draw_bounding_box(
    draw: ImageDraw.ImageDraw, bbox: object, label: str, outline_color: str, fill_color: str
) -> None:
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h  # type: ignore
    draw.rectangle([x, y, x + w, y + h], outline=outline_color, width=2, fill=fill_color)
    draw.text((x + 2, y - 12), label, fill=outline_color)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate debug overlays for a layout profile.")
    parser.add_argument("--semantic", required=True, type=Path, help="Path to semantic.json")
    parser.add_argument("--layout", required=True, type=Path, help="Path to layout.json")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to save the debug images"
    )
    parser.add_argument(
        "--page", type=int, default=None, help="Specific page number to debug (default: all)"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional background image (if --page is specified)",
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
    if args.page is not None:
        pages_to_draw = [p for p in pages_to_draw if p.page_number == args.page]
        if not pages_to_draw:
            print(f"Error: Page {args.page} not found in layout profile.", file=sys.stderr)
            return 1

    if args.image:
        if args.page is None:
            print("Error: --image requires --page to be specified", file=sys.stderr)
            return 1
        if not args.image.exists():
            print(f"Error: Background image {args.image} not found.", file=sys.stderr)
            return 1

    for page in pages_to_draw:
        if args.image and page.page_number == args.page:
            img = Image.open(args.image).convert("RGBA")
            if img.width != page.width_px or img.height != page.height_px:
                print(
                    f"Warning: Image dimensions ({img.width}x{img.height}) do not match layout ({page.width_px}x{page.height_px})",
                    file=sys.stderr,
                )
        else:
            img = Image.new("RGBA", (page.width_px, page.height_px), (255, 255, 255, 255))

        draw = ImageDraw.Draw(img, "RGBA")

        # Draw anchors (Blue)
        for anchor in page.anchors:
            label = f"{anchor.anchor_type}: {anchor.anchor_id}"
            draw_bounding_box(draw, anchor.bbox, label, "blue", (0, 0, 255, 30))

        # Draw ROIs (Red)
        for roi in page.rois:
            label = f"{roi.question_id}:{roi.option_id}"
            draw_bounding_box(draw, roi.bbox, label, "red", (255, 0, 0, 30))

        output_path = args.output_dir / f"debug_page_{page.page_number}.png"
        img.save(output_path)
        print(f"Generated {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

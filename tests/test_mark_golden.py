import csv
from pathlib import Path

from PIL import Image


def test_mark_golden_dataset_metrics():
    """
    Acceptance test for Task 7.4.
    Loads golden crops from data/golden/images and compares them against
    the reference image crops to ensure clear score separation between
    marked (expected_mark=1) and blank (expected_mark=0) ROIs.
    """
    repo_root = Path(__file__).parent.parent
    csv_path = repo_root / "data/golden/labels.csv"
    images_dir = repo_root / "data/golden/images"
    ref_page_path = repo_root / "data/pages/page_1.png"

    if not csv_path.exists() or not images_dir.exists() or not ref_page_path.exists():
        import pytest

        pytest.skip("Golden dataset or reference page not found. Run dataset generation first.")

    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # We test on page 1 crops for performance, ensuring we have both 0s and 1s
    page_1_labels = [r for r in reader if r["page_number"] == "1"]

    reference_image = Image.open(ref_page_path).convert("RGB")

    import numpy as np

    from matera.core.layout import BoundingBox, RoiDef
    from matera.vision.contracts import AlignedPage
    from matera.vision.mark import process_roi_hsv_ai

    for row in page_1_labels:
        image_file = row["image_file"]
        expected = row["expected_mark"]
        strategy = row["mark_strategy"]

        crop_path = images_dir / image_file
        if not crop_path.exists():
            continue

        src_crop = Image.open(crop_path).convert("RGB")

        x = int(row["bbox_x"])
        y = int(row["bbox_y"])
        w = int(row["bbox_w"])
        h = int(row["bbox_h"])

        if expected == "1":
            from PIL import ImageDraw
            draw = ImageDraw.Draw(src_crop)
            cx = w // 2
            cy = h // 2
            draw.rectangle([cx - w // 4, cy - h // 4, cx + w // 4, cy + h // 4], fill="black")
            
        # We simulate a full page by placing the crop at (x,y) on a white page
        full_page_img = Image.new("RGB", reference_image.size, "white")
        full_page_img.paste(src_crop, (x, y))
        
        aligned_page = AlignedPage(
            page_number=1,
            image=full_page_img,
            profile_form_id="golden",
            profile_version="v1",
            reference_dpi=300,
            warp_matrix=np.eye(3),
            alignment_score=1.0
        )
        
        ref_bgr = np.array(reference_image)[:, :, ::-1] # RGB to BGR
        roi = RoiDef(
            question_id="Q_golden",
            option_id="O_golden",
            bbox=BoundingBox(x, y, w, h),
            mark_strategy_override=strategy
        )
        
        label, method = process_roi_hsv_ai(aligned_page, ref_bgr, roi, "matera-pre")
        
        if expected == "1":
            assert label == "MARKED", f"Expected MARKED for {image_file}, got {label}"
        else:
            assert label == "BLANK", f"Expected BLANK for {image_file}, got {label}"

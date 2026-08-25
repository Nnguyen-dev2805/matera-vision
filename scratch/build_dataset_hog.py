import sys
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from matera.data.extract import extract_pages
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig
from matera.core.layout import load_layout_profile

DATASET_DIR = Path("data/dataset")
TRUE_MARKS_DIR = DATASET_DIR / "true_marks"
NOISE_DIR = DATASET_DIR / "noise"

TRUE_MARKS_DIR.mkdir(parents=True, exist_ok=True)
NOISE_DIR.mkdir(parents=True, exist_ok=True)

# Lấy các hàm filter từ v16
def get_local_roi_crops_hsv(aligned_image_rgb, median_ref_bgr, bbox, pad):
    crop_x1 = max(0, bbox.x - pad)
    crop_y1 = max(0, bbox.y - pad)
    crop_x2 = min(aligned_image_rgb.width, bbox.x + bbox.w + pad)
    crop_y2 = min(aligned_image_rgb.height, bbox.y + bbox.h + pad)
    
    crop_img = aligned_image_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    target_bgr = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
    ref_crop = median_ref_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(ref_gray, target_gray)
    blurred = cv2.GaussianBlur(diff, (3, 3), 0)
    _, mask_diff = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    
    hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask_color = cv2.threshold(saturation, 40, 255, cv2.THRESH_BINARY)
    
    mask_final = cv2.bitwise_and(mask_diff, mask_color)
    
    return target_bgr, mask_final, (crop_x1, crop_y1, crop_x2, crop_y2)

def main():
    layout_profile = load_layout_profile(Path("profiles/matera-pre/v1/layout.json"))
    pdf_path = Path("data/pdfs/matera-example.pdf")
    pages = list(extract_pages(pdf_path))
    
    ref_image_align = Image.open("data/pages/page_1.png")
    median_ref = Image.open("scratch/synthetic_median_reference.png")
    median_ref_bgr = cv2.cvtColor(np.array(median_ref), cv2.COLOR_RGB2BGR)
    
    with open("data/ground_truth.json", "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
        
    align_config = AlignmentConfig(algorithm="orb", transform_model="affine", inlier_threshold=0.05)
    
    counts = {"TP": 0, "FP": 0}
    
    for page_idx, page_img in enumerate(pages):
        print(f"Processing Page {page_idx + 1}...")
        aligned_page = align_page(page_img, ref_image_align, align_config)
        gt_page = ground_truth.get(f"page_{page_idx + 1}", {}).get("expected", {})
        
        for roi in layout_profile.pages[0].rois:
            if "Q14" not in roi.question_id:
                # Bỏ qua các câu khác trong tập dataset này vì Q14 là đặc thù nhất
                # Hoặc có thể lấy tất cả. Nhưng tạm thời lấy tất cả đi để dataset lớn.
                pass
            
            # Chỉ lấy các câu hỏi Checkbox
            if "Q13" in roi.question_id:
                continue
                
            q_key = roi.question_id
            if q_key not in gt_page:
                continue
                
            gt_answers = gt_page[q_key]
            mapped_answers = []
            if "Q14" in q_key:
                for ans in gt_answers:
                    if ans == "0": mapped_answers.append("1")
                    elif ans == "1": mapped_answers.append("3")
                    elif ans == "2": mapped_answers.append("2")
                    elif ans == "3": mapped_answers.append("4")
            else:
                for ans in gt_answers:
                    if ans == "0": mapped_answers.append("a")
                    elif ans == "1": mapped_answers.append("b")
                    elif ans == "2": mapped_answers.append("c")
                    elif ans == "3": mapped_answers.append("d")
                    elif ans == "4": mapped_answers.append("e")
                    elif ans == "5": mapped_answers.append("f")
            
            is_true_mark = (roi.option_id in mapped_answers)
            
            target_bgr, mask_raw, crop_coords = get_local_roi_crops_hsv(aligned_page.image, median_ref_bgr, roi.bbox, 0)
            
            h, w = mask_raw.shape
            safe_mask = np.zeros_like(mask_raw)
            m = 5
            safe_mask[m:h-m, m:w-m] = mask_raw[m:h-m, m:w-m]
            ink_pixels = np.sum(safe_mask > 0)
            
            if ink_pixels >= 10:
                # Save it
                save_dir = TRUE_MARKS_DIR if is_true_mark else NOISE_DIR
                if is_true_mark: counts["TP"] += 1
                else: counts["FP"] += 1
                
                # Lưu ảnh Mask để học (mask sạch, chuẩn)
                filename = f"p{page_idx+1}_{roi.question_id}_{roi.option_id}_{ink_pixels}px.png"
                cv2.imwrite(str(save_dir / filename), safe_mask)
                
    print(f"DONE! Dataset built. True Marks: {counts['TP']}, Noise: {counts['FP']}")

if __name__ == "__main__":
    main()

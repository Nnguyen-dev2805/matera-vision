import csv
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

from matera.data.extract import extract_pages
from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig
from matera.vision.mark import extract_mark_scores

import pandas as pd

def load_ground_truth(csv_path: Path) -> dict:
    gt = {}
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        key = f"P{row['Trang']}_{row['Câu hỏi']}_{row['Tùy chọn']}"
        gt[key] = str(row['Thực tế']).strip().lower() == "có"
    return gt

def main():
    pdf_path = Path("data/pdfs/matera-example.pdf")
    layout_profile = load_layout_profile(Path("profiles/matera-pre/v1/layout.json"))
    semantic_profile = load_semantic_profile(Path("profiles/matera-pre/v1/semantic.json"))
    gt = load_ground_truth(Path("data/debug/batch_eval/comparison_report.csv"))
    
    ref_image_align = Image.open("data/pages/page_1.png")
    
    train_pos = Path("data/dataset/train/1_marked")
    train_neg = Path("data/dataset/train/0_ghosting")
    test_pos = Path("data/dataset/test/1_marked")
    test_neg = Path("data/dataset/test/0_ghosting")
    
    for d in [train_pos, train_neg, test_pos, test_neg]:
        d.mkdir(parents=True, exist_ok=True)
    
    pages = list(extract_pages(pdf_path))
    align_config = AlignmentConfig(algorithm="orb", transform_model="affine", inlier_threshold=0.05)
    
    print("Aligning 10 pages...")
    aligned_pages = []
    for page in pages:
        aligned_pages.append(align_page(page, ref_image_align, align_config))
        
    print("Loading median reference...")
    median_ref = Image.open("scratch/synthetic_median_reference.png")
    
    train_pos_count = 0
    train_neg_count = 0
    test_pos_count = 0
    test_neg_count = 0
    
    for page_num, aligned_page in enumerate(aligned_pages, 1):
        print(f"Processing page {page_num}...")
        scores = extract_mark_scores(aligned_page, semantic_profile, layout_profile.pages[0], median_ref)
        
        # Decide if this page is Train or Test
        is_train = page_num <= 7
        
        for score in scores:
            if score.score > 0:
                key = f"P{page_num}_{score.question_id}_{score.option_id}"
                is_true_positive = gt.get(key, False)
                
                img_name = f"{key}.png"
                if is_train:
                    if is_true_positive:
                        score.image_crop.save(train_pos / img_name)
                        train_pos_count += 1
                    else:
                        score.image_crop.save(train_neg / img_name)
                        train_neg_count += 1
                else:
                    if is_true_positive:
                        score.image_crop.save(test_pos / img_name)
                        test_pos_count += 1
                    else:
                        score.image_crop.save(test_neg / img_name)
                        test_neg_count += 1

    print(f"Train Dataset (Pages 1-7): TP={train_pos_count}, FP={train_neg_count}")
    print(f"Test Dataset (Pages 8-10): TP={test_pos_count}, FP={test_neg_count}")

if __name__ == "__main__":
    main()

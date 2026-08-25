import csv
from pathlib import Path
from PIL import Image

from matera.core.layout import PageLayout
from matera.core.profile import FormProfile
from matera.data.contracts import RenderedPage
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig, RoutingConfig
from matera.vision.mark import extract_mark_scores
from matera.vision.routing import route_page


def build_training_dataset(
    labels_csv_path: str,
    semantic_profile: FormProfile,
    page_layout: PageLayout,
    routing_config: RoutingConfig,
    alignment_config: AlignmentConfig | None = None,
) -> list[dict]:
    """
    Parses the ground truth dataset, runs the deterministic vision pipeline,
    and extracts ROIFeature sets for any ROI that falls into the 'needs_review'
    zone (ambiguous).
    """
    if alignment_config is None:
        alignment_config = AlignmentConfig(
            transform_model="affine",
            inlier_threshold=0.05,
        )

    # 1. Load Ground Truth
    # Map (page_number, question_id, option_id) -> expected_mark
    ground_truth = {}
    pages = set()
    with open(labels_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            page_num = int(row["page_number"])
            q_id = row["question_id"]
            opt_id = row["option_id"]
            expected = int(row["expected_mark"])
            ground_truth[(page_num, q_id, opt_id)] = expected
            pages.add((page_num, row["source_page_image"]))

    # 2. Process each unique page
    dataset = []
    
    # Load reference image for alignment
    ref_path = Path("data/pages/page_1.png")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found at {ref_path}")
    reference_img = Image.open(ref_path)

    for page_num, image_name in sorted(pages):
        img_path = Path("data/pages") / image_name
        if not img_path.exists():
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue
            
        raw_img = Image.open(img_path)
        rendered = RenderedPage(
            page_number=page_num,
            width_px=raw_img.width,
            height_px=raw_img.height,
            pdf_width_pt=1.0,
            pdf_height_pt=1.0,
            image=raw_img,
        )
        
        try:
            aligned_page = align_page(rendered, reference_img, alignment_config)
            mark_scores = extract_mark_scores(
                aligned_page,
                semantic_profile,
                page_layout,
                reference_img,
                debug_dir="data/debug/evidence"
            )
            
            # Map features by Q/O
            feature_map = { (ms.question_id, ms.option_id): ms for ms in mark_scores }
            
            # Use route_page to exactly match the baseline logic
            normalized_result = route_page(mark_scores, semantic_profile, page_num, routing_config)
            
            for ans in normalized_result.answers:
                if ans.resolution_status == "needs_review":
                    ms = feature_map.get((ans.answer_key.question_id, ans.answer_key.option_id))
                    if not ms:
                        continue
                        
                    key = (page_num, ms.question_id, ms.option_id)
                    if key in ground_truth:
                        expected_mark = ground_truth[key]
                        
                        dataset.append({
                            "page_number": page_num,
                            "question_id": ms.question_id,
                            "option_id": ms.option_id,
                            "dark_pixel_ratio": ms.features.dark_pixel_ratio,
                            "foreground_area_ratio": ms.features.foreground_area_ratio,
                            "contour_count": ms.features.contour_count,
                            "largest_component_ratio": ms.features.largest_component_ratio,
                            "bbox_fill_ratio": ms.features.bbox_fill_ratio,
                            "deterministic_score": ms.score,
                            "expected_mark": expected_mark
                        })
                        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing {image_name}: {e}")
            
    return dataset

import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from matera.core.layout import LayoutProfile
from matera.core.profile import FormProfile


class AnnotationError(Exception):
    """Raised when ground truth annotations do not match the semantic profile."""

    pass


def load_profile_semantic(semantic_profile: FormProfile) -> dict[str, list[str]]:
    """
    Extracts a mapping of question_id -> list of option_ids from a validated FormProfile.
    """
    mapping = {}
    for q in semantic_profile.questions:
        question_id = q.question_id
        mapping[question_id] = [str(opt.option_id) for opt in q.options]
    return mapping


def parse_ground_truth(
    ground_truth_json: dict[str, Any], semantic_map: dict[str, list[str]], expected_pages: int = 10
) -> dict[str, dict[str, list[str]]]:
    """
    Parses the ground_truth.json and maps option indices (like "0") to semantic option_ids.
    Returns: { "page_1": { "Q1": ["a", "b"], "Q13.1": ["0"] }, ... }
    """
    parsed = {}

    # Check for missing pages
    for page_num in range(1, expected_pages + 1):
        page_key = f"page_{page_num}"
        if page_key not in ground_truth_json:
            raise AnnotationError(f"Missing annotations for {page_key} in ground truth.")

    for page_key, page_data in ground_truth_json.items():
        if not page_key.startswith("page_"):
            continue

        if "expected" not in page_data:
            raise AnnotationError(f"Missing 'expected' block in {page_key}")

        expected_marks = page_data["expected"]
        page_mapped = {}

        for q_key, indices in expected_marks.items():
            # Handle Q13_1 -> Q13.1 alias mapping
            semantic_q_key = q_key.replace("_", ".")

            if semantic_q_key not in semantic_map:
                raise AnnotationError(f"Question {semantic_q_key} not found in semantic profile.")

            valid_options = semantic_map[semantic_q_key]
            mapped_options = []

            for idx_str in indices:
                try:
                    idx = int(idx_str)
                except ValueError:
                    raise AnnotationError(f"Invalid option index '{idx_str}' in {page_key} {q_key}")

                if idx < 0 or idx >= len(valid_options):
                    raise AnnotationError(
                        f"Index out of bounds for {page_key} {q_key}: "
                        f"{idx} (max {len(valid_options) - 1})"
                    )

                mapped_options.append(valid_options[idx])

            page_mapped[semantic_q_key] = mapped_options

        # Ensure all questions in semantic_map are present in this page
        for q_key in semantic_map:
            if q_key not in page_mapped:
                raise AnnotationError(f"Question {q_key} is missing from {page_key} annotations.")

        parsed[page_key] = page_mapped

    return parsed


def load_semantic_metadata(semantic_profile: FormProfile) -> dict[str, dict[str, str]]:
    """
    Extracts metadata (response_type, mark_strategy) for each question.
    """
    meta = {}
    for q in semantic_profile.questions:
        meta[q.question_id] = {"response_type": q.response_type, "mark_strategy": q.mark_strategy}
    return meta


def generate_golden_dataset(
    pages_dir: Path,
    semantic_profile: FormProfile,
    layout_profile: LayoutProfile,
    ground_truth_json: dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Crops image patches based on layout.json and labels them using ground_truth.json.
    Writes output atomically.
    """
    # 1. Load and parse annotations
    semantic_map = load_profile_semantic(semantic_profile)
    semantic_meta = load_semantic_metadata(semantic_profile)

    # Hardcode expected 10 pages based on the fixed dataset spec
    num_expected_pages = 10
    parsed_gt = parse_ground_truth(
        ground_truth_json,
        semantic_map,
        expected_pages=num_expected_pages,
    )

    form_id = layout_profile.form_id
    form_version = layout_profile.form_version

    if not layout_profile.pages:
        raise ValueError("No pages found in layout profile")

    # We only have one page layout in the fixed profile right now
    page_layout = layout_profile.pages[0]
    rois = page_layout.rois

    # 2. Validate that all required source pages exist BEFORE doing any work
    source_paths = {}
    for page_num in range(1, num_expected_pages + 1):
        source_filename = f"page_{page_num}.png"
        source_path = pages_dir / source_filename
        if not source_path.exists():
            raise FileNotFoundError(f"Source page image not found: {source_path}")
        source_paths[page_num] = source_path

    # 3. Create temp directory for atomic writing
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="golden_", dir=output_dir.parent))

    try:
        tmp_images_dir = tmp_dir / "images"
        tmp_images_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_dir / "labels.csv"

        csv_headers = [
            "image_file",
            "form_id",
            "form_version",
            "page_number",
            "source_page_image",
            "question_id",
            "option_id",
            "response_type",
            "mark_strategy",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "annotation_source",
            "expected_mark",
            "split",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()

            # Loop through pages
            for page_num in range(1, num_expected_pages + 1):
                page_key = f"page_{page_num}"
                source_path = source_paths[page_num]
                source_filename = source_path.name
                page_gt = parsed_gt[page_key]
                split = "train" if page_num <= 5 else "dev"

                with Image.open(source_path) as img:
                    for roi in rois:
                        q_id = roi.question_id
                        o_id = roi.option_id
                        bbox = roi.bbox
                        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h

                        # Expected mark logic
                        expected_mark = 0
                        if q_id in page_gt and o_id in page_gt[q_id]:
                            expected_mark = 1

                        # Crop and save
                        crop_box = (x, y, x + w, y + h)
                        patch = img.crop(crop_box)

                        # e.g., page_1_Q2_a.png
                        safe_q_id = q_id.replace(".", "_")
                        patch_filename = f"{page_key}_{safe_q_id}_{o_id}.png"
                        patch_path = tmp_images_dir / patch_filename

                        # Save lossless
                        patch.save(patch_path, format="PNG", compress_level=6, optimize=False)

                        meta = semantic_meta.get(q_id, {})

                        writer.writerow(
                            {
                                "image_file": patch_filename,
                                "form_id": form_id,
                                "form_version": form_version,
                                "page_number": page_num,
                                "source_page_image": source_filename,
                                "question_id": q_id,
                                "option_id": o_id,
                                "response_type": meta.get("response_type", ""),
                                "mark_strategy": meta.get("mark_strategy", ""),
                                "bbox_x": x,
                                "bbox_y": y,
                                "bbox_w": w,
                                "bbox_h": h,
                                "annotation_source": "human_ground_truth.json",
                                "expected_mark": expected_mark,
                                "split": split,
                            }
                        )

        # 4. Atomic promotion
        if output_dir.exists():
            # If the user didn't ask to overwrite or something, we can just replace it
            shutil.rmtree(output_dir)
        os.rename(tmp_dir, output_dir)

    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

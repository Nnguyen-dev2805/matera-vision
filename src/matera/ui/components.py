from copy import deepcopy

import pandas as pd
from PIL import ImageDraw

from matera.core.contracts import NormalizedPageResult
from matera.core.profile import FormProfile
from matera.core.layout import PageLayout
from matera.vision.contracts import AlignedPage, MarkScore


def draw_debug_boxes(aligned_page: AlignedPage, result: NormalizedPageResult, layout: PageLayout) -> None:
    """
    Draws red/green bounding boxes on the aligned page image in-place based on the result.
    Green for selected, Red for unselected, Orange for needs_review.
    """
    draw = ImageDraw.Draw(aligned_page.image)
    
    # Create lookup for status
    status_map = {}
    for ans in result.answers:
        if ans.resolution_status == "needs_review":
            color = "orange"
        elif ans.selected:
            color = "green"
        else:
            color = "red"
        status_map[(ans.answer_key.question_id, ans.answer_key.option_id)] = color
        
    for roi in layout.rois:
        color = status_map.get((roi.question_id, roi.option_id), "gray")
        
        # Draw rectangle
        draw.rectangle(
            [roi.bbox.x, roi.bbox.y, roi.bbox.x + roi.bbox.w, roi.bbox.y + roi.bbox.h],
            outline=color,
            width=3
        )


def result_to_dataframe(result: NormalizedPageResult, semantic_profile: FormProfile, scores: list[MarkScore] = None) -> pd.DataFrame:
    """
    Converts a NormalizedPageResult into a readable pandas DataFrame for the UI.
    """
    rows = []
    
    # Create a lookup for question titles and option labels
    q_lookup = {q.question_id: q for q in semantic_profile.questions}
    
    # Create a lookup for raw scores if provided
    score_map = {}
    if scores:
        for s in scores:
            score_map[(s.question_id, s.option_id)] = s

    for ans in result.answers:
        q_def = q_lookup.get(ans.answer_key.question_id)
        opt_label = ans.answer_key.option_id
        
        if q_def:
            for opt in q_def.options:
                if opt.option_id == ans.answer_key.option_id:
                    opt_label = opt.label or opt_label
                    break
                    
        row_data = {
            "Câu hỏi": ans.answer_key.question_id,
            "ID Tùy chọn": opt_label,
            "Lựa chọn": "Có" if ans.selected else ("Không" if ans.selected is False else "N/A"),
            "Cần Review": "Có" if ans.resolution_status == "needs_review" else "Không",
            "Độ tin cậy": f"{ans.confidence:.2f}" if ans.confidence is not None else "-",
            "Nguồn": ans.decision_source
        }
        
        if scores:
            raw_score = score_map.get((ans.answer_key.question_id, ans.answer_key.option_id))
            if raw_score:
                row_data["Điểm (Score)"] = f"{raw_score.score:.2f}"
                row_data["Phương pháp (Method)"] = raw_score.decision_source
                
        rows.append(row_data)
        
    return pd.DataFrame(rows)

import pytest
import numpy as np
from PIL import Image

from matera.vision.evidence import (
    PageMarkEvidence,
    AlignmentEvidence,
    ReferenceEvidence,
    QuestionMarkEvidence,
    OptionMarkEvidence,
    MarkThresholdEvidence,
    evidence_to_mark_scores,
    evidence_to_json_dict
)


def test_evidence_to_json_dict():
    # Setup some test data with NumPy types
    alignment = AlignmentEvidence(
        alignment_score=0.95,
        warp_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        image_size=(1000, 2000),
        layout_size=(1000, 2000),
        scale_x=1.0,
        scale_y=1.0
    )
    thresholds = MarkThresholdEvidence(
        marked_threshold_deg=20.0,
        blank_threshold_deg=10.0,
        routing_low_threshold=0.2,
        routing_high_threshold=0.8,
        local_pad_px=10,
        outer_radius_px=20,
        diff_threshold=30,
        global_pad_px=10
    )
    
    evidence = PageMarkEvidence(
        page_number=np.int32(1),
        form_id="test_form",
        form_version="v1",
        alignment=alignment,
        reference=ReferenceEvidence(width=1000, height=2000, dpi=300),
        questions=(),
        thresholds=thresholds
    )
    
    result = evidence_to_json_dict(evidence)
    
    # Assert
    assert isinstance(result, dict)
    assert result["page_number"] == 1
    assert isinstance(result["page_number"], int) # Converted from np.int32
    assert result["form_id"] == "test_form"
    assert result["alignment"]["alignment_score"] == 0.95
    assert result["alignment"]["warp_matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_evidence_to_mark_scores():
    option1 = OptionMarkEvidence(
        question_id="q1",
        option_id="o1",
        strategy="checkbox",
        legacy_prediction="MARKED",
        legacy_method="local_radial",
        legacy_score=1.0,
        selected_by_global=False,
        local=None,
        hsv_fallback=None,
        suspicion_notes=()
    )
    option2 = OptionMarkEvidence(
        question_id="q1",
        option_id="o2",
        strategy="checkbox",
        legacy_prediction="BLANK",
        legacy_method="local_radial",
        legacy_score=0.0,
        selected_by_global=False,
        local=None,
        hsv_fallback=None,
        suspicion_notes=()
    )
    option3 = OptionMarkEvidence(
        question_id="q1",
        option_id="o3",
        strategy="checkbox",
        legacy_prediction="AMBIGUOUS",
        legacy_method="hsv_fallback",
        legacy_score=0.5,
        selected_by_global=False,
        local=None,
        hsv_fallback=None,
        suspicion_notes=()
    )
    
    question = QuestionMarkEvidence(
        question_id="q1",
        strategy="checkbox",
        response_type="single",
        rois=(),
        global_topology=None,
        option_evidence=(option1, option2, option3)
    )
    
    evidence = PageMarkEvidence(
        page_number=1,
        form_id="test_form",
        form_version="v1",
        alignment=AlignmentEvidence(0.95, ((1,0,0),(0,1,0),(0,0,1)), (100, 100), (100, 100), 1.0, 1.0),
        reference=ReferenceEvidence(100, 100, 300),
        questions=(question,),
        thresholds=MarkThresholdEvidence(20, 10, None, None, 10, 20, 30, 10)
    )
    
    scores = evidence_to_mark_scores(evidence)
    
    assert len(scores) == 3
    assert scores[0].question_id == "q1"
    assert scores[0].option_id == "o1"
    assert scores[0].score == 1.0
    assert scores[0].method == "local_radial"
    
    assert scores[1].option_id == "o2"
    assert scores[1].score == 0.0
    
    assert scores[2].option_id == "o3"
    assert scores[2].score == 0.5

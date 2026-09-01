import numpy as np

from matera.vision.evidence import (
    AlignmentEvidence,
    MarkThresholdEvidence,
    PageMarkEvidence,
    ReferenceEvidence,
    evidence_to_json_dict,
)


def test_evidence_to_json_dict():
    # Setup some test data with NumPy types
    alignment = AlignmentEvidence(
        alignment_score=0.95,
        warp_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        image_size=(1000, 2000),
        layout_size=(1000, 2000),
        scale_x=1.0,
        scale_y=1.0,
    )
    thresholds = MarkThresholdEvidence(
        marked_threshold_deg=20.0,
        blank_threshold_deg=10.0,
        routing_low_threshold=0.2,
        routing_high_threshold=0.8,
        local_pad_px=10,
        outer_radius_px=20,
        diff_threshold=30,
        global_pad_px=10,
    )

    evidence = PageMarkEvidence(
        page_number=np.int32(1),
        form_id="test_form",
        form_version="v1",
        alignment=alignment,
        reference=ReferenceEvidence(width=1000, height=2000, dpi=300),
        questions=(),
        thresholds=thresholds,
    )

    result = evidence_to_json_dict(evidence)

    # Assert
    assert isinstance(result, dict)
    assert result["page_number"] == 1
    assert isinstance(result["page_number"], int)  # Converted from np.int32
    assert result["form_id"] == "test_form"
    assert result["alignment"]["alignment_score"] == 0.95
    assert result["alignment"]["warp_matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_local_option_evidence():
    import cv2

    from matera.vision.evidence import LocalOptionEvidence, compute_local_option_evidence

    # Create a dummy mask (100x100) with a filled circle
    mask_raw = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask_raw, (50, 50), 20, 255, -1)

    # Dummy text bbox
    text_bbox = (40, 40, 20, 20)
    crop_coords = (0, 0, 100, 100)

    evidence = compute_local_option_evidence(mask_raw, text_bbox, crop_coords, outer_radius=32)

    assert isinstance(evidence, LocalOptionEvidence)
    assert evidence.radial_degrees_covered > 0
    assert evidence.radial_active_bin_count > 0
    assert evidence.radial_ink_pixels > 0

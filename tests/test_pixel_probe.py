import json

import cv2
import numpy as np
from PIL import Image

from matera.core.layout import BoundingBox, RoiDef
from matera.core.profile import FormProfile, OptionDef, QuestionDef
from matera.tools.pixel_probe import analyze_roi_pixels, build_routing_traces
from matera.vision.contracts import RoutingConfig


def test_analyze_roi_pixels_explains_local_radial_marked_circle():
    reference = Image.new("RGB", (140, 140), "white")
    marked = np.full((140, 140, 3), 255, dtype=np.uint8)
    cv2.ellipse(marked, (70, 70), (20, 20), 0, 0, 220, (0, 0, 0), 4)
    aligned = Image.fromarray(marked)

    roi = RoiDef("Q1", "a", BoundingBox(50, 50, 40, 40))

    trace, masks = analyze_roi_pixels(
        aligned_image=aligned,
        reference_image=reference,
        roi=roi,
        strategy="circle",
        is_global_marked=False,
        page_number=1,
    )

    assert trace.prediction == "MARKED"
    assert trace.method == "LOCAL_RADIAL"
    assert trace.score == 1.0
    assert trace.radial_degrees_covered >= 180.0
    assert trace.radial_ink_pixels > 0
    assert trace.radial_active_bin_count > 0
    assert "radial_mask" in masks
    assert masks["radial_mask"].shape == masks["diff_mask"].shape


def test_build_routing_traces_explains_under_selection_for_highest_score():
    profile = FormProfile(
        form_id="matera-test",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="Q1",
                response_type="single_select",
                mark_strategy="circle",
                options=(OptionDef("a"), OptionDef("b")),
                min_selections=1,
                max_selections=1,
            ),
        ),
    )

    traces = build_routing_traces(
        profile=profile,
        page_number=1,
        roi_traces=[
            {"question_id": "Q1", "option_id": "a", "score": 0.10, "method": "LOCAL_RADIAL"},
            {"question_id": "Q1", "option_id": "b", "score": 0.15, "method": "LOCAL_RADIAL"},
        ],
        config=RoutingConfig(low_threshold=0.2, high_threshold=0.6),
    )

    assert traces[("Q1", "a")]["selected"] is False
    assert traces[("Q1", "a")]["resolution_status"] == "resolved"
    assert traces[("Q1", "b")]["selected"] is None
    assert traces[("Q1", "b")]["resolution_status"] == "needs_review"
    assert "Under-selection" in traces[("Q1", "b")]["review_reason"]


def test_analyze_roi_pixels_trace_is_json_serializable_with_opencv_ints():
    reference_cv = np.full((140, 140, 3), 255, dtype=np.uint8)
    cv2.rectangle(reference_cv, (66, 66), (74, 74), (0, 0, 0), -1)
    reference = Image.fromarray(reference_cv)

    marked_cv = reference_cv.copy()
    cv2.ellipse(marked_cv, (70, 70), (20, 20), 0, 0, 220, (0, 0, 0), 4)
    aligned = Image.fromarray(marked_cv)

    trace, _ = analyze_roi_pixels(
        aligned_image=aligned,
        reference_image=reference,
        roi=RoiDef("Q1", "a", BoundingBox(50, 50, 40, 40)),
        strategy="circle",
        is_global_marked=False,
        page_number=1,
    )

    json.dumps(trace.as_dict())


def test_new_report_models_serializability():
    import dataclasses
    from matera.tools.pixel_probe import PixelProbeReport, GlobalTopologyTrace, _json_ready

    report = PixelProbeReport(
        report_version=2,
        source_pdf="test.pdf",
        page_number=1,
        alignment_score=1.0,
        warp_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        filters={"question": None, "option": None, "routing_evaluated": True},
        thresholds={},
        questions=[],
        global_topology={
            "Q1": GlobalTopologyTrace(
                question_id="Q1",
                ran=True,
                skip_reason=None,
                group_crop={},
                option_centers=[],
                contours=[],
                clusters=[],
                global_marked=[],
                artifacts={},
            )
        },
        traces=[],
        artifacts={},
    )
    data = _json_ready(dataclasses.asdict(report))
    assert data["report_version"] == 2


def test_trace_global_topology():
    from matera.tools.pixel_probe import trace_global_topology
    from matera.core.layout import BoundingBox, RoiDef

    reference = Image.new("RGB", (140, 140), "white")
    marked = np.full((140, 140, 3), 255, dtype=np.uint8)
    # Draw a contour that should trigger global marking
    cv2.circle(marked, (70, 70), 30, (0, 0, 0), 2)
    aligned = Image.fromarray(marked)
    
    # Needs at least 2 rois
    rois = [
        RoiDef("Q1", "a", BoundingBox(50, 50, 10, 10)),
        RoiDef("Q1", "b", BoundingBox(80, 50, 10, 10))
    ]
    
    trace = trace_global_topology(
        aligned_image_rgb=aligned,
        median_ref_bgr=np.array(reference),
        rois=rois,
        question_id="Q1"
    )
    
    assert trace.question_id == "Q1"
    assert trace.ran is True
    assert isinstance(trace.global_marked, list)

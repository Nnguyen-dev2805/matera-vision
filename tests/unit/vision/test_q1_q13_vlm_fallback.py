import numpy as np
import pytest
from PIL import Image

from matera.core.layout import BoundingBox, PageLayout, RoiDef
from matera.core.profile import FormProfile, OptionDef, QuestionDef
from matera.vision.contracts import AlignedPage
from matera.vision.question_vlm_resolver import QuestionVlmRuntime
from matera.vision.scoring.extraction import extract_mark_evidence
from matera.vlm.models import VlmResponse


class FakeVlmClient:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def generate_json(self, request):
        return VlmResponse(raw_text=self.response_text, provider_metadata={})


@pytest.fixture
def test_aligned_page():
    # Make a dummy image (200x200)
    img = Image.new("RGB", (200, 200), color="white")
    import cv2

    img_cv = np.array(img)
    # Draw something that looks like ink on A
    cv2.circle(img_cv, (50, 50), 3, (0, 0, 0), -1)  # small enough to be ambiguous by HSV maybe?
    img = Image.fromarray(img_cv)

    return AlignedPage(
        page_number=1,
        image=img,
        profile_form_id="test",
        profile_version="v1",
        reference_dpi=300,
        warp_matrix=np.eye(3),
        alignment_score=1.0,
    )


@pytest.fixture
def test_profile_and_layout():
    semantic = FormProfile(
        form_id="test",
        form_version="v1",
        questions=(
            QuestionDef(
                question_id="Q1",
                response_type="single_select",
                mark_strategy="circle",
                options=(OptionDef("A"), OptionDef("B")),
                max_selections=1,
            ),
        ),
    )

    layout = PageLayout(
        page_number=1,
        width_px=200,
        height_px=200,
        anchors=(),
        rois=(
            RoiDef(question_id="Q1", option_id="A", bbox=BoundingBox(40, 40, 20, 20)),
            RoiDef(question_id="Q1", option_id="B", bbox=BoundingBox(100, 100, 20, 20)),
        ),
    )
    return semantic, layout


def test_vlm_fallback_triggered_on_ambiguous(test_aligned_page, test_profile_and_layout):
    semantic, layout = test_profile_and_layout
    ref_img = Image.new("RGB", (200, 200), color="white")

    # VLM will return MARKED for A, BLANK for B
    client = FakeVlmClient('{"options": {"A": {"state": "MARKED"}, "B": {"state": "BLANK"}}}')
    vlm_runtime = QuestionVlmRuntime(client=client, model="test")

    # The small circle drawn in the fixture will result in AMBIGUOUS from deterministic HSV
    # because it's > 20 but < 200 ink pixels.
    from unittest.mock import patch

    from matera.vision.evidence.models import LocalOptionEvidence

    mock_local = LocalOptionEvidence(
        crop_coords={"x1": 0, "y1": 0, "x2": 0, "y2": 0},
        diff_ink_pixels=0,
        diff_contour_count=0,
        diff_contour_areas=(),
        text_bbox={"x": 0, "y": 0, "w": 0, "h": 0},
        core_mask_pixels=0,
        outer_mask_pixels=0,
        radial_mask_before_morph_px=0,
        radial_ink_pixels=0,
        radial_active_bin_count=0,
        radial_degrees_covered=120.0,  # between blank and marked
        radial_histogram=(),
    )
    from matera.vision.detectors.hsv import HsvAiResult

    mock_hsv = HsvAiResult(
        legacy_prediction="AMBIGUOUS",
        legacy_method="FALLBACK_HSV_UNCERTAIN",
        prob_mark=None,
        stroke_ev=None,
        checkbox_decision=None,
    )

    mock_local_blank = LocalOptionEvidence(
        crop_coords={"x1": 0, "y1": 0, "x2": 0, "y2": 0},
        diff_ink_pixels=0,
        diff_contour_count=0,
        diff_contour_areas=(),
        text_bbox={"x": 0, "y": 0, "w": 0, "h": 0},
        core_mask_pixels=0,
        outer_mask_pixels=0,
        radial_mask_before_morph_px=0,
        radial_ink_pixels=0,
        radial_active_bin_count=0,
        radial_degrees_covered=10.0,  # <= 90 (blank)
        radial_histogram=(),
    )

    with (
        patch(
            "matera.vision.scoring.extraction.compute_local_option_evidence",
            side_effect=[mock_local, mock_local_blank],
        ),
        patch("matera.vision.detectors.hsv.process_roi_hsv", return_value=mock_hsv),
    ):
        page_evidence = extract_mark_evidence(
            test_aligned_page,
            semantic,
            layout,
            ref_img,
            question_vlm_runtime=vlm_runtime,
        )

    q1_ev = page_evidence.questions[0]
    opt_a = next(o for o in q1_ev.option_evidence if o.option_id == "A")
    opt_b = next(o for o in q1_ev.option_evidence if o.option_id == "B")

    assert opt_a.legacy_prediction == "MARKED"
    assert opt_a.legacy_method == "Q1_Q13_VLM_MARKED"
    assert opt_a.vlm is not None
    assert opt_a.vlm.decision == "MARKED"

    # Option B was empty in the image, so CV returned BLANK. VLM scope should only be A.
    # So B's prediction should remain CV's BLANK.
    assert opt_b.legacy_prediction == "BLANK"
    assert opt_b.legacy_method == "LOCAL_RADIAL"  # CV method
    # VLM was NOT used for B, because B was not ambiguous and not in a global cluster
    assert opt_b.vlm is None


def test_confident_cv_no_vlm(test_aligned_page, test_profile_and_layout):
    semantic, layout = test_profile_and_layout
    ref_img = Image.new("RGB", (200, 200), color="white")

    client = FakeVlmClient('{"options": {"A": {"state": "MARKED"}, "B": {"state": "BLANK"}}}')
    vlm_runtime = QuestionVlmRuntime(client=client, model="test")

    from unittest.mock import patch

    from matera.vision.evidence.models import LocalOptionEvidence

    mock_local_marked = LocalOptionEvidence(
        crop_coords={"x1": 0, "y1": 0, "x2": 0, "y2": 0},
        diff_ink_pixels=0,
        diff_contour_count=0,
        diff_contour_areas=(),
        text_bbox={"x": 0, "y": 0, "w": 0, "h": 0},
        core_mask_pixels=0,
        outer_mask_pixels=0,
        radial_mask_before_morph_px=0,
        radial_ink_pixels=0,
        radial_active_bin_count=0,
        radial_degrees_covered=180.0,
        radial_histogram=(),
    )
    mock_local_blank = LocalOptionEvidence(
        crop_coords={"x1": 0, "y1": 0, "x2": 0, "y2": 0},
        diff_ink_pixels=0,
        diff_contour_count=0,
        diff_contour_areas=(),
        text_bbox={"x": 0, "y": 0, "w": 0, "h": 0},
        core_mask_pixels=0,
        outer_mask_pixels=0,
        radial_mask_before_morph_px=0,
        radial_ink_pixels=0,
        radial_active_bin_count=0,
        radial_degrees_covered=10.0,
        radial_histogram=(),
    )

    with patch(
        "matera.vision.scoring.extraction.compute_local_option_evidence",
        side_effect=[mock_local_marked, mock_local_blank],
    ):
        page_evidence = extract_mark_evidence(
            test_aligned_page,
            semantic,
            layout,
            ref_img,
            question_vlm_runtime=vlm_runtime,
        )

    q1_ev = page_evidence.questions[0]
    opt_a = next(o for o in q1_ev.option_evidence if o.option_id == "A")
    opt_b = next(o for o in q1_ev.option_evidence if o.option_id == "B")

    assert opt_a.legacy_prediction == "MARKED"
    assert opt_b.legacy_prediction == "BLANK"
    assert opt_a.vlm is None
    assert opt_b.vlm is None


def test_global_shape_risk_does_not_trigger_fallback(test_aligned_page, test_profile_and_layout):
    semantic, layout = test_profile_and_layout
    ref_img = Image.new("RGB", (200, 200), color="white")

    client = FakeVlmClient('{"options": {"A": {"state": "MARKED"}, "B": {"state": "BLANK"}}}')
    vlm_runtime = QuestionVlmRuntime(client=client, model="test")

    from unittest.mock import patch

    from matera.vision.evidence.models import GlobalTopologyEvidence, LocalOptionEvidence

    mock_local_marked = LocalOptionEvidence(
        crop_coords={"x1": 0, "y1": 0, "x2": 0, "y2": 0},
        diff_ink_pixels=0,
        diff_contour_count=0,
        diff_contour_areas=(),
        text_bbox={"x": 0, "y": 0, "w": 0, "h": 0},
        core_mask_pixels=0,
        outer_mask_pixels=0,
        radial_mask_before_morph_px=0,
        radial_ink_pixels=0,
        radial_active_bin_count=0,
        radial_degrees_covered=180.0,
        radial_histogram=(),
    )

    mock_gt = GlobalTopologyEvidence(
        question_id="Q1",
        ran=True,
        skip_reason=None,
        crop={"x1": 0, "y1": 0, "x2": 200, "y2": 200},
        crop_expansion=None,
        option_centers=(),
        contours=(),
        clusters=(),
        global_marked=frozenset(["A", "B"]),
        shape_diagnostics=None,  # Not attached in production anymore
    )

    with (
        patch(
            "matera.vision.scoring.extraction.compute_local_option_evidence",
            side_effect=[mock_local_marked, mock_local_marked],
        ),
        patch(
            "matera.vision.scoring.extraction.compute_global_topology_evidence",
            return_value=mock_gt,
        ),
    ):
        page_evidence = extract_mark_evidence(
            test_aligned_page,
            semantic,
            layout,
            ref_img,
            question_vlm_runtime=vlm_runtime,
        )

    q1_ev = page_evidence.questions[0]
    
    # Assert shape_diagnostics is None in production
    assert q1_ev.global_topology.shape_diagnostics is None

    opt_a = next(o for o in q1_ev.option_evidence if o.option_id == "A")
    opt_b = next(o for o in q1_ev.option_evidence if o.option_id == "B")

    # Because local evidence is confident and we don't escalate on shape diagnostics anymore
    assert opt_a.vlm is None
    assert opt_b.vlm is None


def test_no_client_scores_0_5(test_aligned_page, test_profile_and_layout):
    semantic, layout = test_profile_and_layout
    ref_img = Image.new("RGB", (200, 200), color="white")

    # Pass client=None
    vlm_runtime = QuestionVlmRuntime(client=None, model="test")

    from unittest.mock import patch

    from matera.vision.detectors.hsv import HsvAiResult
    from matera.vision.evidence import evidence_to_mark_scores
    from matera.vision.evidence.models import LocalOptionEvidence

    mock_local = LocalOptionEvidence(
        crop_coords={"x1": 0, "y1": 0, "x2": 0, "y2": 0},
        diff_ink_pixels=0,
        diff_contour_count=0,
        diff_contour_areas=(),
        text_bbox={"x": 0, "y": 0, "w": 0, "h": 0},
        core_mask_pixels=0,
        outer_mask_pixels=0,
        radial_mask_before_morph_px=0,
        radial_ink_pixels=0,
        radial_active_bin_count=0,
        radial_degrees_covered=120.0,  # triggers ambiguity
        radial_histogram=(),
    )
    mock_hsv = HsvAiResult(
        legacy_prediction="AMBIGUOUS",
        legacy_method="FALLBACK",
        prob_mark=None,
        stroke_ev=None,
        checkbox_decision=None,
    )

    with (
        patch(
            "matera.vision.scoring.extraction.compute_local_option_evidence",
            side_effect=[mock_local, mock_local],
        ),
        patch("matera.vision.detectors.hsv.process_roi_hsv", return_value=mock_hsv),
    ):
        page_evidence = extract_mark_evidence(
            test_aligned_page,
            semantic,
            layout,
            ref_img,
            question_vlm_runtime=vlm_runtime,
        )

    q1_ev = page_evidence.questions[0]
    opt_a = next(o for o in q1_ev.option_evidence if o.option_id == "A")
    assert opt_a.legacy_prediction == "AMBIGUOUS"
    assert opt_a.legacy_method == "Q1_Q13_VLM_CLIENT_UNAVAILABLE"

    scores = evidence_to_mark_scores(page_evidence, test_aligned_page)
    score_a = next(s for s in scores if s.option_id == "A")
    assert score_a.score == 0.5


def test_q14_does_not_use_q1_q13_prompt():
    from matera.vision.q14_vlm_resolver import Q14_PROMPT
    from matera.vision.question_vlm_resolver import QUESTION_PROMPT

    assert Q14_PROMPT != QUESTION_PROMPT
    assert "question 14" in Q14_PROMPT.lower()
    assert "question 14" not in QUESTION_PROMPT.lower()

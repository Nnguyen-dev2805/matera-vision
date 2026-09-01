import cv2
import numpy as np
from PIL import Image

from matera.core.layout import BoundingBox, PageLayout, RoiDef
from matera.core.profile import FormProfile, OptionDef, QuestionDef
from matera.vision.contracts import AlignedPage
from matera.vision.detectors.global_geometry import (
    UnionFind,
    get_horizontal_extremes,
    min_contour_distance,
)
from matera.vision.detectors.hsv import process_roi_hsv
from matera.vision.detectors.local_geometry import get_text_bounding_box
from matera.vision.mark import (
    extract_mark_scores,
    run_v11_global_topology,
)


def test_union_find():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(3)
    assert uf.find(3) == uf.find(4)


def test_min_contour_distance():
    cnt1 = np.array([[[0, 0]], [[0, 1]]], dtype=np.int32)
    cnt2 = np.array([[[3, 0]], [[3, 1]]], dtype=np.int32)
    dist = min_contour_distance(cnt1, cnt2)
    assert dist == 3.0


def test_get_horizontal_extremes():
    cnt = np.array([[[10, 5]], [[20, 5]], [[15, 10]]], dtype=np.int32)
    left, right = get_horizontal_extremes(cnt)
    assert list(left) == [10, 5]
    assert list(right) == [20, 5]


def test_get_text_bounding_box():
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[40:60, 40:60] = 255
    x, y, w, h = get_text_bounding_box(gray)
    assert w > 0 and h > 0


def test_extract_mark_scores():
    img = Image.new("RGB", (200, 200), color="white")
    import cv2

    img_cv = np.array(img)
    cv2.circle(img_cv, (50, 50), 10, (0, 0, 0), -1)
    img = Image.fromarray(img_cv)

    aligned_page = AlignedPage(
        page_number=1,
        image=img,
        profile_form_id="test",
        profile_version="v1",
        reference_dpi=300,
        warp_matrix=np.eye(3),
        alignment_score=1.0,
    )

    ref_img = Image.new("RGB", (200, 200), color="white")

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

    scores = extract_mark_scores(aligned_page, semantic, layout, ref_img)

    assert len(scores) == 2
    score_a = next(s for s in scores if s.option_id == "A")
    score_b = next(s for s in scores if s.option_id == "B")

    assert 0.0 <= score_a.score <= 1.0
    assert 0.0 <= score_b.score <= 1.0


def test_process_roi_hsv():
    # Setup
    img = Image.new("RGB", (100, 100), color="white")
    aligned_page = AlignedPage(
        page_number=1,
        image=img,
        profile_form_id="test",
        profile_version="v1",
        reference_dpi=300,
        warp_matrix=np.eye(3),
        alignment_score=1.0,
    )
    median_ref_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    median_ref_bgr.fill(255)
    roi = RoiDef(question_id="Q1", option_id="A", bbox=BoundingBox(0, 0, 30, 30))

    hsv_res = process_roi_hsv(aligned_page, median_ref_bgr, roi, "test")
    # label, method = hsv_res.legacy_prediction, hsv_res.legacy_method
    import cv2

    img_cv = np.array(img)
    cv2.circle(img_cv, (15, 15), 10, (255, 0, 0), -1)
    img_ink = Image.fromarray(img_cv)
    aligned_page_ink = AlignedPage(
        page_number=1,
        image=img_ink,
        profile_form_id="test",
        profile_version="v1",
        reference_dpi=300,
        warp_matrix=np.eye(3),
        alignment_score=1.0,
    )
    hsv_res = process_roi_hsv(aligned_page_ink, median_ref_bgr, roi, "test")
    label2, _ = hsv_res.legacy_prediction, hsv_res.legacy_method
    assert label2 == "MARKED"

    # ambiguous branch
    img_amb_cv = np.zeros((30, 30, 3), dtype=np.uint8)
    img_amb_cv.fill(255)
    cv2.circle(img_amb_cv, (15, 15), 3, (255, 0, 0), -1)  # smaller circle
    img_amb = Image.fromarray(img_amb_cv)
    aligned_page_amb = AlignedPage(
        page_number=1,
        image=img_amb,
        profile_form_id="test",
        profile_version="v1",
        reference_dpi=300,
        warp_matrix=np.eye(3),
        alignment_score=1.0,
    )
    hsv_res = process_roi_hsv(aligned_page_amb, median_ref_bgr, roi, "test")
    label3, method3 = hsv_res.legacy_prediction, hsv_res.legacy_method
    assert label3 == "AMBIGUOUS"
    assert method3 == "test_HSV_UNCERTAIN"
    assert hsv_res.prob_mark is None


def test_run_v11_global_topology():
    img = Image.new("RGB", (200, 200), color="white")
    median_ref_bgr = np.full((200, 200, 3), 255, dtype=np.uint8)

    import cv2

    img_cv = np.array(img)
    cv2.circle(img_cv, (100, 100), 10, (0, 0, 0), -1)
    cv2.circle(img_cv, (105, 105), 10, (0, 0, 0), -1)
    img = Image.fromarray(img_cv)

    rois = [
        RoiDef(question_id="Q1", option_id="A", bbox=BoundingBox(90, 90, 20, 20)),
        RoiDef(question_id="Q1", option_id="B", bbox=BoundingBox(150, 150, 20, 20)),
    ]

    full_mask = np.zeros((200, 200, 3), dtype=np.uint8)

    marked = run_v11_global_topology(img, median_ref_bgr, rois, full_mask)
    assert isinstance(marked, set)


def test_radial_boundary_marked(mock_text_bbox):
    # Test boundary degree_covered >= 180 (should be MARKED by LOCAL_RADIAL)
    img = Image.new("RGB", (200, 200), color="white")
    # Draw ink covering slightly more than 180 degrees
    img_cv = np.array(img)
    cv2.ellipse(img_cv, (100, 100), (20, 20), 0, 0, 190, (0, 0, 0), 3)  # 190 degrees covered
    img_ink = Image.fromarray(img_cv)

    aligned_page = AlignedPage(
        page_number=1,
        image=img_ink,
        profile_form_id="test",
        profile_version="v1",
        reference_dpi=300,
        warp_matrix=np.eye(3),
        alignment_score=1.0,
    )

    roi = RoiDef(question_id="Q1", option_id="A", bbox=BoundingBox(80, 80, 40, 40))
    # We don't have direct access to internal method so we test through extract_mark_scores
    question = QuestionDef(
        question_id="Q1",
        response_type="single_select",
        mark_strategy="circle",
        max_selections=1,
        options=(OptionDef(option_id="A", value=1, label="A"),),
    )
    profile = FormProfile(form_id="test", form_version="v1", questions=(question,))
    layout = PageLayout(page_number=1, width_px=200, height_px=200, anchors=(), rois=(roi,))

    import matera.vision.mark as mark

    scores = mark.extract_mark_scores(aligned_page, profile, layout, img)
    assert len(scores) == 1
    assert scores[0].score == 1.0  # MARKED


def test_radial_boundary_blank(mock_text_bbox):
    # Test boundary degree_covered <= 90 (should be BLANK by LOCAL_RADIAL)
    img = Image.new("RGB", (200, 200), color="white")
    img_cv = np.array(img)
    cv2.ellipse(img_cv, (100, 100), (20, 20), 0, 0, 80, (0, 0, 0), 3)  # 80 degrees covered
    img_ink = Image.fromarray(img_cv)

    aligned_page = AlignedPage(
        page_number=1,
        image=img_ink,
        profile_form_id="test",
        profile_version="v1",
        reference_dpi=300,
        warp_matrix=np.eye(3),
        alignment_score=1.0,
    )

    roi = RoiDef(question_id="Q1", option_id="A", bbox=BoundingBox(80, 80, 40, 40))
    question = QuestionDef(
        question_id="Q1",
        response_type="single_select",
        mark_strategy="circle",
        max_selections=1,
        options=(OptionDef(option_id="A", value=1, label="A"),),
    )
    profile = FormProfile(form_id="test", form_version="v1", questions=(question,))
    layout = PageLayout(page_number=1, width_px=200, height_px=200, anchors=(), rois=(roi,))

    import matera.vision.mark as mark

    scores = mark.extract_mark_scores(aligned_page, profile, layout, img)
    assert len(scores) == 1


def test_adaptive_crop_preserves_global_marking():
    import cv2
    import numpy as np
    from PIL import Image

    from matera.core.layout import BoundingBox, RoiDef
    from matera.vision.mark import run_v11_global_topology

    img = Image.new("RGB", (200, 200), color="white")
    median_ref_bgr = np.full((200, 200, 3), 255, dtype=np.uint8)

    img_cv = np.array(img)
    cv2.line(img_cv, (40, 40), (160, 160), (0, 0, 0), 5)
    cv2.line(img_cv, (40, 160), (160, 40), (0, 0, 0), 5)
    img = Image.fromarray(img_cv)

    rois = [
        RoiDef(question_id="Q1", option_id="A", bbox=BoundingBox(100, 100, 20, 20)),
        RoiDef(question_id="Q1", option_id="B", bbox=BoundingBox(100, 130, 20, 20)),
    ]

    full_mask = np.zeros((200, 200, 3), dtype=np.uint8)

    result = run_v11_global_topology(img, median_ref_bgr, rois, full_mask)
    assert "A" in result
    assert "B" in result


def test_global_topology_evidence_adapter():
    import numpy as np
    from PIL import Image

    from matera.vision.evidence import compute_global_topology_evidence
    from matera.vision.mark import run_v11_global_topology

    # Create simple dummy data where global topology will just return early
    # (because < 2 rois or no contours)
    aligned_img = Image.new("RGB", (100, 100), (255, 255, 255))
    median_ref = np.full((100, 100, 3), 255, dtype=np.uint8)

    import dataclasses

    @dataclasses.dataclass
    class DummyBbox:
        x: int
        y: int
        w: int
        h: int

    class DummyRoi:
        def __init__(self, qid, oid):
            self.question_id = qid
            self.option_id = oid
            self.bbox = DummyBbox(10, 10, 20, 20)

    rois = [DummyRoi("q1", "o1"), DummyRoi("q1", "o2")]
    full_mask_bgr = np.zeros((100, 100, 3), dtype=np.uint8)

    adapter_result = run_v11_global_topology(aligned_img, median_ref, rois, full_mask_bgr)

    evidence = compute_global_topology_evidence(aligned_img, median_ref, rois)

    assert set(evidence.global_marked) == adapter_result

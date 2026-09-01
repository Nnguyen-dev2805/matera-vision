import pytest
from PIL import Image

from matera.core.layout import BoundingBox, RoiDef
from matera.vision.question_vlm_resolver import QuestionVlmRuntime, resolve_question_with_vlm
from matera.vlm.models import VlmResponse


class FakeClient:
    def __init__(self, response_text: str = "", exception_to_raise: Exception | None = None):
        self.response_text = response_text
        self.exception_to_raise = exception_to_raise
        self.last_request = None

    def generate_json(self, request):
        self.last_request = request
        if self.exception_to_raise:
            raise self.exception_to_raise
        return VlmResponse(raw_text=self.response_text, provider_metadata={})


@pytest.fixture
def dummy_rois():
    return [
        RoiDef(question_id="Q1", option_id="A", bbox=BoundingBox(100, 100, 20, 20)),
        RoiDef(question_id="Q1", option_id="B", bbox=BoundingBox(100, 150, 20, 20)),
    ]


@pytest.fixture
def dummy_image():
    return Image.new("RGB", (500, 500), color="white")


def test_resolve_question_no_client(dummy_image, dummy_rois):
    runtime = QuestionVlmRuntime(client=None, model="test-model")
    evidence = resolve_question_with_vlm(
        aligned_image=dummy_image,
        question_id="Q1",
        rois=dummy_rois,
        option_ids=["A", "B"],
        runtime=runtime,
    )
    assert evidence.provider_error == "VLM client unavailable"
    assert len(evidence.options) == 2
    for opt in evidence.options:
        assert opt.decision == "NEED_REVIEW"
        assert "unavailable" in opt.parse_error


def test_resolve_question_with_valid_response(dummy_image, dummy_rois, tmp_path):
    response = '{"options": {"A": {"state": "MARKED"}, "B": {"state": "BLANK"}}}'
    client = FakeClient(response_text=response)
    runtime = QuestionVlmRuntime(client=client, model="test", debug_dir=tmp_path)

    evidence = resolve_question_with_vlm(
        aligned_image=dummy_image,
        question_id="Q1",
        rois=dummy_rois,
        option_ids=["A", "B"],
        runtime=runtime,
    )

    assert evidence.provider_error is None
    assert evidence.options[0].option_id == "A"
    assert evidence.options[0].decision == "MARKED"
    assert evidence.options[1].option_id == "B"
    assert evidence.options[1].decision == "BLANK"

    # Verify debug artifacts are written
    assert (tmp_path / "Q1_vlm_crop.png").exists()
    assert (tmp_path / "Q1_vlm_raw.txt").exists()


def test_resolve_question_provider_error(dummy_image, dummy_rois):
    client = FakeClient(exception_to_raise=RuntimeError("API down"))
    runtime = QuestionVlmRuntime(client=client, model="test")

    evidence = resolve_question_with_vlm(
        aligned_image=dummy_image,
        question_id="Q1",
        rois=dummy_rois,
        option_ids=["A", "B"],
        runtime=runtime,
    )

    assert evidence.provider_error == "API down"
    assert len(evidence.options) == 2
    for opt in evidence.options:
        assert opt.decision == "NEED_REVIEW"
        assert opt.parse_error == "API down"

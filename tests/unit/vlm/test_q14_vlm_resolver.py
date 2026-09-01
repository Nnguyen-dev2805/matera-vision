import pytest
from PIL import Image

from matera.core.layout import BoundingBox
from matera.core.q14_vlm_profile import Q14VlmProfile
from matera.vision.q14_vlm_resolver import (
    Q14VlmClient,
    Q14VlmRuntime,
    resolve_q14_with_vlm,
)
from matera.vlm.models import VlmRequest, VlmResponse


class FakeVlmClient(Q14VlmClient):
    def __init__(self, response_text: str, should_raise: bool = False):
        self.response_text = response_text
        self.should_raise = should_raise
        self.last_request = None

    def classify_q14(self, request: VlmRequest) -> VlmResponse:
        self.last_request = request
        if self.should_raise:
            raise RuntimeError("Fake provider error")
        return VlmResponse(raw_text=self.response_text, provider_metadata={})


@pytest.fixture
def dummy_profile() -> Q14VlmProfile:
    return Q14VlmProfile(
        schema_version=1,
        question_id="Q14",
        crop_box=BoundingBox(x=10, y=10, w=100, h=50),
        option_order=("1", "2", "3", "4"),
        option_mapping={"1": "TL", "2": "TR", "3": "BL", "4": "BR"},
        prompt_version="v1",
    )


@pytest.fixture
def dummy_image() -> Image.Image:
    # 200x200 black image
    return Image.new("RGB", (200, 200), color="black")


def test_resolver_no_client(dummy_image, dummy_profile):
    runtime = Q14VlmRuntime(profile=dummy_profile, client=None, model="test-model")
    evidence = resolve_q14_with_vlm(dummy_image, runtime)

    assert evidence.provider_error == "Q14 VLM client unavailable"
    assert len(evidence.options) == 4
    for opt in evidence.options:
        assert opt.decision == "NEED_REVIEW"
        assert opt.parse_error == "Q14 VLM client unavailable"


def test_resolver_provider_error(dummy_image, dummy_profile):
    client = FakeVlmClient("", should_raise=True)
    runtime = Q14VlmRuntime(profile=dummy_profile, client=client, model="test-model")
    evidence = resolve_q14_with_vlm(dummy_image, runtime)

    assert evidence.provider_error == "Fake provider error"
    for opt in evidence.options:
        assert opt.decision == "NEED_REVIEW"
        assert opt.parse_error == "Fake provider error"


def test_resolver_success(dummy_image, dummy_profile, tmp_path):
    response_json = """
    {
        "question_id": "Q14",
        "options": {
            "1": {"state": "MARKED", "reason": "visible X"},
            "2": {"state": "BLANK", "reason": "empty"},
            "3": {"state": "UNCERTAIN", "reason": "smudge"},
            "4": {"state": "INVALID_STATE", "reason": "foo"}
        }
    }
    """
    client = FakeVlmClient(response_json)
    runtime = Q14VlmRuntime(
        profile=dummy_profile, client=client, model="test-model", debug_dir=tmp_path
    )
    evidence = resolve_q14_with_vlm(dummy_image, runtime)

    assert evidence.provider_error is None
    opts = {o.option_id: o for o in evidence.options}

    # Check parse_q14_vlm_response logic mappings
    assert opts["1"].decision == "MARKED"
    assert opts["2"].decision == "BLANK"
    assert opts["3"].decision == "NEED_REVIEW"
    assert opts["4"].decision == "NEED_REVIEW"

    assert (tmp_path / "q14_vlm_crop.png").exists()
    assert (tmp_path / "q14_vlm_raw.txt").exists()


def test_resolver_clamp_crop(dummy_profile):
    # Smaller than crop box (10, 10, 100, 50) => goes out of bounds
    img = Image.new("RGB", (50, 50), color="black")
    client = FakeVlmClient("{}")
    runtime = Q14VlmRuntime(profile=dummy_profile, client=client, model="test-model")
    evidence = resolve_q14_with_vlm(img, runtime)

    assert evidence.crop_box["w"] == 40  # 50 - 10
    assert evidence.crop_box["h"] == 40  # 50 - 10

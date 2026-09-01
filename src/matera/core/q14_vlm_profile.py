import json
from dataclasses import dataclass
from pathlib import Path

from matera.core.layout import BoundingBox


@dataclass(frozen=True)
class Q14VlmProfile:
    schema_version: int
    question_id: str
    crop_box: BoundingBox
    option_order: tuple[str, ...]
    option_mapping: dict[str, str]
    prompt_version: str


def load_q14_vlm_profile(path: Path) -> Q14VlmProfile:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"Invalid schema_version {schema_version}, expected 1")

    question_id = data.get("question_id")
    if question_id != "Q14":
        raise ValueError(f"Invalid question_id {question_id}, expected Q14")

    crop_dict = data.get("crop_box", {})
    x = crop_dict.get("x")
    y = crop_dict.get("y")
    w = crop_dict.get("w")
    h = crop_dict.get("h")

    if x is None or y is None or w is None or h is None:
        raise ValueError("crop_box missing required fields (x, y, w, h)")
    if w <= 0 or h <= 0:
        raise ValueError(f"crop_box w and h must be positive, got w={w}, h={h}")

    crop_box = BoundingBox(x=x, y=y, w=w, h=h)

    option_order = tuple(data.get("option_order", []))
    if option_order != ("1", "2", "3", "4"):
        raise ValueError(f"Invalid option_order {option_order}, expected ('1', '2', '3', '4')")

    option_mapping = data.get("option_mapping", {})
    for opt in option_order:
        if opt not in option_mapping:
            raise ValueError(f"option_mapping missing required option {opt}")

    prompt_version = data.get("prompt_version")
    if not prompt_version:
        raise ValueError("prompt_version is required")

    return Q14VlmProfile(
        schema_version=schema_version,
        question_id=question_id,
        crop_box=crop_box,
        option_order=option_order,
        option_mapping=option_mapping,
        prompt_version=prompt_version,
    )

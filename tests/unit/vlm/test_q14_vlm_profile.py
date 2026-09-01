import json
from pathlib import Path

import pytest

from matera.core.q14_vlm_profile import load_q14_vlm_profile


def test_load_q14_vlm_profile_success(tmp_path: Path):
    profile_path = tmp_path / "q14_vlm.json"
    with open(profile_path, "w") as f:
        json.dump(
            {
                "schema_version": 1,
                "question_id": "Q14",
                "crop_box": {"x": 80, "y": 3170, "w": 2285, "h": 197},
                "option_order": ["1", "2", "3", "4"],
                "option_mapping": {
                    "1": "left top",
                    "2": "right top",
                    "3": "left bot",
                    "4": "right bot",
                },
                "prompt_version": "q14_vlm_v1",
            },
            f,
        )

    profile = load_q14_vlm_profile(profile_path)
    assert profile.schema_version == 1
    assert profile.question_id == "Q14"
    assert profile.crop_box.w == 2285
    assert profile.crop_box.h == 197
    assert profile.option_order == ("1", "2", "3", "4")
    assert profile.option_mapping["1"] == "left top"


def test_load_q14_vlm_profile_invalid_schema(tmp_path: Path):
    profile_path = tmp_path / "q14_vlm.json"
    with open(profile_path, "w") as f:
        json.dump({"schema_version": 2}, f)

    with pytest.raises(ValueError, match="Invalid schema_version 2, expected 1"):
        load_q14_vlm_profile(profile_path)


def test_load_q14_vlm_profile_missing_options(tmp_path: Path):
    profile_path = tmp_path / "q14_vlm.json"
    with open(profile_path, "w") as f:
        json.dump(
            {
                "schema_version": 1,
                "question_id": "Q14",
                "crop_box": {"x": 80, "y": 3170, "w": 2285, "h": 197},
                "option_order": ["1", "2", "3", "4"],
                "option_mapping": {
                    "1": "left top",
                    "2": "right top",
                },
                "prompt_version": "q14_vlm_v1",
            },
            f,
        )

    with pytest.raises(ValueError, match="option_mapping missing required option 3"):
        load_q14_vlm_profile(profile_path)


def test_load_q14_vlm_profile_invalid_crop_box(tmp_path: Path):
    profile_path = tmp_path / "q14_vlm.json"
    with open(profile_path, "w") as f:
        json.dump(
            {
                "schema_version": 1,
                "question_id": "Q14",
                "crop_box": {"x": 80, "y": 3170, "w": -10, "h": 197},
                "option_order": ["1", "2", "3", "4"],
                "option_mapping": {
                    "1": "left top",
                    "2": "right top",
                    "3": "left bot",
                    "4": "right bot",
                },
                "prompt_version": "q14_vlm_v1",
            },
            f,
        )

    with pytest.raises(ValueError, match="crop_box w and h must be positive"):
        load_q14_vlm_profile(profile_path)

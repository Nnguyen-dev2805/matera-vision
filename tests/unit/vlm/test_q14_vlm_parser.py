import json

from matera.vlm.parser import parse_q14_vlm_response


def test_valid_json_parses_correctly():
    raw_text = json.dumps(
        {
            "question_id": "Q14",
            "options": {
                "1": {"state": "MARKED", "reason": "visible"},
                "2": {"state": "BLANK", "reason": "empty"},
                "3": {"state": "MARKED", "reason": "X"},
                "4": {"state": "BLANK", "reason": "none"},
            },
        }
    )
    results = parse_q14_vlm_response(raw_text)
    assert results["1"]["vlm_state"] == "MARKED"
    assert results["2"]["vlm_state"] == "BLANK"
    assert results["3"]["vlm_state"] == "MARKED"
    assert results["4"]["vlm_state"] == "BLANK"

    assert results["1"]["parse_error"] is None
    assert results["1"]["reason"] == "visible"
    assert results["1"]["raw_vlm_state"] == "MARKED"


def test_uncertain_normalizes_to_need_review():
    raw_text = json.dumps(
        {
            "question_id": "Q14",
            "options": {
                "1": {"state": "UNCERTAIN", "reason": "faint"},
                "2": {"state": "BLANK", "reason": "empty"},
                "3": {"state": "MARKED", "reason": "X"},
                "4": {"state": "BLANK", "reason": "none"},
            },
        }
    )
    results = parse_q14_vlm_response(raw_text)
    assert results["1"]["vlm_state"] == "NEED_REVIEW"
    assert results["1"]["parse_error"] is None


def test_missing_option_becomes_need_review():
    raw_text = json.dumps(
        {
            "question_id": "Q14",
            "options": {
                "1": {"state": "MARKED", "reason": "visible"},
                # option 2 is missing
                "3": {"state": "MARKED", "reason": "X"},
                "4": {"state": "BLANK", "reason": "none"},
            },
        }
    )
    results = parse_q14_vlm_response(raw_text)
    assert results["2"]["vlm_state"] == "NEED_REVIEW"
    assert "Missing option" in results["2"]["parse_error"]


def test_invalid_state_becomes_need_review():
    raw_text = json.dumps(
        {
            "question_id": "Q14",
            "options": {
                "1": {"state": "YES", "reason": "visible"},
                "2": {"state": "BLANK", "reason": "empty"},
                "3": {"state": "MARKED", "reason": "X"},
                "4": {"state": "BLANK", "reason": "none"},
            },
        }
    )
    results = parse_q14_vlm_response(raw_text)
    assert results["1"]["vlm_state"] == "NEED_REVIEW"
    assert "Invalid state" in results["1"]["parse_error"]
    assert results["1"]["raw_vlm_state"] == "YES"


def test_invalid_json_makes_all_need_review():
    raw_text = "This is not json"
    results = parse_q14_vlm_response(raw_text)
    for opt in ["1", "2", "3", "4"]:
        assert results[opt]["vlm_state"] == "NEED_REVIEW"
        assert "Invalid JSON" in results[opt]["parse_error"]

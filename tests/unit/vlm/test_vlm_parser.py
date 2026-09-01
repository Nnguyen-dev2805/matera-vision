from matera.vlm.parser import parse_vlm_option_response


def test_parse_vlm_valid_response():
    raw_text = (
        '{"question_id": "Q1", "options": {"A": {"state": "MARKED"}, "B": {"state": "BLANK"}}}'
    )
    res = parse_vlm_option_response(raw_text, question_id="Q1", option_ids=["A", "B"])
    assert res["A"]["vlm_state"] == "MARKED"
    assert res["B"]["vlm_state"] == "BLANK"
    assert res["A"]["parse_error"] is None


def test_parse_vlm_uncertain():
    raw_text = '{"question_id": "Q1", "options": {"A": {"state": "UNCERTAIN"}}}'
    res = parse_vlm_option_response(raw_text, question_id="Q1", option_ids=["A"])
    assert res["A"]["vlm_state"] == "NEED_REVIEW"
    assert res["A"]["parse_error"] is None


def test_parse_vlm_missing_option():
    raw_text = '{"question_id": "Q1", "options": {"A": {"state": "MARKED"}}}'
    res = parse_vlm_option_response(raw_text, question_id="Q1", option_ids=["A", "B"])
    assert res["A"]["vlm_state"] == "MARKED"
    assert res["B"]["vlm_state"] == "NEED_REVIEW"
    assert "Missing option 'B'" in res["B"]["parse_error"]


def test_parse_vlm_invalid_json():
    raw_text = "{invalid json}"
    res = parse_vlm_option_response(raw_text, question_id="Q1", option_ids=["A"])
    assert res["A"]["vlm_state"] == "NEED_REVIEW"
    assert "Invalid JSON" in res["A"]["parse_error"]


def test_parse_vlm_invalid_state():
    raw_text = '{"question_id": "Q1", "options": {"A": {"state": "MAYBE"}}}'
    res = parse_vlm_option_response(raw_text, question_id="Q1", option_ids=["A"])
    assert res["A"]["vlm_state"] == "NEED_REVIEW"
    assert "Invalid state" in res["A"]["parse_error"]

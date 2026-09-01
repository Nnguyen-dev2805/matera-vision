import json
from typing import Any, Sequence


def parse_vlm_option_response(
    raw_text: str, *, question_id: str, option_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """
    Parses the raw text from the VLM dynamically for any list of option_ids.
    Returns a dictionary mapping option_id to a dictionary with keys:
    - vlm_state: Literal["MARKED", "BLANK", "NEED_REVIEW"]
    - raw_vlm_state: str | None
    - reason: str | None
    - parse_error: str | None
    """
    results = {}

    # Initialize defaults
    for opt in option_ids:
        results[opt] = {
            "vlm_state": "NEED_REVIEW",
            "raw_vlm_state": None,
            "reason": None,
            "parse_error": "Missing from response",
        }

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        for opt in option_ids:
            results[opt]["parse_error"] = f"Invalid JSON: {str(e)}"
        return results

    options_data = data.get("options", {})
    if not isinstance(options_data, dict):
        for opt in option_ids:
            results[opt]["parse_error"] = "Invalid JSON schema: 'options' is not a dictionary"
        return results

    for opt in option_ids:
        opt_data = options_data.get(opt)
        if not opt_data or not isinstance(opt_data, dict):
            results[opt]["parse_error"] = f"Missing option '{opt}' or not a dictionary"
            continue

        raw_state = opt_data.get("state")
        reason = opt_data.get("reason")

        results[opt]["raw_vlm_state"] = str(raw_state) if raw_state is not None else None
        results[opt]["reason"] = str(reason) if reason is not None else None

        if raw_state == "MARKED":
            results[opt]["vlm_state"] = "MARKED"
            results[opt]["parse_error"] = None
        elif raw_state == "BLANK":
            results[opt]["vlm_state"] = "BLANK"
            results[opt]["parse_error"] = None
        elif raw_state == "UNCERTAIN":
            results[opt]["vlm_state"] = "NEED_REVIEW"
            results[opt]["parse_error"] = None
        else:
            results[opt]["vlm_state"] = "NEED_REVIEW"
            results[opt]["parse_error"] = f"Invalid state: {raw_state}"

    return results


def parse_q14_vlm_response(raw_text: str) -> dict[str, dict[str, Any]]:
    """
    Parses the raw text from the VLM for Q14.
    """
    return parse_vlm_option_response(
        raw_text=raw_text, question_id="Q14", option_ids=("1", "2", "3", "4")
    )

import json
from pathlib import Path
from typing import Any

from matera.core.contracts import NormalizedPageResult
from matera.core.profile import FormProfile


def generate_headers(profile: FormProfile) -> list[str]:
    """Dynamically generate column headers based on profile definitions."""
    columns = ["form_id", "form_version", "page_number", "page_status"]

    for q in profile.questions:
        for opt in q.options:
            columns.append(f"{q.question_id}_{opt.option_id}")

    columns.append("review_tasks")
    return columns


def flatten_result(result: NormalizedPageResult, profile: FormProfile) -> list[Any]:
    """Converts a NormalizedPageResult into a flat list of cell values matching the headers."""
    # Index the answers by key
    answer_map = {
        (ans.answer_key.question_id, ans.answer_key.option_id): ans for ans in result.answers
    }

    row: list[Any] = [
        result.form_id,
        result.form_version,
        result.page_number,
        result.page_status,
    ]

    for q in profile.questions:
        for opt in q.options:
            key = (q.question_id, opt.option_id)
            ans = answer_map.get(key)

            if ans is None or ans.resolution_status == "needs_review":
                row.append("")
            else:
                row.append(1 if ans.selected is True else 0)

    # Serialize review tasks if any
    if result.review_tasks:
        tasks_json = [
            {
                "question": task.answer_key.question_id,
                "option": task.answer_key.option_id,
                "reason": task.reason,
            }
            for task in result.review_tasks
        ]
        row.append(json.dumps(tasks_json))
    else:
        row.append("")

    return row


def export_to_excel(
    results: list[NormalizedPageResult], profile: FormProfile, output_path: Path
) -> None:
    pass

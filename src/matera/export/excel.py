import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook

from matera.core.contracts import NormalizedPageResult
from matera.core.profile import FormProfile


def generate_headers(profile: FormProfile) -> list[str]:
    """Dynamically generate column headers based on profile definitions."""
    columns = ["file_name", "form_id", "form_version", "page_number", "page_status"]

    for q in profile.questions:
        for opt in q.options:
            columns.append(f"{q.question_id}_{opt.option_id}")

    columns.append("review_tasks")
    return columns


def flatten_result(result: NormalizedPageResult, profile: FormProfile) -> dict[str, Any]:
    """Converts a NormalizedPageResult into a dictionary of column_name -> cell_value."""
    # Index the answers by key
    answer_map = {
        (ans.answer_key.question_id, ans.answer_key.option_id): ans for ans in result.answers
    }

    row: dict[str, Any] = {
        "file_name": getattr(result, "file_name", "") or "",
        "form_id": result.form_id,
        "form_version": result.form_version,
        "page_number": result.page_number,
        "page_status": result.page_status,
    }

    for q in profile.questions:
        for opt in q.options:
            key = (q.question_id, opt.option_id)
            ans = answer_map.get(key)

            col_name = f"{q.question_id}_{opt.option_id}"

            if ans is None or ans.resolution_status == "needs_review":
                row[col_name] = ""
            else:
                row[col_name] = 1 if ans.selected is True else 0

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
        row["review_tasks"] = json.dumps(tasks_json)
    else:
        row["review_tasks"] = ""

    return row


def export_to_excel(
    results: list[NormalizedPageResult], profile: FormProfile, output_path: Path
) -> None:
    """Exports a list of NormalizedPageResults to an Excel file (.xlsx) based on the FormProfile."""
    wb = Workbook()
    ws = wb.active

    headers = generate_headers(profile)
    ws.append(headers)

    for result in results:
        row_dict = flatten_result(result, profile)
        row = [row_dict.get(h, "") for h in headers]
        ws.append(row)

    wb.save(output_path)

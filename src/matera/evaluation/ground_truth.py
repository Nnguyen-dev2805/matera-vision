import json
from dataclasses import dataclass
from pathlib import Path

from matera.core.profile import FormProfile


@dataclass
class GroundTruthItem:
    item_id: str
    pdf_path: Path
    page_num: int
    profile_path: Path
    layout_path: Path
    labels: dict[str, list[str] | None]
    notes: dict[str, str]


@dataclass
class ExpectedOption:
    question_id: str
    option_id: str
    expected_state: str  # "MARKED", "BLANK", or "UNKNOWN"


@dataclass
class GroundTruthDataset:
    schema_version: int
    description: str
    items: list[GroundTruthItem]


def load_ground_truth(json_path: Path) -> GroundTruthDataset:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("schema_version") != 1:
        raise ValueError("Unsupported schema version. Expected 1.")

    items = []
    for item in data.get("items", []):
        pdf_path = Path(item["pdf"])
        if not pdf_path.is_absolute():
            pdf_path = json_path.parent / pdf_path

        profile_path = Path(item["profile"])
        if not profile_path.is_absolute():
            profile_path = json_path.parent / profile_path

        layout_path = Path(item["layout"])
        if not layout_path.is_absolute():
            layout_path = json_path.parent / layout_path

        items.append(
            GroundTruthItem(
                item_id=item["item_id"],
                pdf_path=pdf_path,
                page_num=item["page"],
                profile_path=profile_path,
                layout_path=layout_path,
                labels=item.get("labels", {}),
                notes=item.get("notes", {}),
            )
        )

    return GroundTruthDataset(
        schema_version=data.get("schema_version", 1),
        description=data.get("description", ""),
        items=items,
    )


def expand_ground_truth_labels(item: GroundTruthItem, profile: FormProfile) -> list[ExpectedOption]:
    """Expands question-level labels into option-level ExpectedOption objects."""
    expected_options = []
    for q_def in profile.questions:
        q_id = q_def.question_id
        if q_id not in item.labels:
            continue

        marked_options = item.labels[q_id]

        if marked_options is None:
            # Human uncertain, exclude from metrics
            for opt_def in q_def.options:
                expected_options.append(
                    ExpectedOption(
                        question_id=q_id,
                        option_id=opt_def.option_id,
                        expected_state="UNKNOWN",
                    )
                )
        else:
            for opt_def in q_def.options:
                is_marked = opt_def.option_id in marked_options
                expected_options.append(
                    ExpectedOption(
                        question_id=q_id,
                        option_id=opt_def.option_id,
                        expected_state="MARKED" if is_marked else "BLANK",
                    )
                )

    return expected_options

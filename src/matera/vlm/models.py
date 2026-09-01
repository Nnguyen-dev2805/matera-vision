from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class VlmRequest:
    model: str
    image_path: Path
    prompt: str
    temperature: float = 0.0


@dataclass(frozen=True)
class VlmResponse:
    raw_text: str
    provider_metadata: dict[str, Any]


@dataclass(frozen=True)
class Q14VlmOptionResult:
    page_number: int
    option_id: str
    expected_state: str | None
    vlm_state: Literal["MARKED", "BLANK", "NEED_REVIEW"]
    raw_vlm_state: str | None
    reason: str | None
    parse_error: str | None
    image_path: str

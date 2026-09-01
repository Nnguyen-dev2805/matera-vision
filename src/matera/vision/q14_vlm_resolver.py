import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from matera.core.q14_vlm_profile import Q14VlmProfile
from matera.vision.evidence.models import VlmOptionEvidence, VlmQuestionEvidence
from matera.vlm.models import VlmRequest, VlmResponse
from matera.vlm.parser import parse_q14_vlm_response

Q14_PROMPT = """You are grading question 14 in a scanned Vietnamese worksheet.

Look only at question 14 in the image.
There are four checkboxes arranged as:
{option_mapping_text}

Valid student marks are an X mark or a V/check mark in or crossing the checkbox.
Ignore printed checkbox borders, printed text, shadows, paper color, scanning
noise, and faint background artifacts.
A single blue slash or line crossing a checkbox is BLANK, not MARKED, unless
it clearly forms a deliberate X or V/check mark.

For each option, decide:
- MARKED: a deliberate X or V/check mark is visible in or crossing the checkbox.
- BLANK: no deliberate student mark is visible.
- UNCERTAIN: the image is too ambiguous to decide safely.

Return JSON only, exactly in this schema:
{{
  "question_id": "Q14",
  "options": {{
{options_schema_text}
  }}
}}
"""


class Q14VlmClient(Protocol):
    def classify_q14(self, request: VlmRequest) -> VlmResponse: ...


@dataclass(frozen=True)
class Q14VlmRuntime:
    profile: Q14VlmProfile
    client: Q14VlmClient | None
    model: str
    timeout_s: float = 60.0
    debug_dir: Path | None = None


def resolve_q14_with_vlm(
    aligned_image: Image.Image,
    runtime: Q14VlmRuntime,
) -> VlmQuestionEvidence:
    """Resolve Q14 mark states using a VLM."""
    profile = runtime.profile
    box = profile.crop_box

    # 1. Crop Region & Clamp
    img_w, img_h = aligned_image.size
    x1 = max(0, box.x)
    y1 = max(0, box.y)
    x2 = min(img_w, box.x + box.w)
    y2 = min(img_h, box.y + box.h)

    crop = aligned_image.crop((x1, y1, x2, y2))

    crop_path = None
    if runtime.debug_dir:
        runtime.debug_dir.mkdir(parents=True, exist_ok=True)
        crop_path = runtime.debug_dir / "q14_vlm_crop.png"
        crop.save(crop_path)
        crop_path_str = str(crop_path)
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.close()
        crop_path = Path(tmp.name)
        crop.save(crop_path)
        crop_path_str = None

    option_mapping_text = "\n".join(
        [f"- option {k}: {v}" for k, v in profile.option_mapping.items()]
    )
    options_schema_text = ",\n".join(
        [
            f'    "{k}": {{"state": "MARKED|BLANK|UNCERTAIN", "reason": "short reason"}}'
            for k in profile.option_order
        ]
    )

    prompt = Q14_PROMPT.format(
        option_mapping_text=option_mapping_text, options_schema_text=options_schema_text
    )

    if not runtime.client:
        options = []
        for opt in profile.option_order:
            options.append(
                VlmOptionEvidence(
                    option_id=opt,
                    decision="NEED_REVIEW",
                    raw_state=None,
                    reason=None,
                    parse_error="Q14 VLM client unavailable",
                )
            )

        if not runtime.debug_dir and crop_path and crop_path.exists():
            crop_path.unlink(missing_ok=True)

        return VlmQuestionEvidence(
            question_id=profile.question_id,
            provider="unknown",
            model=runtime.model,
            prompt_version=profile.prompt_version,
            crop_box={"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            crop_path=crop_path_str,
            raw_response_path=None,
            provider_error="Q14 VLM client unavailable",
            latency_ms=None,
            options=tuple(options),
        )

    req = VlmRequest(model=runtime.model, image_path=crop_path, prompt=prompt, temperature=0.0)

    start_time = time.time()
    provider_error = None
    raw_response_path_str = None
    parsed_dict = {}

    try:
        resp = runtime.client.classify_q14(req)
        if runtime.debug_dir:
            raw_path = runtime.debug_dir / "q14_vlm_raw.txt"
            raw_path.write_text(resp.raw_text, encoding="utf-8")
            raw_response_path_str = str(raw_path)

        parsed_dict = parse_q14_vlm_response(resp.raw_text)
    except Exception as e:
        provider_error = str(e)

    latency_ms = (time.time() - start_time) * 1000

    options = []
    for opt in profile.option_order:
        if provider_error:
            decision = "NEED_REVIEW"
            raw_state = None
            reason = None
            parse_error = provider_error
        else:
            opt_data = parsed_dict.get(opt, {})
            decision = opt_data.get("vlm_state", "NEED_REVIEW")
            raw_state = opt_data.get("raw_vlm_state")
            reason = opt_data.get("reason")
            parse_error = opt_data.get("parse_error")

        options.append(
            VlmOptionEvidence(
                option_id=opt,
                decision=decision,
                raw_state=raw_state,
                reason=reason,
                parse_error=parse_error,
            )
        )

    if not runtime.debug_dir and crop_path and crop_path.exists():
        crop_path.unlink(missing_ok=True)

    return VlmQuestionEvidence(
        question_id=profile.question_id,
        provider="gemini",
        model=runtime.model,
        prompt_version=profile.prompt_version,
        crop_box={"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
        crop_path=crop_path_str,
        raw_response_path=raw_response_path_str,
        provider_error=provider_error,
        latency_ms=latency_ms,
        options=tuple(options),
    )

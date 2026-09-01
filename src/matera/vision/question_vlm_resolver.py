import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from PIL import Image

from matera.core.layout import RoiDef
from matera.vision.evidence.models import VlmOptionEvidence, VlmQuestionEvidence
from matera.vlm.models import VlmRequest, VlmResponse
from matera.vlm.parser import parse_vlm_option_response

QUESTION_PROMPT = """The image shows one multiple-choice question from a scanned worksheet.

Options may be arranged vertically.
The student may mark one or multiple options.
A valid mark can be:
- a circle around one option
- a large enclosure around multiple options
- a clear check/V mark indicating the option
- a clear X mark indicating the option

Ignore:
- printed option text
- printed punctuation
- paper shadows
- scan noise
- faint background artifacts
- unrelated strokes not clearly selecting an option

For each option, decide:
- MARKED: a deliberate mark selecting the option is visible.
- BLANK: no deliberate student mark selecting the option is visible.
- UNCERTAIN: the image is too ambiguous to decide safely.

Return JSON only, exactly in this schema:
{{
  "question_id": "{question_id}",
  "options": {{
{options_schema_text}
  }}
}}
"""


class VlmClient(Protocol):
    def generate_json(self, request: VlmRequest) -> VlmResponse: ...


@dataclass(frozen=True)
class QuestionVlmRuntime:
    client: VlmClient | None
    model: str
    timeout_s: float = 60.0
    debug_dir: Path | None = None


def resolve_question_with_vlm(
    *,
    aligned_image: Image.Image,
    question_id: str,
    rois: Sequence[RoiDef],
    option_ids: Sequence[str],
    runtime: QuestionVlmRuntime,
) -> VlmQuestionEvidence:
    """Resolve a single multiple-choice question using a VLM."""
    # 1. Compute Crop Region & Clamp
    if not rois:
        raise ValueError("Cannot crop for VLM without ROIs")

    union_x1 = min(roi.bbox.x for roi in rois)
    union_y1 = min(roi.bbox.y for roi in rois)
    union_x2 = max(roi.bbox.x + roi.bbox.w for roi in rois)
    union_y2 = max(roi.bbox.y + roi.bbox.h for roi in rois)

    w_list = sorted([roi.bbox.w for roi in rois])
    h_list = sorted([roi.bbox.h for roi in rois])
    median_roi_w = w_list[len(w_list) // 2]
    median_roi_h = h_list[len(h_list) // 2]

    pad_left = max(80, int(median_roi_w * 3.0))
    pad_right = max(80, int(median_roi_w * 3.0))
    pad_top = max(40, int(median_roi_h * 2.0))
    pad_bottom = max(40, int(median_roi_h * 2.0))

    img_w, img_h = aligned_image.size
    x1 = max(0, union_x1 - pad_left)
    y1 = max(0, union_y1 - pad_top)
    x2 = min(img_w, union_x2 + pad_right)
    y2 = min(img_h, union_y2 + pad_bottom)

    crop = aligned_image.crop((x1, y1, x2, y2))

    crop_path = None
    if runtime.debug_dir:
        runtime.debug_dir.mkdir(parents=True, exist_ok=True)
        crop_path = runtime.debug_dir / f"{question_id}_vlm_crop.png"
        crop.save(crop_path)
        crop_path_str = str(crop_path)
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.close()
        crop_path = Path(tmp.name)
        crop.save(crop_path)
        crop_path_str = None

    options_schema_text = ",\n".join(
        [
            f'    "{k}": {{"state": "MARKED|BLANK|UNCERTAIN", "reason": "short reason"}}'
            for k in option_ids
        ]
    )

    prompt = QUESTION_PROMPT.format(
        question_id=question_id, options_schema_text=options_schema_text
    )

    if not runtime.client:
        options = []
        for opt in option_ids:
            options.append(
                VlmOptionEvidence(
                    option_id=opt,
                    decision="NEED_REVIEW",
                    raw_state=None,
                    reason=None,
                    parse_error="VLM client unavailable",
                )
            )

        if not runtime.debug_dir and crop_path and crop_path.exists():
            crop_path.unlink(missing_ok=True)

        return VlmQuestionEvidence(
            question_id=question_id,
            provider="unknown",
            model=runtime.model,
            prompt_version="v1",
            crop_box={"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            crop_path=crop_path_str,
            raw_response_path=None,
            provider_error="VLM client unavailable",
            latency_ms=None,
            options=tuple(options),
        )

    req = VlmRequest(model=runtime.model, image_path=crop_path, prompt=prompt, temperature=0.0)

    start_time = time.time()
    provider_error = None
    raw_response_path_str = None
    parsed_dict = {}

    try:
        resp = runtime.client.generate_json(req)
        if runtime.debug_dir:
            raw_path = runtime.debug_dir / f"{question_id}_vlm_raw.txt"
            raw_path.write_text(resp.raw_text, encoding="utf-8")
            raw_response_path_str = str(raw_path)

        parsed_dict = parse_vlm_option_response(
            resp.raw_text, question_id=question_id, option_ids=option_ids
        )
    except Exception as e:
        provider_error = str(e)

    latency_ms = (time.time() - start_time) * 1000

    options = []
    for opt in option_ids:
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
        question_id=question_id,
        provider="gemini",
        model=runtime.model,
        prompt_version="v1",
        crop_box={"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
        crop_path=crop_path_str,
        raw_response_path=raw_response_path_str,
        provider_error=provider_error,
        latency_ms=latency_ms,
        options=tuple(options),
    )

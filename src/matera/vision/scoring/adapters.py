from typing import TYPE_CHECKING

from matera.vision.evidence.models import PageMarkEvidence

if TYPE_CHECKING:
    from matera.vision.contracts import AlignedPage, MarkScore


def evidence_to_mark_scores(
    page_evidence: PageMarkEvidence,
    aligned_page: "AlignedPage",
    debug_dir: str | None = None,
) -> list["MarkScore"]:
    from pathlib import Path

    from matera.vision.contracts import MarkScore

    debug_path = Path(debug_dir) if debug_dir else None
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)

    scores = []
    for question in page_evidence.questions:
        for opt in question.option_evidence:
            score_val = (
                1.0
                if opt.legacy_prediction == "MARKED"
                else 0.0
                if opt.legacy_prediction == "BLANK"
                else 0.5
            )

            image_crop = None
            evidence_path = None

            # Spec requires behavior-preserving crop (exact unpadded scaled_bbox)
            roi_ev = next((r for r in question.rois if r.option_id == opt.option_id), None)
            if roi_ev:
                scaled = roi_ev.scaled_bbox
                cx1, cy1 = scaled["x"], scaled["y"]
                cx2, cy2 = cx1 + scaled["w"], cy1 + scaled["h"]
                image_crop = aligned_page.image.crop((cx1, cy1, cx2, cy2))

            if debug_path and image_crop:
                evidence_file = (
                    debug_path
                    / f"page_{page_evidence.page_number}_{opt.question_id}_{opt.option_id}.png"
                )
                image_crop.save(evidence_file)
                evidence_path = evidence_file

            scores.append(
                MarkScore(
                    question_id=opt.question_id,
                    option_id=opt.option_id,
                    score=score_val,
                    strategy=opt.strategy,
                    method=opt.legacy_method,
                    image_crop=image_crop,
                    evidence_path=evidence_path,
                )
            )

    return scores

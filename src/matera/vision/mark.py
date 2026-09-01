from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

    from matera.core.layout import PageLayout
    from matera.core.profile import FormProfile
    from matera.vision.contracts import AlignedPage, MarkScore

from matera.vision.detectors.global_topology import run_v11_global_topology
from matera.vision.detectors.hsv import process_roi_hsv

__all__ = ["extract_mark_scores", "run_v11_global_topology", "process_roi_hsv"]


def extract_mark_scores(
    aligned_page: AlignedPage,
    profile: FormProfile,
    layout: PageLayout,
    reference_image: Image.Image,
    *,
    debug_dir: str | None = None,
    debug_full_mask: np.ndarray | None = None,
    q14_vlm_runtime: Any | None = None,
    question_vlm_runtime: Any | None = None,
) -> list[MarkScore]:
    """
    Primary stable production entry point for scoring all marks on an aligned page.
    """
    from matera.vision.scoring import evidence_to_mark_scores, extract_mark_evidence

    evidence = extract_mark_evidence(
        aligned_page,
        profile,
        layout,
        reference_image,
        debug_dir=debug_dir,
        debug_full_mask=debug_full_mask,
        q14_vlm_runtime=q14_vlm_runtime,
        question_vlm_runtime=question_vlm_runtime,
    )

    return evidence_to_mark_scores(evidence, aligned_page, debug_dir=debug_dir)

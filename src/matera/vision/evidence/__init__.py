from matera.vision.detectors.global_topology import compute_global_topology_evidence
from matera.vision.detectors.local import compute_local_option_evidence
from matera.vision.scoring.adapters import evidence_to_mark_scores
from matera.vision.scoring.extraction import extract_mark_evidence

from .models import (
    AlignmentEvidence,
    CheckboxComponentEvidence,
    CheckboxDecision,
    CheckboxStrokeEvidence,
    GlobalClusterEvidence,
    GlobalContourEvidence,
    GlobalTopologyEvidence,
    HsvFallbackEvidence,
    LocalOptionEvidence,
    MarkThresholdEvidence,
    OptionCenterEvidence,
    OptionMarkEvidence,
    PageMarkEvidence,
    QuestionMarkEvidence,
    ReferenceEvidence,
    RoiEvidence,
)
from .serialization import evidence_to_json_dict

extract_page_evidence = extract_mark_evidence

__all__ = [
    "AlignmentEvidence",
    "CheckboxComponentEvidence",
    "CheckboxDecision",
    "CheckboxStrokeEvidence",
    "GlobalClusterEvidence",
    "GlobalContourEvidence",
    "GlobalTopologyEvidence",
    "HsvFallbackEvidence",
    "LocalOptionEvidence",
    "MarkThresholdEvidence",
    "OptionCenterEvidence",
    "OptionMarkEvidence",
    "PageMarkEvidence",
    "QuestionMarkEvidence",
    "ReferenceEvidence",
    "RoiEvidence",
    "compute_global_topology_evidence",
    "compute_local_option_evidence",
    "evidence_to_json_dict",
    "evidence_to_mark_scores",
    "extract_mark_evidence",
    "extract_page_evidence",
]

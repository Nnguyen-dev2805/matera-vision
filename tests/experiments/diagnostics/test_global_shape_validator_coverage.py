from matera.vision.diagnostics.global_shape_validator import (
    GlobalShapeValidatorConfig,
    diagnose_global_shape_candidates,
)


# Minimal mock classes
class MockRoi:
    def __init__(self, option_id: str, x: int, y: int, w: int, h: int):
        self.option_id = option_id
        self.bbox = type("Bbox", (), {"x": x, "y": y, "w": w, "h": h})


class MockLocalEvidence:
    def __init__(self, radial_degrees_covered: float):
        self.radial_degrees_covered = radial_degrees_covered


class MockCluster:
    def __init__(self, cluster_id: str, inside_options: list[str]):
        self.cluster_id = cluster_id
        self.inside_options = inside_options
        # Simple square hull around (0,0)-(10,10)
        self.hull_points = [(0, 0), (10, 0), (10, 10), (0, 10)]


class MockGlobalTopology:
    def __init__(self, clusters: list[MockCluster]):
        self.clusters = clusters
        self.global_marked = frozenset()
        self.crop = {"x1": 0, "y1": 0}


def test_classify_and_diagnose():
    # Setup a cluster with two options
    cluster = MockCluster("c1", ["opt1", "opt2"])
    rois = [
        MockRoi("opt1", 1, 1, 4, 4),
        MockRoi("opt2", 5, 5, 4, 4),
    ]
    # Local evidence: one strong, one weak
    local_by_option = {
        "opt1": MockLocalEvidence(radial_degrees_covered=400.0),  # strong
        "opt2": MockLocalEvidence(radial_degrees_covered=30.0),  # weak (> blank 0)
    }
    topology = MockGlobalTopology([cluster])
    cfg = GlobalShapeValidatorConfig()
    diagnostics = diagnose_global_shape_candidates(
        question_id="Q1",
        rois=rois,
        global_topology=topology,
        local_evidence_by_option=local_by_option,
        config=cfg,
    )
    # Expect one cluster diagnostic
    assert len(diagnostics.clusters) == 1
    diag = diagnostics.clusters[0]
    # Group score should be computed from boundary coverage (simple case gives >0)
    assert 0 <= diag.group_enclosure_score <= 1
    # Local score should be >0 because we have a strong option
    assert diag.local_circle_cluster_score > 0
    # Action should be KEEP_GLOBAL or CALL_VLM_LATER based on risk threshold
    assert diag.recommended_action in {"KEEP_GLOBAL", "CALL_VLM_LATER"}
    # Option diagnostics should be created for each option
    assert len(diagnostics.options) == 2
    opt_ids = {opt.option_id for opt in diagnostics.options}
    assert opt_ids == {"opt1", "opt2"}

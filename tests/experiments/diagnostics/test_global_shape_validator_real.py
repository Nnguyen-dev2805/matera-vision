import numpy as np

from matera.vision.diagnostics.global_shape_validator import (
    GlobalShapeValidatorConfig,
    diagnose_global_shape_candidates,
)


class MockRoi:
    def __init__(self, option_id: str, x: float, y: float, w: int = 40, h: int = 34):
        self.option_id = option_id
        self.bbox = type(
            "Bbox",
            (),
            {"x": int(x - w / 2), "y": int(y - h / 2), "w": w, "h": h},
        )


class MockLocalEvidence:
    def __init__(self, radial_degrees_covered: float):
        self.radial_degrees_covered = radial_degrees_covered
        self.radial_ink_pixels = 0


class MockCluster:
    def __init__(
        self, cluster_id: str, inside_options: list[str], hull_points: list[tuple[int, int]]
    ):
        self.cluster_id = cluster_id
        self.inside_options = inside_options
        self.hull_points = hull_points


class MockGlobalTopology:
    def __init__(
        self,
        clusters: list[MockCluster],
        crop: dict[str, int],
        global_marked: frozenset[str],
    ):
        self.clusters = clusters
        self.global_marked = global_marked
        self.crop = crop


def _mask_with_vertical_span(
    *,
    shape: tuple[int, int],
    x_range: tuple[int, int],
    y_range: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x1, x2 = x_range
    y1, y2 = y_range
    mask[y1:y2, x1:x2] = 255
    return mask


def test_q3_page2_merged_local_circles_call_vlm():
    crop = {"x1": 169, "y1": 1081, "x2": 319, "y2": 1373}
    centers = {
        "a": (49.0, 46.0),
        "b": (53.0, 98.0),
        "c": (56.5, 148.5),
        "d": (54.0, 198.5),
    }
    rois = [MockRoi(opt, crop["x1"] + x, crop["y1"] + y) for opt, (x, y) in centers.items()]
    hull_points = [(49, 28), (60, 32), (66, 41), (69, 73), (66, 212), (24, 211), (23, 38)]
    cluster = MockCluster("cluster_11", ["a", "b", "c", "d"], hull_points)

    # Ink lives mostly on one side of the option group, like several local
    # circles merged into one thin hull. There is no convincing opposite side.
    mask_raw = _mask_with_vertical_span(shape=(292, 150), x_range=(23, 38), y_range=(28, 212))
    local_by_option = {
        "a": MockLocalEvidence(360.0),
        "b": MockLocalEvidence(360.0),
        "c": MockLocalEvidence(305.0),
        "d": MockLocalEvidence(75.0),
    }

    topology = MockGlobalTopology([cluster], crop, frozenset({"a", "b", "c", "d"}))
    cfg = GlobalShapeValidatorConfig(strong_local_degrees=180.0, blank_local_degrees=90.0)
    diagnostics = diagnose_global_shape_candidates(
        question_id="Q3",
        rois=rois,
        global_topology=topology,
        local_evidence_by_option=local_by_option,
        mask_raw=mask_raw,
        config=cfg,
    )

    diag = diagnostics.clusters[0]
    assert diag.should_call_vlm_later is True
    assert diag.recommended_action == "CALL_VLM_LATER"
    assert diag.shape_class in {"LOCAL_CIRCLE_CLUSTER", "UNCERTAIN"}
    d_diag = next(o for o in diagnostics.options if o.option_id == "d")
    assert d_diag.recommended_action == "CALL_VLM_LATER"


def test_q4_page1_large_group_enclosure_keeps_global():
    crop = {"x1": 143, "y1": 1432, "x2": 277, "y2": 1785}
    centers = {
        "a": (71.0, 45.5),
        "b": (73.0, 99.5),
        "c": (73.0, 153.5),
        "d": (76.5, 204.0),
        "e": (77.5, 251.5),
        "f": (77.5, 302.5),
    }
    rois = [MockRoi(opt, crop["x1"] + x, crop["y1"] + y) for opt, (x, y) in centers.items()]
    hull_points = [(55, 21), (76, 26), (101, 44), (117, 107), (115, 337), (7, 334), (6, 68)]
    cluster = MockCluster("cluster_2", ["a", "b", "c", "d", "e", "f"], hull_points)

    mask_raw = np.zeros((353, 134), dtype=np.uint8)
    mask_raw[38:337, 6:20] = 255
    mask_raw[50:337, 108:118] = 255
    mask_raw[322:337, 16:118] = 255

    local_by_option = {
        "a": MockLocalEvidence(220.0),
        "b": MockLocalEvidence(25.0),
        "c": MockLocalEvidence(155.0),
        "d": MockLocalEvidence(360.0),
        "e": MockLocalEvidence(55.0),
        "f": MockLocalEvidence(85.0),
    }

    topology = MockGlobalTopology([cluster], crop, frozenset({"a", "b", "c", "d", "e", "f"}))
    cfg = GlobalShapeValidatorConfig(strong_local_degrees=180.0, blank_local_degrees=90.0)
    diagnostics = diagnose_global_shape_candidates(
        question_id="Q4",
        rois=rois,
        global_topology=topology,
        local_evidence_by_option=local_by_option,
        mask_raw=mask_raw,
        config=cfg,
    )

    diag = diagnostics.clusters[0]
    assert diag.should_call_vlm_later is False
    assert diag.shape_class == "GROUP_ENCLOSURE"
    assert diag.recommended_action == "KEEP_GLOBAL"
    e_diag = next(o for o in diagnostics.options if o.option_id == "e")
    assert e_diag.recommended_action == "KEEP_GLOBAL"

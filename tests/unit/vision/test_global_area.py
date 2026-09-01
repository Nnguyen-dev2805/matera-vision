import cv2
import numpy as np

import matera.vision.evidence  # noqa: F401, I001
from matera.vision.detectors.global_area import compute_cluster_metrics


def test_compute_cluster_metrics_filled_contour():
    # Create a dummy mask where ink is just a thin circle outline
    mask_raw = np.zeros((100, 100), dtype=np.uint8)
    # Draw a thin circle with radius 30 (perimeter only)
    cv2.circle(mask_raw, (50, 50), 30, 255, 2)

    # We find contours
    cnts, _ = cv2.findContours(mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(cnts) == 1

    cnt_list = [("cnt_0", cnts[0])]

    total_area, hull_area, solidity, hull = compute_cluster_metrics(mask_raw, cnt_list)

    # The geometric area of a filled circle of radius 30 is ~ 2827 (pi*r^2)
    # The pixels of a thin perimeter of radius 30, thickness 2, is roughly 2*pi*30 * 2 ~ 377
    # So total_area should be around 300-500, but geometric area would be ~2827.
    assert total_area < 1000, f"Expected total_area to just be perimeter pixels, got {total_area}"
    assert hull_area > 2000, f"Expected hull_area to encompass the circle, got {hull_area}"

    # solidity should be total_area / hull_area, so very low for a thin enclosure
    assert 0.05 < solidity < 0.3, f"Expected low solidity for a thin ring, got {solidity}"


def test_compute_cluster_metrics_solid_blob():
    # Create a solid filled circle
    mask_raw = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask_raw, (50, 50), 30, 255, -1)

    cnts, _ = cv2.findContours(mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(cnts) == 1

    cnt_list = [("cnt_0", cnts[0])]

    total_area, hull_area, solidity, hull = compute_cluster_metrics(mask_raw, cnt_list)

    # Both should be similar, around ~2827
    assert 2500 < total_area < 3200
    assert 2500 < hull_area < 3200

    # Solidity should be close to 1
    assert solidity > 0.9, f"Expected high solidity for a filled blob, got {solidity}"

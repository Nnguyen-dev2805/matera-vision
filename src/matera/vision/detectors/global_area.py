import cv2
import numpy as np


def compute_cluster_metrics(
    mask_raw: np.ndarray, cnt_list: list
) -> tuple[float, float, float, np.ndarray]:
    """
    Computes total_area (counted pixels), hull_area, solidity, and hull for a cluster of contours.

    Args:
        mask_raw: The binary mask containing the raw differences/ink pixels.
        cnt_list: A list of tuples where the second element is the contour (e.g. `(id, contour)`).

    Returns:
        total_area: The number of non-zero pixels inside the filled contours of the cluster.
        hull_area: The geometric area of the convex hull enclosing all contours in the cluster.
        solidity: total_area / hull_area (capped/fallback as appropriate).
        hull: The convex hull contour.
    """
    combined_points = np.vstack([c[1] for c in cnt_list])

    total_area = 0.0
    for _, c in cnt_list:
        mask = np.zeros_like(mask_raw)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        total_area += float(cv2.countNonZero(cv2.bitwise_and(mask_raw, mask)))

    hull = cv2.convexHull(combined_points)
    hull_area = float(cv2.contourArea(hull))

    solidity = total_area / hull_area if hull_area > 0 else 1.0

    return total_area, hull_area, solidity, hull

import cv2
import numpy as np
from PIL import Image

from matera.vision.evidence.models import (
    GlobalClusterEvidence,
    GlobalContourEvidence,
    GlobalTopologyEvidence,
    OptionCenterEvidence,
)


def compute_global_topology_evidence(
    aligned_image_rgb: Image.Image,
    median_ref_bgr: np.ndarray,
    rois: list,
) -> GlobalTopologyEvidence:
    from collections import defaultdict

    from matera.vision.adaptive_crop import compute_adaptive_global_crop
    from matera.vision.global_area import compute_cluster_metrics
    from matera.vision.mark_global import (
        UnionFind,
        get_horizontal_extremes,
        min_contour_distance,
    )
    from matera.vision.mark_thresholds import (
        CLOSE_ITERATIONS,
        CLOSE_KERNEL_SIZE,
        DIFF_THRESHOLD,
        EXTREMES_REJECT_THRESHOLD,
        GAUSS_KERNEL,
        MERGE_THRESHOLD,
    )

    question_id = rois[0].question_id if rois else ""

    if len(rois) <= 1:
        return GlobalTopologyEvidence(
            question_id=question_id,
            ran=False,
            skip_reason="<= 1 rois",
            crop={"x1": 0, "y1": 0, "x2": 0, "y2": 0},
            crop_expansion=None,
            option_centers=(),
            contours=(),
            clusters=(),
            global_marked=frozenset(),
        )

    crop_result = compute_adaptive_global_crop(aligned_image_rgb, median_ref_bgr, rois)
    crop = crop_result.crop
    crop_x1, crop_y1, crop_x2, crop_y2 = crop["x1"], crop["y1"], crop["x2"], crop["y2"]

    crop_img = aligned_image_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    target_bgr = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
    ref_crop = median_ref_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(ref_gray, target_gray)
    blurred = cv2.GaussianBlur(diff, GAUSS_KERNEL, 0)
    _, mask_raw = cv2.threshold(blurred, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    h, w = mask_raw.shape
    mask_raw[:15, :] = 0
    mask_raw[h - 15 :, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERATIONS)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    global_marked = set()
    option_centers = {}
    option_centers_ev = []
    for r in rois:
        cx = (r.bbox.x + r.bbox.w / 2.0) - crop_x1
        cy = (r.bbox.y + r.bbox.h / 2.0) - crop_y1
        option_centers[r.option_id] = (cx, cy)
        option_centers_ev.append(OptionCenterEvidence(r.option_id, int(cx), int(cy)))

    valid_cnts = []
    contours_ev = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x, y, w_b, h_b = cv2.boundingRect(cnt)
        bbox = {"x": x, "y": y, "w": w_b, "h": h_b}
        if area <= 30:
            contours_ev.append(
                GlobalContourEvidence(
                    f"cnt_{i}",
                    float(area),
                    bbox,
                    "REJECTED",
                    "area <= 30",
                )
            )
            continue
        if (h_b > 100 and w_b < 25) or (w_b > 100 and h_b < 25):
            contours_ev.append(
                GlobalContourEvidence(
                    f"cnt_{i}",
                    float(area),
                    bbox,
                    "REJECTED",
                    "aspect ratio",
                )
            )
            continue

        valid_cnts.append((f"cnt_{i}", cnt))
        contours_ev.append(
            GlobalContourEvidence(
                f"cnt_{i}",
                float(area),
                bbox,
                "ACCEPTED",
                None,
            )
        )

    n = len(valid_cnts)
    if n == 0:
        return GlobalTopologyEvidence(
            question_id=question_id,
            ran=True,
            skip_reason="no valid contours",
            crop={"x1": crop_x1, "y1": crop_y1, "x2": crop_x2, "y2": crop_y2},
            crop_expansion={
                "pads": crop_result.pads,
                "iterations": crop_result.iterations,
                "expanded": crop_result.expanded,
                "stop_reason": crop_result.stop_reason,
            }
            if hasattr(crop_result, "expanded")
            else None,
            option_centers=tuple(option_centers_ev),
            contours=tuple(contours_ev),
            clusters=(),
            global_marked=frozenset(),
        )

    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            dist = min_contour_distance(valid_cnts[i][1], valid_cnts[j][1])
            if dist <= MERGE_THRESHOLD:
                l1, r1 = get_horizontal_extremes(valid_cnts[i][1])
                l2, r2 = get_horizontal_extremes(valid_cnts[j][1])
                dist_left = np.linalg.norm(l1 - l2)
                dist_right = np.linalg.norm(r1 - r2)

                if dist_left > EXTREMES_REJECT_THRESHOLD and dist_right > EXTREMES_REJECT_THRESHOLD:
                    pass
                else:
                    uf.union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(valid_cnts[i])

    clusters_ev = []
    for root, cnt_list in clusters.items():
        total_area, hull_area, solidity, hull = compute_cluster_metrics(mask_raw, cnt_list)

        # Calculate cluster bbox and points
        cx_min, cy_min, cx_max, cy_max = float("inf"), float("inf"), 0.0, 0.0
        for pt in hull:
            px, py = pt[0]
            cx_min = min(cx_min, px)
            cy_min = min(cy_min, py)
            cx_max = max(cx_max, px)
            cy_max = max(cy_max, py)
        cluster_bbox = {
            "x": int(cx_min),
            "y": int(cy_min),
            "w": int(cx_max - cx_min),
            "h": int(cy_max - cy_min),
        }
        hull_pts = tuple((int(pt[0][0]), int(pt[0][1])) for pt in hull)

        inside_options = []
        option_dists = {}
        solidity = total_area / float(hull_area) if hull_area > 0 else 1.0

        if hull_area > 1000 and solidity < 0.4:
            for opt_id, (cx, cy) in option_centers.items():
                dist = cv2.pointPolygonTest(hull, (cx, cy), True)
                option_dists[opt_id] = float(dist)
                if dist >= 0:
                    global_marked.add(opt_id)
                    inside_options.append(opt_id)

        clusters_ev.append(
            GlobalClusterEvidence(
                cluster_id=f"cluster_{root}",
                contour_ids=tuple(c[0] for c in cnt_list),
                total_area=float(total_area),
                hull_area=float(hull_area),
                solidity=float(solidity),
                bbox=cluster_bbox,
                hull_points=hull_pts,
                inside_options=tuple(inside_options),
                option_center_distances=option_dists,
            )
        )

    return GlobalTopologyEvidence(
        question_id=question_id,
        ran=True,
        skip_reason=None,
        crop={"x1": crop_x1, "y1": crop_y1, "x2": crop_x2, "y2": crop_y2},
        crop_expansion={
            "pads": crop_result.pads,
            "iterations": crop_result.iterations,
            "expanded": crop_result.expanded,
            "stop_reason": crop_result.stop_reason,
        }
        if hasattr(crop_result, "expanded")
        else None,
        option_centers=tuple(option_centers_ev),
        contours=tuple(contours_ev),
        clusters=tuple(clusters_ev),
        global_marked=frozenset(global_marked),
    )

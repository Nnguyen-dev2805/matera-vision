from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from matera.vision.contracts import MarkScore

if TYPE_CHECKING:
    from matera.classifier.model import AmbiguityClassifier
    from matera.core.layout import PageLayout
    from matera.core.profile import FormProfile
    from matera.vision.contracts import AlignedPage

import math

from scipy.spatial.distance import cdist

# --- V17 Geometric Configs ---
NUM_BINS = 72
DEGREES_PER_BIN = 360 / NUM_BINS
MIN_INK_PER_BIN = 2
MARKED_THRESHOLD_DEG = 180
BLANK_THRESHOLD_DEG = 90
LOCAL_PAD = 15
OUTER_RADIUS = 32
GLOBAL_PAD = 20
MERGE_THRESHOLD = 15.0
EXTREMES_REJECT_THRESHOLD = 50.0
CHECKBOX_INNER_MARGIN = 5
HSV_SATURATION_THRESHOLD = 40

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
    def find(self, i: int) -> int:
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def min_contour_distance(cnt1: np.ndarray, cnt2: np.ndarray) -> float:
    pts1 = cnt1.reshape(-1, 2)
    pts2 = cnt2.reshape(-1, 2)
    dists = cdist(pts1, pts2, metric='euclidean')
    return float(np.min(dists))

def get_horizontal_extremes(cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = cnt.reshape(-1, 2)
    left_pt = pts[np.argmin(pts[:, 0])]
    right_pt = pts[np.argmax(pts[:, 0])]
    return left_pt, right_pt

def get_text_bounding_box(ref_crop_gray: np.ndarray) -> tuple[int, int, int, int]:
    _, thresh = cv2.threshold(ref_crop_gray, 200, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    h, w = ref_crop_gray.shape
    cx, cy = w / 2.0, h / 2.0
    min_dist_to_center = float('inf')
    best_component_stats = None
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        centroid_x, centroid_y = centroids[i]
        
        if x <= 1 or y <= 1 or (x + bw) >= w - 1 or (y + bh) >= h - 1: continue
        aspect_ratio = max(bw / float(bh), bh / float(bw))
        if aspect_ratio > 4.0: continue
            
        dist_to_center = math.sqrt((centroid_x - cx)**2 + (centroid_y - cy)**2)
        if dist_to_center < min_dist_to_center:
            min_dist_to_center = dist_to_center
            best_component_stats = (x, y, bw, bh)
            
    if best_component_stats is not None:
        return best_component_stats
    else:
        return (int(cx - 7), int(cy - 7), 14, 14)

def get_local_roi_crops(aligned_image_rgb: Image.Image, median_ref_bgr: np.ndarray, bbox, pad: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], np.ndarray]:
    crop_x1 = max(0, bbox.x - pad)
    crop_y1 = max(0, bbox.y - pad)
    crop_x2 = min(aligned_image_rgb.width, bbox.x + bbox.w + pad)
    crop_y2 = min(aligned_image_rgb.height, bbox.y + bbox.h + pad)
    
    crop_img = aligned_image_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    target_bgr = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
    ref_crop = median_ref_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(ref_gray, target_gray)
    blurred = cv2.GaussianBlur(diff, (3, 3), 0)
    _, mask_raw = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    
    return target_bgr, mask_raw, (crop_x1, crop_y1, crop_x2, crop_y2), ref_gray

def run_v11_global_topology(aligned_image_rgb: Image.Image, median_ref_bgr: np.ndarray, rois: list, full_mask_bgr: np.ndarray) -> set[str]:
    from collections import defaultdict
    if len(rois) <= 1:
        return set()
        
    min_x = min(r.bbox.x for r in rois)
    min_y = min(r.bbox.y for r in rois)
    max_x = max(r.bbox.x + r.bbox.w for r in rois)
    max_y = max(r.bbox.y + r.bbox.h for r in rois)
    
    crop_x1 = max(0, min_x - GLOBAL_PAD)
    crop_y1 = max(0, min_y - GLOBAL_PAD)
    crop_x2 = min(aligned_image_rgb.width, max_x + GLOBAL_PAD)
    crop_y2 = min(aligned_image_rgb.height, max_y + GLOBAL_PAD)
    
    crop_img = aligned_image_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    target_bgr = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
    ref_crop = median_ref_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(ref_gray, target_gray)
    blurred = cv2.GaussianBlur(diff, (3, 3), 0)
    _, mask_raw = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    
    h, w = mask_raw.shape
    mask_raw[:15, :] = 0
    mask_raw[h-15:, :] = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    global_marked = set()
    option_centers = {}
    for r in rois:
        cx = (r.bbox.x + r.bbox.w / 2.0) - crop_x1
        cy = (r.bbox.y + r.bbox.h / 2.0) - crop_y1
        option_centers[r.option_id] = (cx, cy)
        
    valid_cnts = []
    for cnt in contours:
        if cv2.contourArea(cnt) <= 30: continue
        x, y, w, h = cv2.boundingRect(cnt)
        if (h > 100 and w < 25) or (w > 100 and h < 25): continue
        valid_cnts.append(cnt)
            
    n = len(valid_cnts)
    if n == 0: return global_marked
        
    uf = UnionFind(n)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = min_contour_distance(valid_cnts[i], valid_cnts[j])
            if dist <= MERGE_THRESHOLD:
                l1, r1 = get_horizontal_extremes(valid_cnts[i])
                l2, r2 = get_horizontal_extremes(valid_cnts[j])
                dist_left = np.linalg.norm(l1 - l2)
                dist_right = np.linalg.norm(r1 - r2)
                
                if dist_left > EXTREMES_REJECT_THRESHOLD and dist_right > EXTREMES_REJECT_THRESHOLD:
                    pass
                else:
                    uf.union(i, j)
                    
    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(valid_cnts[i])
        
    for root, cnt_list in clusters.items():
        combined_points = np.vstack(cnt_list)
        total_area = sum([cv2.contourArea(c) for c in cnt_list])
        
        hull = cv2.convexHull(combined_points)
        hull_area = cv2.contourArea(hull)
        
        if hull_area > 1000:
            solidity = total_area / float(hull_area) if hull_area > 0 else 1.0
            if solidity < 0.4:
                hull_offset = hull.copy()
                for pt in hull_offset:
                    pt[0][0] += crop_x1
                    pt[0][1] += crop_y1
                cv2.drawContours(full_mask_bgr, [hull_offset], 0, (0, 255, 255), 2)
                
                for opt_id, (cx, cy) in option_centers.items():
                    if cv2.pointPolygonTest(hull, (cx, cy), False) >= 0:
                        global_marked.add(opt_id)
                        
    return global_marked

def get_local_roi_crops_hsv(aligned_image_rgb: Image.Image, median_ref_bgr: np.ndarray, bbox, pad: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    crop_x1 = max(0, bbox.x - pad)
    crop_y1 = max(0, bbox.y - pad)
    crop_x2 = min(aligned_image_rgb.width, bbox.x + bbox.w + pad)
    crop_y2 = min(aligned_image_rgb.height, bbox.y + bbox.h + pad)
    
    crop_img = aligned_image_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    target_bgr = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
    ref_crop = median_ref_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(ref_gray, target_gray)
    blurred = cv2.GaussianBlur(diff, (3, 3), 0)
    _, mask_diff = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    
    hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask_color = cv2.threshold(saturation, HSV_SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    mask_final = cv2.bitwise_and(mask_diff, mask_color)
    
    return target_bgr, mask_final, (crop_x1, crop_y1, crop_x2, crop_y2)

def process_roi_hsv_ai(aligned_page: "AlignedPage", median_ref_bgr: np.ndarray, roi, method_prefix: str) -> tuple[str, str]:
    from matera.classifier.model import extract_hog_features
    _, mask_hsv, _ = get_local_roi_crops_hsv(aligned_page.image, median_ref_bgr, roi.bbox, 0)
    h, w = mask_hsv.shape
    safe_mask = np.zeros_like(mask_hsv)
    m = CHECKBOX_INNER_MARGIN
    if h > 2*m and w > 2*m:
        safe_mask[m:h-m, m:w-m] = mask_hsv[m:h-m, m:w-m]
        
    ink_pixels = np.sum(safe_mask > 0)
    
    if ink_pixels < 20:
        return "BLANK", f"{method_prefix}_L2"
    elif ink_pixels > 200:
        return "MARKED", f"{method_prefix}_L2"
    else:
        feat = extract_hog_features(safe_mask)
        try:
            clf = _get_classifier()
        except FileNotFoundError:
            # Fallback to simple threshold if model missing
            return "AMBIGUOUS", f"{method_prefix}_L2_NO_AI"
        prob_mark = clf.predict_proba([feat])[0]
        if prob_mark > 0.85:
            return "MARKED", f"{method_prefix}_AI ({prob_mark:.2f})"
        elif prob_mark < 0.15:
            return "BLANK", f"{method_prefix}_AI ({prob_mark:.2f})"
        else:
            return "AMBIGUOUS", f"{method_prefix}_AI ({prob_mark:.2f})"


# --- Global Classifier Instance (Lazy loaded) ---
_clf = None
def _get_classifier() -> "AmbiguityClassifier":
    global _clf
    if _clf is None:
        model_path = Path("models/shape_classifier.pkl")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        from matera.classifier.model import AmbiguityClassifier
        _clf = AmbiguityClassifier.load(str(model_path))
    return _clf

def extract_mark_scores(
    aligned_page: "AlignedPage",
    profile: "FormProfile",
    layout: "PageLayout",
    reference_image: Image.Image,
    debug_dir: str | None = None,
    debug_full_mask: np.ndarray | None = None,
) -> list[MarkScore]:
    import pathlib
    from collections import defaultdict
    
    # 1. Prepare Reference
    median_ref_bgr = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
    orig_bgr = cv2.cvtColor(np.array(aligned_page.image), cv2.COLOR_RGB2BGR)
    
    if debug_full_mask is None:
        debug_full_mask = np.zeros_like(orig_bgr)
    
    strategy_map = {q.question_id: q.mark_strategy for q in profile.questions}
    
    if debug_dir:
        debug_path = pathlib.Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        
    import dataclasses
    scale_x = orig_bgr.shape[1] / layout.width_px
    scale_y = orig_bgr.shape[0] / layout.height_px
    
    scaled_rois = []
    for roi in layout.rois:
        new_bbox = dataclasses.replace(
            roi.bbox,
            x=int(roi.bbox.x * scale_x),
            y=int(roi.bbox.y * scale_y),
            w=int(roi.bbox.w * scale_x),
            h=int(roi.bbox.h * scale_y)
        )
        scaled_rois.append(dataclasses.replace(roi, bbox=new_bbox))
        
    rois_by_q = defaultdict(list)
    for roi in scaled_rois:
        rois_by_q[roi.question_id].append(roi)
        
    # 2. Run Layer 1 (Global Topology)
    global_marked_dict = {}
    for q_id, rois in rois_by_q.items():
        if "Q14" not in q_id:
            global_marked_dict[q_id] = run_v11_global_topology(aligned_page.image, median_ref_bgr, rois, debug_full_mask)
        else:
            global_marked_dict[q_id] = set()
            
    # 3. Process ROIs
    scores = []
    
    for roi in scaled_rois:
        strategy = roi.mark_strategy_override or strategy_map.get(roi.question_id)
        if not strategy:
            raise ValueError(f"Cannot resolve mark strategy for ROI question_id={roi.question_id}")
            
        opt_id = roi.option_id
        bbox = (roi.bbox.x, roi.bbox.y, roi.bbox.x + roi.bbox.w, roi.bbox.y + roi.bbox.h)
        src_crop = aligned_page.image.crop(bbox)
        
        pred = "AMBIGUOUS"
        method = "UNKNOWN"
        
        if "Q14" not in roi.question_id:
            # Q1-Q13 Logic
            if opt_id in global_marked_dict[roi.question_id]:
                pred = "MARKED"
                method = "GLOBAL_HULL"
            else:
                target_bgr, mask_raw, crop_coords, ref_gray = get_local_roi_crops(aligned_page.image, median_ref_bgr, roi.bbox, LOCAL_PAD)
                cx1, cy1, cx2, cy2 = crop_coords
                text_bbox = get_text_bounding_box(ref_gray)
                bx, by, bw, bh = text_bbox
                h, w = mask_raw.shape
                cx, cy = w / 2.0, h / 2.0
                
                bx = max(0, bx - 2)
                by = max(0, by - 2)
                bw = bw + 4
                bh = bh + 4
                
                Y, X = np.ogrid[:h, :w]
                dist_sq = (X - cx)**2 + (Y - cy)**2
                outer_mask = dist_sq > OUTER_RADIUS**2
                
                core_mask = np.zeros((h, w), dtype=bool)
                by_e = min(h, by+bh)
                bx_e = min(w, bx+bw)
                core_mask[by:by_e, bx:bx_e] = True
                
                mask_radial = mask_raw.copy()
                mask_radial[core_mask] = 0
                mask_radial[outer_mask] = 0
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_dilated = cv2.dilate(mask_radial, kernel, iterations=1)
                mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
                
                ys, xs = np.where(mask_closed > 0)
                total_ink = len(xs)
                degrees_covered = 0
                
                if total_ink > 0:
                    dx = xs.astype(float) - cx
                    dy = ys.astype(float) - cy
                    angles = np.arctan2(dy, dx)
                    angles = np.degrees(angles) % 360
                    hist_counts, _ = np.histogram(angles, bins=NUM_BINS, range=(0, 360))
                    active_bins = (hist_counts >= MIN_INK_PER_BIN).astype(int)
                    for i in range(NUM_BINS):
                        left = active_bins[(i - 1) % NUM_BINS]
                        right = active_bins[(i + 1) % NUM_BINS]
                        if active_bins[i] == 0 and left == 1 and right == 1:
                            active_bins[i] = 1
                    active_bin_count = np.sum(active_bins)
                    degrees_covered = active_bin_count * DEGREES_PER_BIN
                    
                if degrees_covered >= MARKED_THRESHOLD_DEG:
                    pred = "MARKED"
                    method = "LOCAL_RADIAL"
                    mask_bgr = cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR)
                elif degrees_covered <= BLANK_THRESHOLD_DEG:
                    pred = "BLANK"
                    method = "LOCAL_RADIAL"
                    mask_bgr = cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR)
                else:
                    pred, method = process_roi_hsv_ai(aligned_page, median_ref_bgr, roi, "FALLBACK")
                    _, mask_hsv, _ = get_local_roi_crops_hsv(aligned_page.image, median_ref_bgr, roi.bbox, 0)
                    mask_bgr = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)
                    mask_bgr[np.where((mask_bgr == [255, 255, 255]).all(axis=2))] = (255, 0, 255) # Magenta for Layer 3
                    
            if debug_full_mask is not None and opt_id not in global_marked_dict.get(roi.question_id, set()):
                try:
                    cx1, cy1 = crop_coords[0], crop_coords[1]
                    cx2, cy2 = crop_coords[2], crop_coords[3]
                    debug_full_mask[cy1:cy2, cx1:cx2] = cv2.addWeighted(debug_full_mask[cy1:cy2, cx1:cx2], 0.5, mask_bgr, 0.5, 0)
                except Exception:
                    pass
        else:
            # Q14 Checkbox Logic
            pred, method = process_roi_hsv_ai(aligned_page, median_ref_bgr, roi, "HSV_AI")
            if debug_full_mask is not None:
                _, mask_hsv, _ = get_local_roi_crops_hsv(aligned_page.image, median_ref_bgr, roi.bbox, 0)
                mask_bgr = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)
                mask_bgr[np.where((mask_bgr == [255, 255, 255]).all(axis=2))] = (255, 0, 255)
                cx1, cy1 = max(0, roi.bbox.x), max(0, roi.bbox.y)
                cx2, cy2 = cx1 + roi.bbox.w, cy1 + roi.bbox.h
                try:
                    debug_full_mask[cy1:cy2, cx1:cx2] = cv2.addWeighted(debug_full_mask[cy1:cy2, cx1:cx2], 0.5, mask_bgr, 0.5, 0)
                except Exception:
                    pass
            
        # 4. Map to MarkScore
        score_val = 1.0 if pred == "MARKED" else 0.0 if pred == "BLANK" else 0.5
        
        evidence_path = None
        if debug_dir:
            evidence_file = debug_path / f"page_{aligned_page.page_number}_{roi.question_id}_{roi.option_id}.png"
            src_crop.save(evidence_file)
            evidence_path = evidence_file
            
        scores.append(
            MarkScore(
                question_id=roi.question_id,
                option_id=roi.option_id,
                score=score_val,
                strategy=strategy,
                method=method,
                image_crop=src_crop,
                evidence_path=evidence_path,
            )
        )
        
    return scores




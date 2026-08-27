import json
import dataclasses
from dataclasses import dataclass
from typing import Any, Literal
from pathlib import Path
import numpy as np

from matera.vision.contracts import MarkScore


@dataclass(frozen=True)
class AlignmentEvidence:
    alignment_score: float
    warp_matrix: tuple[tuple[float, ...], ...]
    image_size: tuple[int, int]
    layout_size: tuple[int, int]
    scale_x: float
    scale_y: float


@dataclass(frozen=True)
class ReferenceEvidence:
    width: int
    height: int
    dpi: int


@dataclass(frozen=True)
class MarkThresholdEvidence:
    marked_threshold_deg: float
    blank_threshold_deg: float
    routing_low_threshold: float | None
    routing_high_threshold: float | None
    local_pad_px: int
    outer_radius_px: int
    diff_threshold: int
    global_pad_px: int


@dataclass(frozen=True)
class RoiEvidence:
    question_id: str
    option_id: str
    bbox: dict[str, int]
    scaled_bbox: dict[str, int]
    strategy: str


@dataclass(frozen=True)
class LocalOptionEvidence:
    crop_coords: dict[str, int]
    diff_ink_pixels: int
    diff_contour_count: int
    diff_contour_areas: tuple[float, ...]
    text_bbox: dict[str, int]
    core_mask_pixels: int
    outer_mask_pixels: int
    radial_mask_before_morph_px: int
    radial_ink_pixels: int
    radial_active_bin_count: int
    radial_degrees_covered: float
    radial_histogram: tuple[int, ...]


@dataclass(frozen=True)
class OptionCenterEvidence:
    option_id: str
    center_x: int
    center_y: int


@dataclass(frozen=True)
class GlobalContourEvidence:
    contour_id: str
    area: float
    bbox: dict[str, int]
    status: Literal["ACCEPTED", "REJECTED"]
    rejection_reason: str | None


@dataclass(frozen=True)
class GlobalClusterEvidence:
    cluster_id: str
    contour_ids: tuple[str, ...]
    total_area: float
    hull_area: float
    solidity: float
    bbox: dict[str, int]
    hull_points: tuple[tuple[int, int], ...]
    inside_options: tuple[str, ...]
    option_center_distances: dict[str, float]


@dataclass(frozen=True)
class GlobalTopologyEvidence:
    question_id: str
    ran: bool
    skip_reason: str | None
    crop: dict[str, int]
    crop_expansion: dict[str, Any] | None
    option_centers: tuple[OptionCenterEvidence, ...]
    contours: tuple[GlobalContourEvidence, ...]
    clusters: tuple[GlobalClusterEvidence, ...]
    global_marked: frozenset[str]


@dataclass(frozen=True)
class HsvFallbackEvidence:
    method_prefix: str
    hsv_ink_pixels: int | None
    classifier_probability: float | None
    classifier_available: bool
    decision: Literal["MARKED", "BLANK", "AMBIGUOUS"]
    method: str


@dataclass(frozen=True)
class OptionMarkEvidence:
    question_id: str
    option_id: str
    strategy: str
    legacy_prediction: Literal["MARKED", "BLANK", "AMBIGUOUS"]
    legacy_method: str
    legacy_score: float
    selected_by_global: bool
    local: LocalOptionEvidence | None
    hsv_fallback: HsvFallbackEvidence | None
    suspicion_notes: tuple[str, ...]


@dataclass(frozen=True)
class QuestionMarkEvidence:
    question_id: str
    strategy: str
    response_type: str
    rois: tuple[RoiEvidence, ...]
    global_topology: GlobalTopologyEvidence | None
    option_evidence: tuple[OptionMarkEvidence, ...]


@dataclass(frozen=True)
class PageMarkEvidence:
    page_number: int
    form_id: str
    form_version: str
    alignment: AlignmentEvidence
    reference: ReferenceEvidence
    questions: tuple[QuestionMarkEvidence, ...]
    thresholds: MarkThresholdEvidence


def evidence_to_mark_scores(evidence: PageMarkEvidence, *, debug_dir: str | Path | None = None) -> list[MarkScore]:
    """Convert the structured PageMarkEvidence into legacy MarkScore objects."""
    scores = []
    
    for question in evidence.questions:
        for opt in question.option_evidence:
            scores.append(MarkScore(
                question_id=opt.question_id,
                option_id=opt.option_id,
                score=opt.legacy_score,
                strategy=opt.strategy,
                method=opt.legacy_method,
                image_crop=None,
                evidence_path=None
            ))
            
    return scores


def evidence_to_json_dict(evidence: PageMarkEvidence) -> dict[str, Any]:
    """Convert PageMarkEvidence into a JSON-serializable dictionary."""
    def _convert(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            if obj.size > 100:
                return {"_type": "ndarray", "shape": obj.shape, "dtype": str(obj.dtype)}
            return obj.tolist()
        elif isinstance(obj, (tuple, list, frozenset, set)):
            return [_convert(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
            
    return _convert(evidence)

def compute_local_option_evidence(
    mask_raw: np.ndarray,
    text_bbox: tuple[int, int, int, int],
    crop_coords: tuple[int, int, int, int],
    outer_radius: int = 32,
    num_bins: int = 72,
    min_ink_per_bin: int = 2
) -> LocalOptionEvidence:
    import cv2
    import numpy as np
    
    bx, by, bw, bh = text_bbox
    h, w = mask_raw.shape
    cx, cy = w / 2.0, h / 2.0
    
    bx = max(0, bx - 2)
    by = max(0, by - 2)
    bw = bw + 4
    bh = bh + 4
    
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - cx)**2 + (Y - cy)**2
    outer_mask = dist_sq > outer_radius**2
    
    core_mask = np.zeros((h, w), dtype=bool)
    by_e = min(h, by+bh)
    bx_e = min(w, bx+bw)
    core_mask[by:by_e, bx:bx_e] = True
    
    mask_radial = mask_raw.copy()
    mask_radial[core_mask] = 0
    mask_radial[outer_mask] = 0
    
    radial_mask_before_morph_px = int(np.sum(mask_radial > 0))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask_radial, kernel, iterations=1)
    mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    ys, xs = np.where(mask_closed > 0)
    radial_ink_pixels = len(xs)
    
    degrees_covered = 0.0
    active_bin_count = 0
    hist_counts = np.zeros(num_bins, dtype=int)
    degrees_per_bin = 360.0 / num_bins
    
    if radial_ink_pixels > 0:
        dx = xs.astype(float) - cx
        dy = ys.astype(float) - cy
        angles = np.arctan2(dy, dx)
        angles = np.degrees(angles) % 360
        hist_counts, _ = np.histogram(angles, bins=num_bins, range=(0, 360))
        active_bins = (hist_counts >= min_ink_per_bin).astype(int)
        for i in range(num_bins):
            left = active_bins[(i - 1) % num_bins]
            right = active_bins[(i + 1) % num_bins]
            if active_bins[i] == 0 and left == 1 and right == 1:
                active_bins[i] = 1
        active_bin_count = int(np.sum(active_bins))
        degrees_covered = float(active_bin_count * degrees_per_bin)
        
    core_mask_pixels = int(np.sum(core_mask))
    outer_mask_pixels = int(np.sum(outer_mask))

    crop_dict = {"x1": crop_coords[0], "y1": crop_coords[1], "x2": crop_coords[2], "y2": crop_coords[3]}
    text_bbox_dict = {"x": text_bbox[0], "y": text_bbox[1], "w": text_bbox[2], "h": text_bbox[3]}
        
    return LocalOptionEvidence(
        crop_coords=crop_dict,
        diff_ink_pixels=0,
        diff_contour_count=0,
        diff_contour_areas=(),
        text_bbox=text_bbox_dict,
        core_mask_pixels=core_mask_pixels,
        outer_mask_pixels=outer_mask_pixels,
        radial_mask_before_morph_px=radial_mask_before_morph_px,
        radial_ink_pixels=radial_ink_pixels,
        radial_active_bin_count=active_bin_count,
        radial_degrees_covered=degrees_covered,
        radial_histogram=tuple(hist_counts.tolist())
    )

def compute_global_topology_evidence(aligned_image_rgb, median_ref_bgr, rois) -> GlobalTopologyEvidence:
    import cv2
    import numpy as np
    from collections import defaultdict
    from matera.vision.adaptive_crop import compute_adaptive_global_crop
    from matera.vision.mark import UnionFind, min_contour_distance, get_horizontal_extremes
    from matera.vision.mark import GAUSS_KERNEL, DIFF_THRESHOLD, CLOSE_KERNEL_SIZE, CLOSE_ITERATIONS, MERGE_THRESHOLD, EXTREMES_REJECT_THRESHOLD
    
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
            global_marked=frozenset()
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
    mask_raw[h-15:, :] = 0
    
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
            contours_ev.append(GlobalContourEvidence(f"cnt_{i}", float(area), bbox, "REJECTED", "area <= 30"))
            continue
        if (h_b > 100 and w_b < 25) or (w_b > 100 and h_b < 25):
            contours_ev.append(GlobalContourEvidence(f"cnt_{i}", float(area), bbox, "REJECTED", "aspect ratio"))
            continue
        
        valid_cnts.append(cnt)
        contours_ev.append(GlobalContourEvidence(f"cnt_{i}", float(area), bbox, "ACCEPTED", None))
            
    n = len(valid_cnts)
    if n == 0:
        return GlobalTopologyEvidence(
            question_id=question_id,
            ran=True,
            skip_reason="no valid contours",
            crop={"x1": crop_x1, "y1": crop_y1, "x2": crop_x2, "y2": crop_y2},
            crop_expansion=getattr(crop_result, "crop_expansion", None),
            option_centers=tuple(option_centers_ev),
            contours=tuple(contours_ev),
            clusters=(),
            global_marked=frozenset()
        )
        
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
        
    clusters_ev = []
    for root, cnt_list in clusters.items():
        combined_points = np.vstack(cnt_list)
        total_area = sum([cv2.contourArea(c) for c in cnt_list])
        
        hull = cv2.convexHull(combined_points)
        hull_area = cv2.contourArea(hull)
        
        # Calculate cluster bbox and points
        cx_min, cy_min, cx_max, cy_max = float("inf"), float("inf"), 0.0, 0.0
        for pt in hull:
            px, py = pt[0]
            cx_min = min(cx_min, px)
            cy_min = min(cy_min, py)
            cx_max = max(cx_max, px)
            cy_max = max(cy_max, py)
        cluster_bbox = {"x": int(cx_min), "y": int(cy_min), "w": int(cx_max - cx_min), "h": int(cy_max - cy_min)}
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
                    
        clusters_ev.append(GlobalClusterEvidence(
            cluster_id=f"cluster_{root}",
            contour_ids=tuple([f"valid_{i}" for i in range(len(cnt_list))]),
            total_area=float(total_area),
            hull_area=float(hull_area),
            solidity=float(solidity),
            bbox=cluster_bbox,
            hull_points=hull_pts,
            inside_options=tuple(inside_options),
            option_center_distances=option_dists
        ))
                        
    return GlobalTopologyEvidence(
        question_id=question_id,
        ran=True,
        skip_reason=None,
        crop={"x1": crop_x1, "y1": crop_y1, "x2": crop_x2, "y2": crop_y2},
        crop_expansion=getattr(crop_result, "crop_expansion", None),
        option_centers=tuple(option_centers_ev),
        contours=tuple(contours_ev),
        clusters=tuple(clusters_ev),
        global_marked=frozenset(global_marked)
    )

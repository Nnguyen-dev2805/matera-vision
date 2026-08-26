from __future__ import annotations

import argparse
import csv
import dataclasses
import html
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from matera.core.layout import BoundingBox, PageLayout, RoiDef, load_layout_profile
from matera.core.profile import FormProfile, load_semantic_profile
from matera.data.extract import extract_pages
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig, MarkScore, RoutingConfig
from matera.vision.mark import (
    BLANK_THRESHOLD_DEG,
    CHECKBOX_INNER_MARGIN,
    DEGREES_PER_BIN,
    LOCAL_PAD,
    MARKED_THRESHOLD_DEG,
    MIN_INK_PER_BIN,
    NUM_BINS,
    OUTER_RADIUS,
    get_local_roi_crops,
    get_local_roi_crops_hsv,
    get_text_bounding_box,
    run_v11_global_topology,
    GLOBAL_PAD,
    MERGE_THRESHOLD,
    EXTREMES_REJECT_THRESHOLD,
    GAUSS_KERNEL,
    CLOSE_KERNEL_SIZE,
    CLOSE_ITERATIONS,
    LOCAL_CLOSE_KERNEL,
    LOCAL_CLOSE_ITERS,
    DIFF_THRESHOLD,

    UnionFind,
    min_contour_distance,
    get_horizontal_extremes,
)


@dataclasses.dataclass
class ContourTrace:
    contour_id: int
    area: float
    bbox: list[int]
    status: str
    reject_reason: str | None

@dataclasses.dataclass
class ClusterTrace:
    cluster_id: int
    contour_ids: list[int]
    total_area: float
    hull_area: float
    solidity: float
    bbox: list[int]
    qualifies_global: bool
    inside_options: list[str]
    rejection_reasons: list[str]

@dataclasses.dataclass
class GlobalTopologyTrace:
    question_id: str
    ran: bool
    skip_reason: str | None
    group_crop: dict[str, int]
    option_centers: list[dict[str, Any]]
    contours: list[ContourTrace]
    clusters: list[ClusterTrace]
    global_marked: list[str]
    artifacts: dict[str, str]
    ink_pipeline_metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    ink_pipeline_artifacts: dict[str, str] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class RoiStageTrace:
    name: str
    status: str
    reason: str
    metrics: dict[str, Any]
    artifacts: list[str]

@dataclasses.dataclass
class QuestionTrace:
    question_id: str

@dataclasses.dataclass
class PixelProbeReport:
    report_version: int
    source_pdf: str
    page_number: int
    alignment_score: float
    warp_matrix: list[list[float]]
    filters: dict[str, Any]
    thresholds: dict[str, Any]
    questions: list[Any]
    global_topology: dict[str, GlobalTopologyTrace]
    traces: list[Any]
    artifacts: dict[str, str]


@dataclasses.dataclass
class RoiPixelTrace:
    page_number: int
    question_id: str
    option_id: str
    strategy: str
    bbox: dict[str, int]
    crop_coords: dict[str, int]
    prediction: str
    method: str
    score: float
    is_global_marked: bool
    diff_ink_pixels: int
    diff_contour_count: int
    diff_contour_areas: list[float]
    text_bbox: dict[str, int] | None
    core_mask_pixels: int
    outer_mask_pixels: int
    radial_ink_pixels: int
    radial_active_bin_count: int
    radial_degrees_covered: float
    radial_histogram: list[int]
    hsv_ink_pixels: int | None
    classifier_probability: float | None
    classifier_available: bool
    routing_selected: bool | None = None
    routing_status: str | None = None
    routing_reason: str | None = None
    decision_path: list[str] = dataclasses.field(default_factory=list)
    suspicion_notes: list[str] = dataclasses.field(default_factory=list)
    stages: list[RoiStageTrace] = dataclasses.field(default_factory=list)
    radial_mask_before_morph_px: int | None = None
    local_ink_pipeline_metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    local_ink_pipeline_artifacts: dict[str, str] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return _json_ready(dataclasses.asdict(self))


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _bbox_dict(bbox: BoundingBox) -> dict[str, int]:
    return {"x": bbox.x, "y": bbox.y, "w": bbox.w, "h": bbox.h}


def _crop_coords_dict(coords: tuple[int, int, int, int]) -> dict[str, int]:
    x1, y1, x2, y2 = coords
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "unnamed"


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    if len(mask.shape) == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def _write_image(path: Path, image: Image.Image | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(path)
        return
    if len(image.shape) == 3:
        cv2.imwrite(str(path), image)
    else:
        cv2.imwrite(str(path), image)


def _contour_stats(mask: np.ndarray) -> tuple[int, list[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted((float(cv2.contourArea(c)) for c in contours), reverse=True)
    return len(contours), areas[:20]


def _analyze_hsv_ai(
    aligned_image: Image.Image,
    median_ref_bgr: np.ndarray,
    roi: RoiDef,
    method_prefix: str,
) -> tuple[str, str, float | None, int, bool, np.ndarray, np.ndarray]:
    _, mask_hsv, _ = get_local_roi_crops_hsv(aligned_image, median_ref_bgr, roi.bbox, 0)
    h, w = mask_hsv.shape
    safe_mask = np.zeros_like(mask_hsv)
    margin = CHECKBOX_INNER_MARGIN
    if h > 2 * margin and w > 2 * margin:
        safe_mask[margin : h - margin, margin : w - margin] = mask_hsv[
            margin : h - margin, margin : w - margin
        ]

    ink_pixels = int(np.sum(safe_mask > 0))
    if ink_pixels < 20:
        return "BLANK", f"{method_prefix}_L2", None, ink_pixels, False, mask_hsv, safe_mask
    if ink_pixels > 200:
        return "MARKED", f"{method_prefix}_L2", None, ink_pixels, False, mask_hsv, safe_mask

    try:
        import matera.vision.mark as mark_module
        from matera.classifier.model import extract_hog_features

        feat = extract_hog_features(safe_mask)
        clf = mark_module._get_classifier()
        probability = float(clf.predict_proba([feat])[0])
    except FileNotFoundError:
        return (
            "AMBIGUOUS",
            f"{method_prefix}_L2_NO_AI",
            None,
            ink_pixels,
            False,
            mask_hsv,
            safe_mask,
        )

    if probability > 0.85:
        return (
            "MARKED",
            f"{method_prefix}_AI ({probability:.2f})",
            probability,
            ink_pixels,
            True,
            mask_hsv,
            safe_mask,
        )
    if probability < 0.15:
        return (
            "BLANK",
            f"{method_prefix}_AI ({probability:.2f})",
            probability,
            ink_pixels,
            True,
            mask_hsv,
            safe_mask,
        )
    return (
        "AMBIGUOUS",
        f"{method_prefix}_AI ({probability:.2f})",
        probability,
        ink_pixels,
        True,
        mask_hsv,
        safe_mask,
    )


def analyze_roi_pixels(
    *,
    aligned_image: Image.Image,
    reference_image: Image.Image,
    roi: RoiDef,
    strategy: str,
    is_global_marked: bool,
    page_number: int,
) -> tuple[RoiPixelTrace, dict[str, Any]]:
    median_ref_bgr = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
    bbox_tuple = (
        roi.bbox.x,
        roi.bbox.y,
        roi.bbox.x + roi.bbox.w,
        roi.bbox.y + roi.bbox.h,
    )
    aligned_roi = aligned_image.crop(bbox_tuple)
    reference_roi = reference_image.crop(bbox_tuple)

    target_bgr, diff_mask, crop_coords, ref_gray = get_local_roi_crops(
        aligned_image, median_ref_bgr, roi.bbox, LOCAL_PAD
    )
    diff_contour_count, diff_contour_areas = _contour_stats(diff_mask)

    h, w = diff_mask.shape
    center_x, center_y = w / 2.0, h / 2.0
    raw_text_bbox = get_text_bounding_box(ref_gray)
    bx, by, bw, bh = raw_text_bbox
    bx = max(0, bx - 2)
    by = max(0, by - 2)
    bw = bw + 4
    bh = bh + 4

    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    outer_mask = dist_sq > OUTER_RADIUS**2

    core_mask = np.zeros((h, w), dtype=bool)
    by_end = min(h, by + bh)
    bx_end = min(w, bx + bw)
    core_mask[by:by_end, bx:bx_end] = True

    radial_mask = diff_mask.copy()
    radial_mask[core_mask] = 0
    radial_mask[outer_mask] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    radial_dilated = cv2.dilate(radial_mask, kernel, iterations=1)
    radial_closed = cv2.morphologyEx(radial_dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

    ys, xs = np.where(radial_closed > 0)
    radial_ink_pixels = int(len(xs))
    radial_degrees_covered = 0.0
    radial_active_bin_count = 0
    radial_histogram = [0] * NUM_BINS

    if radial_ink_pixels > 0:
        dx = xs.astype(float) - center_x
        dy = ys.astype(float) - center_y
        angles = np.degrees(np.arctan2(dy, dx)) % 360
        hist_counts, _ = np.histogram(angles, bins=NUM_BINS, range=(0, 360))
        active_bins = (hist_counts >= MIN_INK_PER_BIN).astype(int)
        for i in range(NUM_BINS):
            left = active_bins[(i - 1) % NUM_BINS]
            right = active_bins[(i + 1) % NUM_BINS]
            if active_bins[i] == 0 and left == 1 and right == 1:
                active_bins[i] = 1
        radial_active_bin_count = int(np.sum(active_bins))
        radial_degrees_covered = float(radial_active_bin_count * DEGREES_PER_BIN)
        radial_histogram = [int(v) for v in hist_counts.tolist()]

    prediction = "AMBIGUOUS"
    method = "UNKNOWN"
    hsv_ink_pixels: int | None = None
    classifier_probability: float | None = None
    classifier_available = False
    hsv_mask = np.zeros((max(roi.bbox.h, 1), max(roi.bbox.w, 1)), dtype=np.uint8)
    safe_hsv_mask = hsv_mask.copy()
    decision_path: list[str] = []
    suspicion_notes: list[str] = []

    if is_global_marked:
        prediction = "MARKED"
        method = "GLOBAL_HULL"
        decision_path.append("Global topology marked this option before local radial analysis.")
    elif "Q14" in roi.question_id:
        (
            prediction,
            method,
            classifier_probability,
            hsv_ink_pixels,
            classifier_available,
            hsv_mask,
            safe_hsv_mask,
        ) = _analyze_hsv_ai(aligned_image, median_ref_bgr, roi, "HSV_AI")
        decision_path.append("Q14 uses HSV/AI checkbox logic and skips circle radial analysis.")
        decision_path.append(f"HSV safe-mask ink pixels: {hsv_ink_pixels}.")
    else:
        decision_path.append(
            f"Local radial kept {radial_ink_pixels} pixels after removing "
            "text core and outer radius."
        )
        decision_path.append(
            f"Local radial covered {radial_degrees_covered:.1f} degrees "
            f"({radial_active_bin_count}/{NUM_BINS} active bins)."
        )
        if radial_degrees_covered >= MARKED_THRESHOLD_DEG:
            prediction = "MARKED"
            method = "LOCAL_RADIAL"
            decision_path.append(
                f"Decision is MARKED because {radial_degrees_covered:.1f} "
                f">= {MARKED_THRESHOLD_DEG}."
            )
        elif radial_degrees_covered <= BLANK_THRESHOLD_DEG:
            prediction = "BLANK"
            method = "LOCAL_RADIAL"
            decision_path.append(
                f"Decision is BLANK because {radial_degrees_covered:.1f} <= {BLANK_THRESHOLD_DEG}."
            )
        else:
            decision_path.append(
                f"Radial result is ambiguous because {BLANK_THRESHOLD_DEG} < "
                f"{radial_degrees_covered:.1f} < {MARKED_THRESHOLD_DEG}; running HSV/AI fallback."
            )
            (
                prediction,
                method,
                classifier_probability,
                hsv_ink_pixels,
                classifier_available,
                hsv_mask,
                safe_hsv_mask,
            ) = _analyze_hsv_ai(aligned_image, median_ref_bgr, roi, "FALLBACK")
            decision_path.append(f"HSV safe-mask ink pixels: {hsv_ink_pixels}.")

    score = 1.0 if prediction == "MARKED" else 0.0 if prediction == "BLANK" else 0.5

    if int(np.sum(diff_mask > 0)) > 0 and radial_ink_pixels == 0 and not is_global_marked:
        suspicion_notes.append(
            "Diff mask has ink-like pixels, but radial analysis removed all of them."
        )
    if hsv_ink_pixels is not None and hsv_ink_pixels < 20 and int(np.sum(diff_mask > 0)) >= 20:
        suspicion_notes.append(
            "Diff mask sees pixels, but HSV safe mask is below the blank threshold."
        )
    if prediction == "AMBIGUOUS":
        suspicion_notes.append("Extractor returned AMBIGUOUS, which routes to needs_review.")

    trace = RoiPixelTrace(
        page_number=page_number,
        question_id=roi.question_id,
        option_id=roi.option_id,
        strategy=strategy,
        bbox=_bbox_dict(roi.bbox),
        crop_coords=_crop_coords_dict(crop_coords),
        prediction=prediction,
        method=method,
        score=score,
        is_global_marked=is_global_marked,
        diff_ink_pixels=int(np.sum(diff_mask > 0)),
        diff_contour_count=diff_contour_count,
        diff_contour_areas=diff_contour_areas,
        text_bbox={"x": bx, "y": by, "w": bw, "h": bh},
        core_mask_pixels=int(np.sum(core_mask)),
        outer_mask_pixels=int(np.sum(outer_mask)),
        radial_ink_pixels=radial_ink_pixels,
        radial_active_bin_count=radial_active_bin_count,
        radial_degrees_covered=radial_degrees_covered,
        radial_histogram=radial_histogram,
        hsv_ink_pixels=hsv_ink_pixels,
        classifier_probability=classifier_probability,
        classifier_available=classifier_available,
        decision_path=decision_path,
        suspicion_notes=suspicion_notes,
        radial_mask_before_morph_px=int(np.sum(radial_mask > 0)),
    )
    masks = {
        "aligned_roi": aligned_roi,
        "reference_roi": reference_roi,
        "target_bgr": target_bgr,
        "diff_mask": diff_mask,
        "radial_mask": radial_closed,
        "hsv_mask": hsv_mask,
        "safe_hsv_mask": safe_hsv_mask,
        "core_mask": (core_mask.astype(np.uint8) * 255),
        "outer_mask": (outer_mask.astype(np.uint8) * 255),
    }
    return trace, masks


def build_routing_traces(
    *,
    profile: FormProfile,
    page_number: int,
    roi_traces: list[dict[str, Any]],
    config: RoutingConfig,
) -> dict[tuple[str, str], dict[str, Any]]:
    score_by_key = {
        (str(t["question_id"]), str(t["option_id"])): float(t["score"])
        for t in roi_traces
    }
    result: dict[tuple[str, str], dict[str, Any]] = {}

    for q_def in profile.questions:
        original_selections: dict[str, bool | None] = {}
        final_selections: dict[str, bool | None] = {}

        for opt in q_def.options:
            score = score_by_key[(q_def.question_id, opt.option_id)]
            if score >= config.high_threshold:
                original_selections[opt.option_id] = True
            elif score < config.low_threshold:
                original_selections[opt.option_id] = False
            else:
                original_selections[opt.option_id] = None
            final_selections[opt.option_id] = original_selections[opt.option_id]

        num_selected = sum(1 for v in original_selections.values() if v is True)
        q_scores = [
            (opt.option_id, score_by_key[(q_def.question_id, opt.option_id)])
            for opt in q_def.options
        ]
        max_score = max((score for _, score in q_scores), default=0.0)

        if q_def.max_selections is not None and num_selected > q_def.max_selections:
            for opt_id, is_selected in original_selections.items():
                if is_selected is True:
                    final_selections[opt_id] = None
        elif num_selected < q_def.min_selections and q_scores:
            for opt_id, score in q_scores:
                if score == max_score:
                    final_selections[opt_id] = None

        for opt in q_def.options:
            score = score_by_key[(q_def.question_id, opt.option_id)]
            selected = final_selections[opt.option_id]
            status = "resolved" if selected is not None else "needs_review"
            reason = ""
            original = original_selections[opt.option_id]
            if status == "needs_review":
                if original is True:
                    reason = (
                        f"Over-selection: {num_selected} options selected, "
                        f"max is {q_def.max_selections}"
                    )
                elif original is False:
                    reason = (
                        f"Under-selection: {num_selected} options selected, "
                        f"min is {q_def.min_selections}. Highest score was {score:.3f}"
                    )
                elif num_selected < q_def.min_selections and score == max_score:
                    reason = (
                        f"Under-selection (and ambiguous): {num_selected} options selected, "
                        f"min is {q_def.min_selections}. Highest score was {score:.3f}"
                    )
                else:
                    reason = (
                        f"Score {score:.3f} is ambiguous "
                        f"(thresholds: {config.low_threshold}-{config.high_threshold})"
                    )

            result[(q_def.question_id, opt.option_id)] = {
                "page_number": page_number,
                "question_id": q_def.question_id,
                "option_id": opt.option_id,
                "score": score,
                "original_selection": original,
                "selected": selected,
                "resolution_status": status,
                "review_reason": reason,
                "low_threshold": config.low_threshold,
                "high_threshold": config.high_threshold,
                "min_selections": q_def.min_selections,
                "max_selections": q_def.max_selections,
                "num_selected_before_constraints": num_selected,
            }

    return result


def _scale_layout_rois(layout: PageLayout, image: Image.Image) -> list[RoiDef]:
    scale_x = image.width / layout.width_px
    scale_y = image.height / layout.height_px
    scaled = []
    for roi in layout.rois:
        scaled_bbox = dataclasses.replace(
            roi.bbox,
            x=int(roi.bbox.x * scale_x),
            y=int(roi.bbox.y * scale_y),
            w=int(roi.bbox.w * scale_x),
            h=int(roi.bbox.h * scale_y),
        )
        scaled.append(dataclasses.replace(roi, bbox=scaled_bbox))
    return scaled


def _selected_page(pdf_path: Path, page_number: int):
    for page in extract_pages(pdf_path):
        if page.page_number == page_number:
            return page
    raise ValueError(f"PDF {pdf_path} does not contain page {page_number}")


def _draw_page_overlay(image: Image.Image, traces: list[RoiPixelTrace]) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for trace in traces:
        bbox = trace.bbox
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        if trace.routing_status == "needs_review":
            color = "yellow"
        elif trace.prediction == "MARKED":
            color = "lime"
        elif trace.prediction == "BLANK":
            color = "red"
        else:
            color = "magenta"
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        label = f"{trace.question_id}/{trace.option_id} {trace.score:.1f}"
        draw.text((x, max(0, y - 12)), label, fill=color)
    return overlay


def _trace_row(trace: RoiPixelTrace) -> dict[str, Any]:
    return {
        "page_number": trace.page_number,
        "question_id": trace.question_id,
        "option_id": trace.option_id,
        "strategy": trace.strategy,
        "prediction": trace.prediction,
        "method": trace.method,
        "score": trace.score,
        "is_global_marked": trace.is_global_marked,
        "routing_status": trace.routing_status or "",
        "routing_selected": trace.routing_selected,
        "routing_reason": trace.routing_reason or "",
        "diff_ink_pixels": trace.diff_ink_pixels,
        "diff_contour_count": trace.diff_contour_count,
        "radial_ink_pixels": trace.radial_ink_pixels,
        "radial_active_bin_count": trace.radial_active_bin_count,
        "radial_degrees_covered": f"{trace.radial_degrees_covered:.3f}",
        "hsv_ink_pixels": trace.hsv_ink_pixels if trace.hsv_ink_pixels is not None else "",
        "classifier_probability": (
            f"{trace.classifier_probability:.6f}"
            if trace.classifier_probability is not None
            else ""
        ),
        "suspicion_notes": " | ".join(trace.suspicion_notes),
    }

def _write_roi_artifacts(out_dir: Path, trace: RoiPixelTrace, masks: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_image(out_dir / "01_aligned_roi.png", masks["aligned_roi"])
    _write_image(out_dir / "02_reference_roi.png", masks["reference_roi"])
    _write_image(out_dir / "03_diff_mask.png", masks["diff_mask"])
    _write_image(out_dir / "04_radial_mask.png", masks["radial_mask"])
    _write_image(out_dir / "05_hsv_mask.png", masks["hsv_mask"])
    _write_image(out_dir / "06_safe_hsv_mask.png", masks["safe_hsv_mask"])
    _write_image(out_dir / "07_core_mask.png", masks["core_mask"])
    _write_image(out_dir / "08_outer_mask.png", masks["outer_mask"])

    overlay = np.array(masks["aligned_roi"].convert("RGB"))
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.putText(
        overlay_bgr,
        f"{trace.prediction} {trace.method}",
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay_bgr,
        f"score={trace.score:.1f} deg={trace.radial_degrees_covered:.1f}",
        (4, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 0, 255),
        1,
        cv2.LINE_AA,
    )
    _write_image(out_dir / "09_overlay_decision.png", overlay_bgr)

    with open(out_dir / "angle_histogram.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_index", "start_deg", "end_deg", "pixel_count"])
        for i, count in enumerate(trace.radial_histogram):
            writer.writerow([i, i * DEGREES_PER_BIN, (i + 1) * DEGREES_PER_BIN, count])

    with open(out_dir / "trace.json", "w", encoding="utf-8") as f:
        json.dump(trace.as_dict(), f, indent=2, ensure_ascii=False)

    lines = [
        f"{trace.question_id}/{trace.option_id}",
        f"Prediction: {trace.prediction}",
        f"Method: {trace.method}",
        f"Score: {trace.score}",
        "",
        "Decision path:",
    ]
    lines.extend(f"{idx}. {step}" for idx, step in enumerate(trace.decision_path, start=1))
    if trace.routing_status:
        lines.extend(
            [
                "",
                "Routing:",
                f"- selected: {trace.routing_selected}",
                f"- status: {trace.routing_status}",
                f"- reason: {trace.routing_reason or ''}",
            ]
        )
    if trace.suspicion_notes:
        lines.extend(["", "Suspicion notes:"])
        lines.extend(f"- {note}" for note in trace.suspicion_notes)
    (out_dir / "decision_trace.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(out_dir: Path, report: PixelProbeReport) -> None:
    traces = report.traces
    review = [t for t in traces if t.routing_status == "needs_review"]
    suspect = [t for t in traces if t.suspicion_notes]
    marked = [t for t in traces if t.prediction == "MARKED"]
    blank = [t for t in traces if t.prediction == "BLANK"]

    lines = [
        "# Pixel Probe Summary",
        "",
        f"- Source PDF: `{report.source_pdf}`",
        f"- Page: {report.page_number}",
        f"- Total options: {len(traces)}",
        f"- Marked: {len(marked)}",
        f"- Blank: {len(blank)}",
        f"- Needs review: {len(review)}",
        f"- Suspicious traces: {len(suspect)}",
        "",
        "## Global Topology",
    ]
    for q_id, gt in report.global_topology.items():
        if gt.ran:
            lines.append(f"- **{q_id}**: {len(gt.contours)} valid contours, {len(gt.clusters)} clusters. Global marked: {', '.join(gt.global_marked) if gt.global_marked else 'None'}")
        else:
            lines.append(f"- **{q_id}**: Skipped ({gt.skip_reason})")
            
    lines.extend([
        "",
        "## Needs Review",
    ])
    if review:
        for t in review:
            lines.append(f"- `{t.question_id}/{t.option_id}`: {t.routing_reason}")
    else:
        lines.append("- none")

    lines.extend(["", "## Suspect Notes"])
    if suspect:
        for t in suspect:
            lines.append(f"- `{t.question_id}/{t.option_id}`: {' | '.join(t.suspicion_notes)}")
    else:
        lines.append("- none")

    lines.extend(["", "## Top Diff Pixels"])
    for t in sorted(traces, key=lambda x: x.diff_ink_pixels, reverse=True)[:15]:
        lines.append(
            f"- `{t.question_id}/{t.option_id}` diff={t.diff_ink_pixels}, "
            f"radial={t.radial_ink_pixels}, deg={t.radial_degrees_covered:.1f}, "
            f"method={t.method}, score={t.score}"
        )

    (out_dir / "page_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def _write_index_html(out_dir: Path, report: PixelProbeReport) -> None:
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Matera Pixel Probe Debugger</title>
  <style>
    :root {{
      --bg: #121212; --text: #e0e0e0; --surface: #1e1e1e; --border: #333;
      --primary: #bb86fc; --success: #03dac6; --error: #cf6679; --warn: #ffb74d;
    }}
    body {{
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      margin: 0; padding: 0; display: flex; height: 100vh;
      background: var(--bg); color: var(--text);
    }}
    a {{ color: var(--primary); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    #sidebar {{
      width: 280px; border-right: 1px solid var(--border); background: var(--surface);
      overflow-y: auto; padding: 16px;
    }}
    #main {{
      flex: 1; overflow-y: auto; padding: 24px;
    }}
    .q-item {{
      padding: 8px 12px; margin-bottom: 8px; border-radius: 4px; cursor: pointer;
      border: 1px solid var(--border); transition: background 0.2s;
    }}
    .q-item:hover, .q-item.active {{ background: #333; border-color: var(--primary); }}
    .status-badge {{
      display: inline-block; padding: 2px 6px; border-radius: 12px; font-size: 11px;
      font-weight: bold; margin-left: 8px;
    }}
    .status-resolved {{ background: rgba(3, 218, 198, 0.2); color: var(--success); }}
    .status-review {{ background: rgba(207, 102, 121, 0.2); color: var(--error); }}
    
    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
    .card h3 {{ margin-top: 0; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
    
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }}
    
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 13px; }}
    th, td {{ border: 1px solid var(--border); padding: 8px; text-align: left; }}
    th {{ background: #2c2c2c; }}
    
    .img-box {{ background: #000; text-align: center; border: 1px solid var(--border); padding: 8px; border-radius: 4px; }}
    .img-box img {{ max-width: 100%; height: auto; image-rendering: pixelated; }}
    
    .metrics-list {{ list-style: none; padding: 0; margin: 0; font-size: 13px; }}
    .metrics-list li {{ display: flex; justify-content: space-between; margin-bottom: 4px; border-bottom: 1px dashed #444; }}
  </style>
</head>
<body>
  <div id="sidebar">
    <h2>Pixel Probe</h2>
    <div style="font-size: 12px; color: #888; margin-bottom: 16px;">
      PDF: {report.source_pdf} (Page {report.page_number})<br>
      Align Score: {report.alignment_score:.3f}
    </div>
    <div id="q-list"></div>
  </div>
  <div id="main">
    <div id="content">
      <h2 style="color:#888;">Select a question to view details.</h2>
      <a href="page_summary.md">page_summary.md</a> | <a href="roi_trace.csv">roi_trace.csv</a>
    </div>
  </div>

  <script>
    const REPORT_DATA = {json.dumps(_json_ready(dataclasses.asdict(report)))};
    
    function init() {{
      const qList = document.getElementById("q-list");
      
      // Group traces by question
      const tracesByQ = {{}};
      REPORT_DATA.traces.forEach(t => {{
        if(!tracesByQ[t.question_id]) tracesByQ[t.question_id] = [];
        tracesByQ[t.question_id].push(t);
      }});
      
      const sortedQs = Object.keys(tracesByQ).sort();
      
      sortedQs.forEach(qId => {{
        const div = document.createElement("div");
        div.className = "q-item";
        
        let needsReview = tracesByQ[qId].some(t => t.routing_status === "needs_review");
        let badgeClass = needsReview ? "status-review" : "status-resolved";
        let badgeText = needsReview ? "Review" : "Resolved";
        
        div.innerHTML = `<strong>${{qId}}</strong> <span class="status-badge ${{badgeClass}}">${{badgeText}}</span>`;
        div.onclick = () => {{
          document.querySelectorAll(".q-item").forEach(el => el.classList.remove("active"));
          div.classList.add("active");
          renderQuestion(qId, tracesByQ[qId], REPORT_DATA.global_topology[qId]);
        }};
        qList.appendChild(div);
      }});
    }}
    
    function renderQuestion(qId, traces, globalTopo) {{
      const content = document.getElementById("content");
      let html = `<h2>Question: ${{qId}}</h2>`;
      
      // Global Topology Card
      if (globalTopo && globalTopo.ran) {{
        html += `<div class="card">
          <h3>Global Topology Analysis</h3>
          <div class="grid-2">
            <div>
              <ul class="metrics-list">
                <li><span>Contours found:</span> <span>${{globalTopo.contours.length}}</span></li>
                <li><span>Clusters:</span> <span>${{globalTopo.clusters.length}}</span></li>
                <li><span>Global Marked:</span> <span>${{globalTopo.global_marked.length > 0 ? globalTopo.global_marked.join(', ') : 'None'}}</span></li>
              </ul>
            </div>
            <div class="img-box">
              ${{globalTopo.artifacts && globalTopo.artifacts.global ? `<img src="${{globalTopo.artifacts.global}}">` : `<i>No global artifact</i>`}}
            </div>
          </div>
        </div>`;
      }} else if (globalTopo) {{
        html += `<div class="card"><h3>Global Topology Analysis</h3><p>Skipped: ${{globalTopo.skip_reason}}</p></div>`;
      }}
      
      // Local Options Cards
      html += `<div class="card"><h3>Local Options Analysis</h3><div class="grid-3">`;
      traces.forEach(t => {{
        let art = globalTopo && globalTopo.artifacts ? globalTopo.artifacts["option_"+t.option_id] : null;
        let imgHtml = art ? `<div class="img-box"><img src="${{art}}"></div>` : "";
        
        let suspHtml = t.suspicion_notes && t.suspicion_notes.length ? `<div style="color:var(--warn);font-size:12px;margin-top:8px;">Suspicion:<ul><li>${{t.suspicion_notes.join('</li><li>')}}</li></ul></div>` : "";
        
        html += `
          <div style="border: 1px solid #444; border-radius: 4px; padding: 12px; background: #222;">
            <div style="font-weight:bold;font-size:16px;margin-bottom:8px;">Option: ${{t.option_id}}</div>
            ${{imgHtml}}
            <div style="margin-top: 12px;">
              <ul class="metrics-list">
                <li><span>Prediction:</span> <strong>${{t.prediction}}</strong></li>
                <li><span>Score:</span> <span>${{t.score.toFixed(3)}}</span></li>
                <li><span>Method:</span> <span>${{t.method}}</span></li>
                <li><span>Routing:</span> <span class="${{t.routing_status === 'needs_review' ? 'status-review' : 'status-resolved'}} status-badge" style="margin:0">${{t.routing_status || 'N/A'}}</span></li>
              </ul>
              ${{suspHtml}}
            </div>
          </div>
        `;
      }});
      html += `</div></div>`;
      
      // Decision Path
      if (traces.length > 0) {{
         let traceHtml = traces.map(t => `<div><strong>${{t.option_id}}:</strong> ${{(t.decision_path||[]).join(" &rarr; ")}}</div>`).join("");
         html += `<div class="card"><h3>Decision Paths</h3>${{traceHtml}}</div>`;
      }}
      
      content.innerHTML = html;
    }}
    
    document.addEventListener("DOMContentLoaded", init);
  </script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")

def run_pixel_probe(
    *,
    pdf_path: Path,
    page_number: int,
    output_root: Path,
    profile_dir: Path,
    reference_path: Path,
    question: str | None = None,
    option: str | None = None,
) -> Path:
    semantic_profile = load_semantic_profile(profile_dir / "semantic.json")
    layout_profile = load_layout_profile(profile_dir / "layout.json", semantic=semantic_profile)
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_path}")

    reference_image = Image.open(reference_path).convert("RGB")
    rendered_page = _selected_page(pdf_path, page_number)
    align_config = AlignmentConfig(algorithm="orb", transform_model="affine", inlier_threshold=0.05)
    aligned_page = align_page(rendered_page, reference_image, align_config)

    layout = layout_profile.pages[0]
    scaled_rois = _scale_layout_rois(layout, aligned_page.image)
    if question:
        scaled_rois = [r for r in scaled_rois if r.question_id == question]
    if option:
        scaled_rois = [r for r in scaled_rois if r.option_id == option]
    if not scaled_rois:
        raise ValueError("No ROIs matched the requested filters.")

    strategy_map = {q.question_id: q.mark_strategy for q in semantic_profile.questions}
    rois_by_q: dict[str, list[RoiDef]] = defaultdict(list)
    for roi in scaled_rois:
        rois_by_q[roi.question_id].append(roi)

    median_ref_bgr = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
    debug_full_mask = np.zeros_like(np.array(aligned_page.image))
    global_marked_by_q: dict[str, set[str]] = {}
    for q_id, rois in rois_by_q.items():
        if "Q14" in q_id:
            global_marked_by_q[q_id] = set()
        else:
            global_marked_by_q[q_id] = run_v11_global_topology(
                aligned_page.image, median_ref_bgr, rois, debug_full_mask
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    page_dir = output_root / timestamp / f"page_{page_number:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    _write_image(page_dir / "raw_page.png", rendered_page.image)
    _write_image(page_dir / "aligned_page.png", aligned_page.image)
    _write_image(page_dir / "reference_page.png", reference_image)

    traces: list[RoiPixelTrace] = []
    masks_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for roi in scaled_rois:
        strategy = roi.mark_strategy_override or strategy_map.get(roi.question_id)
        if not strategy:
            raise ValueError(f"Cannot resolve mark strategy for ROI question_id={roi.question_id}")
        trace, masks = analyze_roi_pixels(
            aligned_image=aligned_page.image,
            reference_image=reference_image,
            roi=roi,
            strategy=strategy,
            is_global_marked=roi.option_id in global_marked_by_q.get(roi.question_id, set()),
            page_number=page_number,
        )
        traces.append(trace)
        masks_by_key[(trace.question_id, trace.option_id)] = masks

    global_topology_traces = {}
    
    # Trace global topology per question for explainability UI
    for q_id, rois in rois_by_q.items():
        if "Q14" in q_id:
            continue
        g_trace = trace_global_topology(aligned_page.image, median_ref_bgr, rois, q_id)
        global_topology_traces[q_id] = g_trace

    routing_config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)
    if question or option:
        # Routing constraints need the complete page. For filtered microscope runs,
        # report extraction details and leave routing blank rather than inventing context.
        routing_traces: dict[tuple[str, str], dict[str, Any]] = {}
    else:
        routing_traces = build_routing_traces(
            profile=semantic_profile,
            page_number=page_number,
            roi_traces=[t.as_dict() for t in traces],
            config=routing_config,
        )

    for trace in traces:
        routing = routing_traces.get((trace.question_id, trace.option_id))
        if routing:
            trace.routing_selected = routing["selected"]
            trace.routing_status = routing["resolution_status"]
            trace.routing_reason = routing["review_reason"]
        roi_dir = page_dir / _safe_name(trace.question_id) / _safe_name(trace.option_id)
        _write_roi_artifacts(roi_dir, trace, masks_by_key[(trace.question_id, trace.option_id)])
        
    for q_id, g_trace in global_topology_traces.items():
        q_traces = [t for t in traces if t.question_id == q_id]
        if not q_traces:
            continue
        q_artifacts = generate_question_artifacts(
            aligned_page.image, median_ref_bgr, g_trace, q_traces, page_dir
        )
        g_trace.artifacts = q_artifacts
        
        # Capture global ink pipeline
        g_metrics, g_ink_artifacts = _capture_global_ink_pipeline(
            aligned_page.image, median_ref_bgr, g_trace, page_dir / _safe_name(q_id)
        )
        g_trace.ink_pipeline_metrics = g_metrics
        g_trace.ink_pipeline_artifacts = g_ink_artifacts

    for trace in traces:
        roi = next(r for r in scaled_rois if r.question_id == trace.question_id and r.option_id == trace.option_id)
        roi_dir = page_dir / _safe_name(trace.question_id) / _safe_name(trace.option_id)
        
        # Capture local ink pipeline
        l_metrics, l_artifacts = _capture_local_ink_pipeline(
            aligned_page.image, reference_image, roi, trace, roi_dir
        )
        trace.local_ink_pipeline_metrics = l_metrics
        trace.local_ink_pipeline_artifacts = l_artifacts

    overlay = _draw_page_overlay(aligned_page.image, traces)
    _write_image(page_dir / "page_overlay_all_rois.png", overlay)

    report = PixelProbeReport(
        report_version=2,
        source_pdf=str(pdf_path),
        page_number=page_number,
        alignment_score=aligned_page.alignment_score,
        warp_matrix=np.asarray(aligned_page.warp_matrix).tolist(),
        filters={"question": question, "option": option, "routing_evaluated": not bool(question or option)},
        thresholds={"routing_low": routing_config.low_threshold, "routing_high": routing_config.high_threshold},
        questions=[], 
        global_topology=global_topology_traces,
        traces=traces,
        artifacts={"page_overlay": "page_overlay_all_rois.png"}
    )

    with open(page_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready(dataclasses.asdict(report)), f, indent=2, ensure_ascii=False)

    with open(page_dir / "roi_trace.csv", "w", newline="", encoding="utf-8") as f:
        rows = [_trace_row(t) for t in traces]
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _write_summary(page_dir, report)
    _write_index_html(page_dir, report)
    return page_dir

def _default_debug_pdf() -> Path:
    debug_dir = Path("data/test/debug")
    matches = sorted(debug_dir.glob("*.pdf"))
    if not matches:
        raise FileNotFoundError("No PDF files found in data/test/debug. Pass --pdf explicitly.")
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deep pixel-level Matera OMR pipeline debugger.")
    parser.add_argument("--pdf", type=Path, default=None, help="PDF to debug.")
    parser.add_argument("--page", type=int, default=1, help="1-based PDF page number to debug.")
    parser.add_argument("--profile-dir", type=Path, default=Path("profiles"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("scratch/synthetic_median_reference.png"),
        help="Reference image used by the production pipeline.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/debug/pixel_probe"))
    parser.add_argument(
        "--question", type=str, default=None, help="Optional question filter, e.g. Q4."
    )
    parser.add_argument("--option", type=str, default=None, help="Optional option filter, e.g. c.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pdf_path = args.pdf or _default_debug_pdf()
    out_dir = run_pixel_probe(
        pdf_path=pdf_path,
        page_number=args.page,
        output_root=args.output_dir,
        profile_dir=args.profile_dir,
        reference_path=args.reference,
        question=args.question,
        option=args.option,
    )
    print(f"Pixel probe written to {out_dir}")
    print(f"Open {out_dir / 'index.html'} to inspect ROI traces.")
    return 0


def trace_global_topology(
    aligned_image_rgb: Image.Image,
    median_ref_bgr: np.ndarray,
    rois: list[RoiDef],
    question_id: str
) -> GlobalTopologyTrace:
    if len(rois) <= 1:
        return GlobalTopologyTrace(
            question_id=question_id,
            ran=False,
            skip_reason="Insufficient ROIs (<= 1)",
            group_crop={},
            option_centers=[],
            contours=[],
            clusters=[],
            global_marked=[],
            artifacts={}
        )
        
    min_x = min(r.bbox.x for r in rois)
    min_y = min(r.bbox.y for r in rois)
    max_x = max(r.bbox.x + r.bbox.w for r in rois)
    max_y = max(r.bbox.y + r.bbox.h for r in rois)
    
    crop_x1 = max(0, min_x - GLOBAL_PAD)
    crop_y1 = max(0, min_y - GLOBAL_PAD)
    crop_x2 = min(aligned_image_rgb.width, max_x + GLOBAL_PAD)
    crop_y2 = min(aligned_image_rgb.height, max_y + GLOBAL_PAD)
    group_crop = {"x1": crop_x1, "y1": crop_y1, "x2": crop_x2, "y2": crop_y2}
    
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
    
    option_centers = []
    for r in rois:
        cx = (r.bbox.x + r.bbox.w / 2.0) - crop_x1
        cy = (r.bbox.y + r.bbox.h / 2.0) - crop_y1
        option_centers.append({"option_id": r.option_id, "x": cx, "y": cy})
        
    valid_cnts = []
    contour_traces = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        bbox = [x, y, cw, ch]
        
        status = "valid"
        reject_reason = None
        if area <= 30:
            status = "rejected"
            reject_reason = "Area <= 30"
        elif (ch > 100 and cw < 25) or (cw > 100 and ch < 25):
            status = "rejected"
            reject_reason = "Extreme aspect ratio"
            
        contour_traces.append(ContourTrace(
            contour_id=idx,
            area=float(area),
            bbox=bbox,
            status=status,
            reject_reason=reject_reason
        ))
        
        if status == "valid":
            valid_cnts.append((idx, cnt))
            
    n = len(valid_cnts)
    if n == 0:
        return GlobalTopologyTrace(
            question_id=question_id,
            ran=True,
            skip_reason=None,
            group_crop=group_crop,
            option_centers=option_centers,
            contours=contour_traces,
            clusters=[],
            global_marked=[],
            artifacts={}
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
                    
    clusters_dict = defaultdict(list)
    for i in range(n):
        clusters_dict[uf.find(i)].append(valid_cnts[i])
        
    cluster_traces = []
    global_marked = []
    
    for root, cnt_list in clusters_dict.items():
        combined_points = np.vstack([c[1] for c in cnt_list])
        total_area = sum([cv2.contourArea(c[1]) for c in cnt_list])
        
        hull = cv2.convexHull(combined_points)
        hull_area = cv2.contourArea(hull)
        hx, hy, hw, hh = cv2.boundingRect(hull)
        
        rejection_reasons = []
        qualifies_global = False
        inside_options = []
        
        if hull_area > 1000:
            solidity = total_area / float(hull_area) if hull_area > 0 else 1.0
            if solidity < 0.4:
                qualifies_global = True
                for opt in option_centers:
                    if cv2.pointPolygonTest(hull, (opt['x'], opt['y']), False) >= 0:
                        global_marked.append(opt['option_id'])
                        inside_options.append(opt['option_id'])
            else:
                rejection_reasons.append(f"Solidity {solidity:.2f} >= 0.4")
        else:
            rejection_reasons.append(f"Hull area {hull_area} <= 1000")
            
        cluster_traces.append(ClusterTrace(
            cluster_id=root,
            contour_ids=[c[0] for c in cnt_list],
            total_area=float(total_area),
            hull_area=float(hull_area),
            solidity=float(total_area / hull_area) if hull_area > 0 else 1.0,
            bbox=[hx, hy, hw, hh],
            qualifies_global=qualifies_global,
            inside_options=inside_options,
            rejection_reasons=rejection_reasons
        ))
        
    return GlobalTopologyTrace(
        question_id=question_id,
        ran=True,
        skip_reason=None,
        group_crop=group_crop,
        option_centers=option_centers,
        contours=contour_traces,
        clusters=cluster_traces,
        global_marked=global_marked,
        artifacts={}
    )



def _capture_global_ink_pipeline(
    aligned_image_rgb: Image.Image,
    median_ref_bgr: np.ndarray,
    global_trace: GlobalTopologyTrace,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}

    if not global_trace.group_crop:
        return metrics, artifacts

    crop = global_trace.group_crop
    crop_x1, crop_y1 = crop['x1'], crop['y1']
    crop_x2, crop_y2 = crop['x2'], crop['y2']

    # Step 0: Raw Crop
    crop_img = aligned_image_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    target_bgr = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / 'global_step0_raw_crop.png'), target_bgr)
    artifacts['step0'] = 'global_step0_raw_crop.png'

    # Step 1: Reference Crop
    ref_crop = median_ref_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    cv2.imwrite(str(out_dir / 'global_step1_ref_crop.png'), ref_crop)
    artifacts['step1'] = 'global_step1_ref_crop.png'

    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)

    # Step 2: AbsDiff
    diff = cv2.absdiff(ref_gray, target_gray)
    metrics['absdiff_nonzero_px'] = int(np.sum(diff > 0))
    cv2.imwrite(str(out_dir / 'global_step2_absdiff.png'), diff)
    artifacts['step2'] = 'global_step2_absdiff.png'

    # Step 3: Blurred Diff
    blurred = cv2.GaussianBlur(diff, GAUSS_KERNEL, 0)
    cv2.imwrite(str(out_dir / 'global_step3_blurred.png'), blurred)
    artifacts['step3'] = 'global_step3_blurred.png'

    # Step 4: Threshold Mask
    _, mask_raw = cv2.threshold(blurred, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    h, w = mask_raw.shape
    mask_raw[:15, :] = 0
    mask_raw[h-15:, :] = 0
    metrics['threshold_value'] = DIFF_THRESHOLD
    metrics['threshold_ink_px'] = int(np.sum(mask_raw > 0))
    cv2.imwrite(str(out_dir / 'global_step4_threshold.png'), mask_raw)
    artifacts['step4'] = 'global_step4_threshold.png'

    # Step 5: Closed Mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERATIONS)
    metrics['closed_ink_px'] = int(np.sum(mask_closed > 0))
    cv2.imwrite(str(out_dir / 'global_step5_closed.png'), mask_closed)
    artifacts['step5'] = 'global_step5_closed.png'

    # Prepare for Step 6 and 7 by drawing on original crop
    contours_img = target_bgr.copy()
    clusters_img = target_bgr.copy()

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    metrics['contour_count_raw'] = len(contours)
    
    valid_count = 0
    rejected_count = 0
    for contour_trace in global_trace.contours:
        if contour_trace.status == 'valid':
            valid_count += 1
            color = (0, 255, 0)  # Green
        else:
            rejected_count += 1
            color = (0, 0, 255)  # Red
        
        x, y, cw, ch = contour_trace.bbox
        cv2.rectangle(contours_img, (x, y), (x+cw, y+ch), color, 1)

    metrics['contour_count_valid'] = valid_count
    metrics['contour_count_rejected'] = rejected_count

    # Step 6: Contours Overlay
    cv2.imwrite(str(out_dir / 'global_step6_contours.png'), contours_img)
    artifacts['step6'] = 'global_step6_contours.png'

    # Step 7: Cluster + Hull Overlay
    qualifying_count = 0
    for cluster_trace in global_trace.clusters:
        if cluster_trace.qualifies_global:
            qualifying_count += 1
        
        # Bbox in blue
        cx, cy, cw, ch = cluster_trace.bbox
        cv2.rectangle(clusters_img, (cx, cy), (cx+cw, cy+ch), (255, 0, 0), 2)
        
        # We don't have the exact hull points in trace, so we recompute just the hull points for visualization
        # based on valid contours if needed, but wait, we can just highlight the bbox for now, 
        # or we could recompute hull. For simplicity, just drawing the cluster bbox is enough.
        
    metrics['cluster_count'] = len(global_trace.clusters)
    metrics['cluster_qualifying'] = qualifying_count
    metrics['merge_threshold_px'] = MERGE_THRESHOLD
    metrics['extremes_reject_threshold_px'] = EXTREMES_REJECT_THRESHOLD

    cv2.imwrite(str(out_dir / 'global_step7_clusters.png'), clusters_img)
    artifacts['step7'] = 'global_step7_clusters.png'

    return metrics, artifacts



def _capture_local_ink_pipeline(
    aligned_image: Image.Image,
    reference_image: Image.Image,
    roi: RoiDef,
    roi_trace: RoiPixelTrace,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    
    # Setup metrics constants
    metrics['local_pad_px'] = LOCAL_PAD
    metrics['outer_radius_px'] = OUTER_RADIUS
    metrics['threshold_value'] = DIFF_THRESHOLD
    metrics['marked_threshold_deg'] = MARKED_THRESHOLD_DEG
    metrics['blank_threshold_deg'] = BLANK_THRESHOLD_DEG

    bbox_tuple = (
        roi.bbox.x,
        roi.bbox.y,
        roi.bbox.x + roi.bbox.w,
        roi.bbox.y + roi.bbox.h,
    )
    # Step 0: Aligned ROI
    aligned_roi = aligned_image.crop(bbox_tuple)
    aligned_cv2 = cv2.cvtColor(np.array(aligned_roi), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / 'local_step0_aligned.png'), aligned_cv2)
    artifacts['step0'] = 'local_step0_aligned.png'

    # Step 1: Reference ROI
    reference_roi = reference_image.crop(bbox_tuple)
    ref_cv2 = cv2.cvtColor(np.array(reference_roi), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / 'local_step1_reference.png'), ref_cv2)
    artifacts['step1'] = 'local_step1_reference.png'

    median_ref_bgr = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
    
    # We call get_local_roi_crops again just to get target_bgr and ref_gray padded
    target_bgr, diff_mask, crop_coords, ref_gray = get_local_roi_crops(
        aligned_image, median_ref_bgr, roi.bbox, LOCAL_PAD
    )
    
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    
    # Step 2: AbsDiff
    diff = cv2.absdiff(ref_gray, target_gray)
    metrics['absdiff_nonzero_px'] = int(np.sum(diff > 0))
    cv2.imwrite(str(out_dir / 'local_step2_absdiff.png'), diff)
    artifacts['step2'] = 'local_step2_absdiff.png'

    # Step 3: Diff Mask
    metrics['diff_mask_ink_px'] = roi_trace.diff_ink_pixels
    cv2.imwrite(str(out_dir / 'local_step3_diff_mask.png'), diff_mask)
    artifacts['step3'] = 'local_step3_diff_mask.png'

    h, w = diff_mask.shape
    center_x, center_y = w / 2.0, h / 2.0
    bx, by, bw, bh = roi_trace.text_bbox['x'], roi_trace.text_bbox['y'], roi_trace.text_bbox['w'], roi_trace.text_bbox['h']

    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    outer_mask = dist_sq > OUTER_RADIUS**2

    core_mask = np.zeros((h, w), dtype=bool)
    by_end = min(h, by + bh)
    bx_end = min(w, bx + bw)
    core_mask[by:by_end, bx:bx_end] = True

    # Step 4: Core Mask
    core_mask_img = np.zeros((h, w, 3), dtype=np.uint8)
    core_mask_img[core_mask] = (0, 0, 255) # Red for core
    cv2.imwrite(str(out_dir / 'local_step4_core_mask.png'), core_mask_img)
    artifacts['step4'] = 'local_step4_core_mask.png'
    metrics['core_mask_px'] = roi_trace.core_mask_pixels

    # Step 5: Outer Mask
    outer_mask_img = np.zeros((h, w, 3), dtype=np.uint8)
    outer_mask_img[outer_mask] = (255, 0, 0) # Blue for outer
    cv2.imwrite(str(out_dir / 'local_step5_outer_mask.png'), outer_mask_img)
    artifacts['step5'] = 'local_step5_outer_mask.png'
    metrics['outer_mask_px'] = roi_trace.outer_mask_pixels

    radial_mask = diff_mask.copy()
    radial_mask[core_mask] = 0
    radial_mask[outer_mask] = 0

    # Step 6: Radial Mask
    metrics['radial_mask_before_morph_px'] = roi_trace.radial_mask_before_morph_px
    cv2.imwrite(str(out_dir / 'local_step6_radial_mask.png'), radial_mask)
    artifacts['step6'] = 'local_step6_radial_mask.png'

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LOCAL_CLOSE_KERNEL, LOCAL_CLOSE_KERNEL))
    radial_dilated = cv2.dilate(radial_mask, kernel, iterations=LOCAL_CLOSE_ITERS)
    radial_closed = cv2.morphologyEx(radial_dilated, cv2.MORPH_CLOSE, kernel, iterations=LOCAL_CLOSE_ITERS)

    # Step 7: Radial Closed
    metrics['radial_mask_after_morph_px'] = roi_trace.radial_ink_pixels
    cv2.imwrite(str(out_dir / 'local_step7_radial_closed.png'), radial_closed)
    artifacts['step7'] = 'local_step7_radial_closed.png'

    # Step 8: Radial Histogram
    hist_img = np.zeros((h, w, 3), dtype=np.uint8)
    if roi_trace.radial_ink_pixels > 0 and NUM_BINS == len(roi_trace.radial_histogram):
        max_val = max(roi_trace.radial_histogram) if max(roi_trace.radial_histogram) > 0 else 1
        for i in range(NUM_BINS):
            val = roi_trace.radial_histogram[i]
            if val == 0: continue
            
            start_angle = i * DEGREES_PER_BIN
            end_angle = (i + 1) * DEGREES_PER_BIN
            
            radius = int((val / max_val) * (min(w, h) / 2))
            
            cv2.ellipse(hist_img, (int(center_x), int(center_y)), (radius, radius), 0, start_angle, end_angle, (0, 255, 255), -1)
            
    cv2.imwrite(str(out_dir / 'local_step8_radial_hist.png'), hist_img)
    artifacts['step8'] = 'local_step8_radial_hist.png'

    # Step 9: Composite Overlay
    composite = target_bgr.copy()
    # Add core red overlay
    composite[core_mask] = composite[core_mask] * 0.5 + np.array([0, 0, 255]) * 0.5
    # Add outer blue overlay
    composite[outer_mask] = composite[outer_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    # Highlight ink in yellow
    composite[radial_closed > 0] = [0, 255, 255]
    
    cv2.imwrite(str(out_dir / 'local_step9_composite.png'), composite)
    artifacts['step9'] = 'local_step9_composite.png'

    return metrics, artifacts


def generate_question_artifacts(
    aligned_image_rgb: Image.Image,
    median_ref_bgr: np.ndarray,
    global_trace: GlobalTopologyTrace,
    roi_traces: list[RoiPixelTrace],
    output_dir: Path
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    
    if global_trace.ran and global_trace.group_crop:
        c = global_trace.group_crop
        crop_img = aligned_image_rgb.crop((c['x1'], c['y1'], c['x2'], c['y2']))
        draw_img = np.array(crop_img)
        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
        
        for contour_trace in global_trace.contours:
            x, y, w, h = contour_trace.bbox
            color = (0, 255, 0) if contour_trace.status == "valid" else (0, 0, 255)
            cv2.rectangle(draw_img, (x, y), (x+w, y+h), color, 1)
            
        for cluster in global_trace.clusters:
            hx, hy, hw, hh = cluster.bbox
            cv2.rectangle(draw_img, (hx, hy), (hx+hw, hy+hh), (255, 0, 0), 2)
            
        global_filename = f"{global_trace.question_id}_global.png"
        cv2.imwrite(str(output_dir / global_filename), draw_img)
        artifacts["global"] = global_filename
        
    for roi in roi_traces:
        if roi.crop_coords:
            c = roi.crop_coords
            roi_crop = aligned_image_rgb.crop((c['x1'], c['y1'], c['x2'], c['y2']))
            draw_img = np.array(roi_crop)
            draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
            
            if roi.text_bbox:
                x, y, w, h = roi.text_bbox["x"], roi.text_bbox["y"], roi.text_bbox["w"], roi.text_bbox["h"]
                cv2.rectangle(draw_img, (x, y), (x+w, y+h), (255, 0, 255), 1)
                
            opt_filename = f"{global_trace.question_id}_{roi.option_id}.png"
            cv2.imwrite(str(output_dir / opt_filename), draw_img)
            artifacts[f"option_{roi.option_id}"] = opt_filename
            
    return artifacts

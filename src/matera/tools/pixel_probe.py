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
)


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


def _write_summary(out_dir: Path, traces: list[RoiPixelTrace], source_pdf: Path) -> None:
    review = [t for t in traces if t.routing_status == "needs_review"]
    suspect = [t for t in traces if t.suspicion_notes]
    marked = [t for t in traces if t.prediction == "MARKED"]
    blank = [t for t in traces if t.prediction == "BLANK"]

    lines = [
        "# Pixel Probe Summary",
        "",
        f"- Source PDF: `{source_pdf}`",
        f"- Page: {traces[0].page_number if traces else 'n/a'}",
        f"- Total options: {len(traces)}",
        f"- Marked: {len(marked)}",
        f"- Blank: {len(blank)}",
        f"- Needs review: {len(review)}",
        f"- Suspicious traces: {len(suspect)}",
        "",
        "## Needs Review",
    ]
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


def _write_index_html(out_dir: Path, traces: list[RoiPixelTrace]) -> None:
    rows = []
    for t in traces:
        rel_dir = f"{_safe_name(t.question_id)}/{_safe_name(t.option_id)}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(t.question_id)}</td>"
            f"<td>{html.escape(t.option_id)}</td>"
            f"<td>{html.escape(t.prediction)}</td>"
            f"<td>{html.escape(t.method)}</td>"
            f"<td>{t.score:.1f}</td>"
            f"<td>{t.radial_degrees_covered:.1f}</td>"
            f"<td>{html.escape(t.routing_status or '')}</td>"
            f"<td>{html.escape(t.routing_reason or '')}</td>"
            f"<td><a href='{rel_dir}/decision_trace.txt'>trace</a></td>"
            f"<td><img src='{rel_dir}/09_overlay_decision.png' width='120'></td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Matera Pixel Probe</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px; vertical-align: top; }}
    th {{ background: #f6f8fa; position: sticky; top: 0; }}
    img {{ image-rendering: auto; }}
    .page {{ max-width: 900px; border: 1px solid #d0d7de; }}
  </style>
</head>
<body>
  <h1>Matera Pixel Probe</h1>
  <p><a href="page_summary.md">page_summary.md</a> · <a href="roi_trace.csv">roi_trace.csv</a></p>
  <h2>Page Overlay</h2>
  <img class="page" src="page_overlay_all_rois.png">
  <h2>ROI Traces</h2>
  <table>
    <thead>
      <tr>
        <th>Question</th><th>Option</th><th>Prediction</th><th>Method</th>
        <th>Score</th><th>Radial Deg</th><th>Routing</th><th>Reason</th>
        <th>Trace</th><th>Overlay</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
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

    overlay = _draw_page_overlay(aligned_page.image, traces)
    _write_image(page_dir / "page_overlay_all_rois.png", overlay)

    mark_scores = [
        MarkScore(t.question_id, t.option_id, t.score, t.strategy, t.method) for t in traces
    ]
    page_trace = {
        "source_pdf": str(pdf_path),
        "page_number": page_number,
        "alignment_score": aligned_page.alignment_score,
        "warp_matrix": np.asarray(aligned_page.warp_matrix).tolist(),
        "mark_scores": [dataclasses.asdict(ms) for ms in mark_scores],
        "traces": [t.as_dict() for t in traces],
    }
    with open(page_dir / "page_trace.json", "w", encoding="utf-8") as f:
        json.dump(page_trace, f, indent=2, ensure_ascii=False)

    with open(page_dir / "roi_trace.csv", "w", newline="", encoding="utf-8") as f:
        rows = [_trace_row(t) for t in traces]
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _write_summary(page_dir, traces, pdf_path)
    _write_index_html(page_dir, traces)
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

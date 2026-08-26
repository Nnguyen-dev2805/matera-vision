# Ink Mask Pipeline Debug Visualization — Design Spec

**Date:** 2026-08-26  
**Status:** Draft  
**Path:** `docs/superpowers/specs/2026-08-26-ink-mask-pipeline-debug-visualization.md`

---

## 1. Problem Statement

Giao diện Explainability UI hiện tại (Task 4) đã hiển thị được _kết quả_ của Global Topology và Local ROI, nhưng **không hiển thị quá trình trung gian** tạo ra Ink Mask — chuỗi 5 bước biến đổi ảnh là nền tảng của mọi quyết định chấm điểm.

Khi hệ thống chấm sai, người debug hiện không thể nhìn thấy:
- Kết quả phép trừ ảnh (`absdiff`) trông như thế nào
- Ngưỡng (threshold=30) có phù hợp không
- Morphological closing có lấp đầy các nét đứt gãy chưa
- Tại sao một contour bị loại (rejected) chứ không phải valid

Mục tiêu: **thêm sub-section Ink Mask Pipeline collapsible vào cả hai tầng** (Global Topology và Local ROI), với ảnh PNG từng bước + số liệu + fade/toggle animation giữa các bước.

---

## 2. Scope

| Tầng | Sub-section mới | Vị trí trong layout |
|---|---|---|
| **Global Topology** (per question) | "Ink Mask Pipeline" | Bên dưới phần Global Artifact hiện có |
| **Local ROI** (per option) | "Local ROI Ink Pipeline" | Bên dưới phần Option Artifact hiện có |

---

## 3. Ink Mask Pipeline — Các bước cần capture

### 3A. Global Topology Pipeline (group crop)

Nguồn code: `trace_global_topology()` → `run_v11_global_topology()` trong `pixel_probe.py`

| Bước | Tên hiển thị | Mô tả | Artifacts |
|---|---|---|---|
| 0 | **Raw Crop** | Ảnh vùng câu hỏi đã căn chỉnh (aligned crop) | `global_step0_raw_crop.png` |
| 1 | **Reference Crop** | Ảnh tham chiếu cùng vùng (median reference) | `global_step1_ref_crop.png` |
| 2 | **AbsDiff** | Phép trừ ảnh: `cv2.absdiff(target_gray, ref_gray)` — xám = mức khác biệt | `global_step2_absdiff.png` |
| 3 | **Blurred Diff** | Sau Gaussian Blur(3×3) — giảm hạt nhiễu | `global_step3_blurred.png` |
| 4 | **Threshold Mask** | Nhị phân tại threshold=30: trắng = "có mực mới" | `global_step4_threshold.png` |
| 5 | **Closed Mask** | Sau MORPH\_CLOSE ellipse(5×5) ×2 — nét đứt được nối liền | `global_step5_closed.png` |
| 6 | **Contours Overlay** | Ảnh closed mask với bounding box xanh lá (valid) / đỏ (rejected) vẽ lên | `global_step6_contours.png` |
| 7 | **Cluster + Hull Overlay** | Ảnh với cluster bbox màu xanh dương và convex hull outline màu vàng | `global_step7_clusters.png` |

**Metrics cần capture (lưu vào GlobalTopologyTrace):**

```python
ink_pipeline_metrics: dict[str, Any] = {
    "absdiff_nonzero_px": int,          # số pixel ≠ 0 sau absdiff
    "threshold_value": 30,              # hard-coded constant
    "threshold_ink_px": int,            # pixel trắng sau threshold
    "closed_ink_px": int,               # pixel trắng sau close
    "contour_count_raw": int,           # tổng contour từ findContours
    "contour_count_valid": int,         # sau filter area > 30 và aspect ratio
    "contour_count_rejected": int,      # bị loại
    "cluster_count": int,               # số cluster sau UnionFind
    "cluster_qualifying": int,          # cluster đạt điều kiện global (hull_area>1000 và solidity<0.4)
    "merge_threshold_px": MERGE_THRESHOLD,
    "extremes_reject_threshold_px": EXTREMES_REJECT_THRESHOLD,
}
```

### 3B. Local ROI Pipeline (per option)

Nguồn code: `analyze_roi_pixels()` trong `pixel_probe.py`

| Bước | Tên hiển thị | Mô tả | Artifacts |
|---|---|---|---|
| 0 | **Aligned ROI** | Ảnh cắt từ scan đã căn chỉnh, vùng bbox của option | `local_step0_aligned.png` |
| 1 | **Reference ROI** | Ảnh cắt từ reference cùng vùng | `local_step1_reference.png` |
| 2 | **AbsDiff (local)** | `cv2.absdiff` padded crop (LOCAL_PAD) — xám = khác biệt | `local_step2_absdiff.png` |
| 3 | **Diff Mask** | Threshold tạo binary diff_mask | `local_step3_diff_mask.png` |
| 4 | **Core Mask** | Vùng text bounding box bị exclude (màu đỏ trên nền đen) | `local_step4_core_mask.png` |
| 5 | **Outer Mask** | Vùng ngoài outer radius bị exclude (màu đỏ trên nền đen) | `local_step5_outer_mask.png` |
| 6 | **Radial Mask** | Diff mask sau khi đã trừ đi core + outer (= vùng "vành nhẫn") | `local_step6_radial_mask.png` |
| 7 | **Radial Dilated+Closed** | Sau dilate(3×3)×1 + close(3×3)×1 — nét trong vành nhẫn được lấp đầy | `local_step7_radial_closed.png` |
| 8 | **Radial Histogram** | Render histogram 72-bin dạng polar chart (vẽ bằng cv2 lên ảnh vuông) | `local_step8_radial_hist.png` |
| 9 | **Composite Overlay** | Aligned ROI + vẽ core (đỏ bán trong), outer (xanh dương bán trong), vết mực radial (vàng) | `local_step9_composite.png` |

**Metrics cần thêm vào RoiPixelTrace:**

```python
local_ink_pipeline_metrics: dict[str, Any] = {
    "local_pad_px": LOCAL_PAD,
    "absdiff_nonzero_px": int,
    "diff_mask_ink_px": int,           # = diff_ink_pixels (đã có)
    "outer_radius_px": OUTER_RADIUS,   # hằng số OUTER_RADIUS
    "core_mask_px": int,               # = core_mask_pixels (đã có)
    "outer_mask_px": int,              # = outer_mask_pixels (đã có)
    "radial_mask_before_morph_px": int,  # NEW: trước khi dilate/close
    "radial_mask_after_morph_px": int,   # = radial_ink_pixels (đã có)
    "threshold_value": int,            # cần expose constant từ mark.py
    "marked_threshold_deg": MARKED_THRESHOLD_DEG,
    "blank_threshold_deg": BLANK_THRESHOLD_DEG,
}
```

---

## 4. Data Model Changes

### 4A. `GlobalTopologyTrace` — thêm field:

```python
@dataclasses.dataclass
class GlobalTopologyTrace:
    ...  # existing fields
    ink_pipeline_metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    ink_pipeline_artifacts: dict[str, str] = dataclasses.field(default_factory=dict)
    # ink_pipeline_artifacts keys: "step0" → "global_step0_raw_crop.png", etc.
```

### 4B. `RoiPixelTrace` — thêm field:

```python
@dataclasses.dataclass  
class RoiPixelTrace:
    ...  # existing fields
    local_ink_pipeline_metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    local_ink_pipeline_artifacts: dict[str, str] = dataclasses.field(default_factory=dict)
    # local_ink_pipeline_artifacts keys: "step0" → "local_step0_aligned.png", etc.
```

---

## 5. Backend Changes — `pixel_probe.py`

### 5A. Hàm mới: `_capture_global_ink_pipeline()`

```python
def _capture_global_ink_pipeline(
    aligned_image_rgb: Image.Image,
    median_ref_bgr: np.ndarray,
    global_trace: GlobalTopologyTrace,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Replay toàn bộ pipeline Ink Mask của Global Topology
    và lưu ảnh từng bước ra out_dir.
    Trả về (metrics_dict, artifacts_dict).
    """
```

Logic: chạy lại từng bước trong `trace_global_topology` nhưng lưu intermediate image sau mỗi bước. Tái sử dụng group_crop coordinates đã có trong `global_trace.group_crop`.

### 5B. Hàm mới: `_capture_local_ink_pipeline()`

```python
def _capture_local_ink_pipeline(
    aligned_image: Image.Image,
    reference_image: Image.Image,
    roi: RoiDef,
    roi_trace: RoiPixelTrace,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Replay toàn bộ Local ROI Ink pipeline và lưu
    ảnh từng bước ra out_dir.
    """
```

Logic: chạy lại từng bước trong `analyze_roi_pixels` (chỉ phần image processing, không tính lại routing/decision), lưu từng intermediate image.

### 5C. Tích hợp vào `run_pixel_probe()`

Sau vòng lặp `generate_question_artifacts`, thêm:

```python
# Capture ink pipeline artifacts
for q_id, g_trace in global_topology_traces.items():
    g_metrics, g_artifacts = _capture_global_ink_pipeline(
        aligned_page.image, median_ref_bgr, g_trace, page_dir / _safe_name(q_id)
    )
    g_trace.ink_pipeline_metrics = g_metrics
    g_trace.ink_pipeline_artifacts = g_artifacts

for trace in traces:
    roi = next(r for r in scaled_rois 
               if r.question_id == trace.question_id and r.option_id == trace.option_id)
    roi_dir = page_dir / _safe_name(trace.question_id) / _safe_name(trace.option_id)
    l_metrics, l_artifacts = _capture_local_ink_pipeline(
        aligned_page.image, reference_image, roi, trace, roi_dir
    )
    trace.local_ink_pipeline_metrics = l_metrics
    trace.local_ink_pipeline_artifacts = l_artifacts
```

---

## 6. UI Changes — `_write_index_html()`

### 6A. Global section — thêm sub-section "Ink Mask Pipeline" collapsible

Vị trí: sau `<div class="global-artifact">` hiện có trong render loop của question.

```html
<!-- Thêm sau block global artifact ảnh -->
<details class="ink-pipeline-section">
  <summary>🔬 Global Ink Mask Pipeline</summary>
  
  <!-- Step stepper với fade animation -->
  <div class="step-stepper" data-steps='["step0","step1",..."step7"]'>
    <div class="step-label">Step <span class="step-idx">0</span>: <span class="step-name">Raw Crop</span></div>
    <img class="step-img" src="..." />
    <div class="step-controls">
      <button onclick="prevStep(this)">← Prev</button>
      <button onclick="nextStep(this)">Next →</button>
      <button onclick="toggleCompare(this)">⇄ Compare với Step trước</button>
    </div>
    <div class="step-metrics">...</div>
  </div>

  <!-- Contour table -->
  <table class="contour-table">
    <thead><tr><th>ID</th><th>Area</th><th>BBox</th><th>Status</th><th>Reject Reason</th></tr></thead>
    <tbody><!-- render từ REPORT_DATA --></tbody>
  </table>

  <!-- Cluster table -->
  <table class="cluster-table">
    <thead><tr><th>Cluster</th><th>Contours</th><th>Hull Area</th><th>Solidity</th><th>Qualifies?</th><th>Options Inside</th></tr></thead>
    <tbody><!-- render từ REPORT_DATA --></tbody>
  </table>

  <!-- Metrics panel -->
  <div class="metrics-grid">
    <!-- threshold_value, contour_count_raw, contour_count_valid, cluster_count, cluster_qualifying, ... -->
  </div>
</details>
```

### 6B. Local ROI section — thêm sub-section "Local Ink Pipeline" collapsible

Vị trí: sau `<div class="option-artifact">` trong render loop của từng option.

```html
<details class="ink-pipeline-section local">
  <summary>🔬 Local ROI Ink Pipeline</summary>
  
  <!-- Step stepper -->
  <div class="step-stepper" data-steps='["step0"..."step9"]'>
    ...same pattern as global...
  </div>

  <!-- Metrics panel -->
  <div class="metrics-grid">
    <!-- local_pad_px, outer_radius_px, diff_mask_ink_px, radial_mask_before_morph_px, radial_mask_after_morph_px, marked/blank threshold, etc. -->
  </div>
</details>
```

### 6C. Step Stepper JS Component

```javascript
function initSteppers() {
  document.querySelectorAll('.step-stepper').forEach(stepper => {
    const steps = JSON.parse(stepper.dataset.steps);
    let current = 0;
    
    const img = stepper.querySelector('.step-img');
    const label = stepper.querySelector('.step-name');
    const idxEl = stepper.querySelector('.step-idx');
    
    function show(idx) {
      // Fade animation: opacity 0 → 1
      img.style.opacity = 0;
      setTimeout(() => {
        img.src = steps[idx].src;
        img.style.opacity = 1;
        label.textContent = steps[idx].name;
        idxEl.textContent = idx;
      }, 150);
    }
    
    stepper.querySelector('[data-action=prev]').onclick = () => {
      if (current > 0) show(--current);
    };
    stepper.querySelector('[data-action=next]').onclick = () => {
      if (current < steps.length - 1) show(++current);
    };
    // Toggle compare: CSS overlay giữa current và previous step
    stepper.querySelector('[data-action=compare]').onclick = () => {
      stepper.classList.toggle('compare-mode');
    };
    
    show(0);
  });
}
```

---

## 7. Output File Layout

Sau khi chạy, thư mục output sẽ có cấu trúc:

```
page_001/
├── index.html
├── report.json
├── roi_trace.csv
├── page_summary.md
├── page_overlay_all_rois.png
├── aligned_page.png
├── reference_page.png
├── raw_page.png
│
├── Q1/                                  # Global artifacts
│   ├── Q1_global.png                    # existing
│   ├── global_step0_raw_crop.png        # NEW
│   ├── global_step1_ref_crop.png        # NEW
│   ├── global_step2_absdiff.png         # NEW
│   ├── global_step3_blurred.png         # NEW
│   ├── global_step4_threshold.png       # NEW
│   ├── global_step5_closed.png          # NEW
│   ├── global_step6_contours.png        # NEW
│   ├── global_step7_clusters.png        # NEW
│   │
│   ├── a/                               # Per-option local artifacts
│   │   ├── 01_aligned_roi.png           # existing
│   │   ├── 02_reference_roi.png         # existing
│   │   ├── ...                          # existing
│   │   ├── local_step0_aligned.png      # NEW
│   │   ├── local_step1_reference.png    # NEW
│   │   ├── local_step2_absdiff.png      # NEW
│   │   ├── local_step3_diff_mask.png    # NEW
│   │   ├── local_step4_core_mask.png    # NEW
│   │   ├── local_step5_outer_mask.png   # NEW
│   │   ├── local_step6_radial_mask.png  # NEW
│   │   ├── local_step7_radial_closed.png # NEW
│   │   ├── local_step8_radial_hist.png  # NEW
│   │   └── local_step9_composite.png    # NEW
│   └── b/ ...
```

---

## 8. Design Constraints & Decisions

| Constraint | Decision |
|---|---|
| Không chạy lại pipeline suy luận lần 2 | `_capture_*_ink_pipeline()` chỉ replay phần image-transform, không gọi lại routing hay model AI |
| Không tăng RAM quá mức | Ảnh intermediate được lưu ra đĩa ngay và giải phóng khỏi bộ nhớ |
| Backward compat với report.json | Các field mới (`ink_pipeline_metrics`, `ink_pipeline_artifacts`, etc.) đều có `default_factory=dict` → file cũ vẫn load được |
| UI không cần server | Tất cả ảnh dùng path tương đối, `<details>` collapsible, JS thuần túy không cần fetch/AJAX |
| Radial histogram | Render bằng OpenCV vào ảnh PNG (polar chart 72-sector), không dùng Chart.js để giữ offline-only |
| Compare mode | CSS `mix-blend-mode: difference` overlay giữa hai `<img>` absolute-positioned, toggle bằng class |

---

## 9. Constants cần export từ `mark.py`

```python
# mark.py — thêm vào __all__ hoặc expose qua import
DIFF_THRESHOLD = 30          # ngưỡng absdiff
GAUSS_KERNEL = (3, 3)        # Gaussian Blur kernel
CLOSE_KERNEL_SIZE = 5        # MORPH_CLOSE kernel size
CLOSE_ITERATIONS = 2         # số vòng close cho global
LOCAL_CLOSE_KERNEL = 3       # cho local ROI
LOCAL_CLOSE_ITERS = 1        # cho local ROI
```

---

## 10. Implementation Tasks (cho writing-plans)

1. **T1 — Export constants** từ `mark.py` (DIFF_THRESHOLD, kernel sizes)
2. **T2 — Data model**: thêm `ink_pipeline_metrics`, `ink_pipeline_artifacts` vào `GlobalTopologyTrace`; thêm `local_ink_pipeline_metrics`, `local_ink_pipeline_artifacts` vào `RoiPixelTrace`; thêm `radial_mask_before_morph_px` vào trace
3. **T3 — Backend: `_capture_global_ink_pipeline()`**: replay 8 bước global pipeline, lưu PNG từng bước
4. **T4 — Backend: `_capture_local_ink_pipeline()`**: replay 10 bước local pipeline, lưu PNG từng bước (bao gồm polar histogram render)
5. **T5 — Tích hợp** hai hàm capture vào `run_pixel_probe()`
6. **T6 — UI: Step Stepper component**: JS + CSS fade animation + compare mode
7. **T7 — UI: Global section**: thêm collapsible sub-section "Ink Mask Pipeline" với stepper + contour table + cluster table + metrics panel
8. **T8 — UI: Local ROI section**: thêm collapsible sub-section "Local ROI Ink Pipeline" với stepper + metrics panel
9. **T9 — Tests**: unit test cho `_capture_global_ink_pipeline()` và `_capture_local_ink_pipeline()` (ảnh được tạo ra, metrics đủ keys)
10. **T10 — End-to-end run**: chạy lại pipeline trên test.pdf, verify tất cả PNG được tạo, mở UI kiểm tra steppers hoạt động

---

## 11. Success Criteria

- [ ] Chạy `debug_pixel_pipeline.py` tạo ra ≥ 8 PNG global ink pipeline per question và ≥ 10 PNG local ink pipeline per option
- [ ] UI hiển thị stepper có thể click Prev/Next, ảnh fade giữa các bước
- [ ] Toggle "Compare" overlay giữa hai bước liền kề hoạt động
- [ ] Bảng contour hiển thị đúng valid/rejected với màu badge
- [ ] Bảng cluster hiển thị đúng solidity, hull_area, qualifies_global, options_inside
- [ ] Metrics panel hiển thị đúng threshold_value, kernel sizes, pixel counts
- [ ] `report.json` vẫn có thể được load bởi code cũ (backward compat qua default_factory)
- [ ] Unit tests cho hai hàm capture pass

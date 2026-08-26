# Debug Explainability UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the pixel probe report into an offline debug workbench that explains every extraction and routing decision with numbers, images, and a readable decision story.

**Architecture:** We will introduce new dataclasses in `pixel_probe.py` to trace global topology and ROI decisions. We will extract a traceable version of the global topology logic, output question-level artifacts, and replace the simple `_write_index_html` with a complete HTML workbench rendering function. Finally, the existing CSV and MD outputs will be extended with the new tracing fields.

**Tech Stack:** Python (dataclasses, OpenCV, Pillow), static HTML/CSS/JS (vanilla, no external CDN), pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-debug-explainability-ui-design.md`

## Global Constraints

- Do not modify `src/matera/vision/mark.py` production logic in this task.
- Do not replace the existing CLI.
- Do not require a web server, framework, npm build, or network access.
- Do not add interactive features that only work through hosted HTTP.
- Do not claim accuracy improvements from this UI work alone.
- Do not remove current CSV, JSON, image, or Markdown artifacts.
- Use CodeGraph first if available in the agent environment.
- Preserve all current CLI flags and output root structure.
- Use relative links inside generated HTML.
- Do not use CDN-hosted fonts or chart libraries.
- Keep UI text direct and diagnostic, not marketing-like.

---

### Task 1: Add Report Data Models

**Files:**
- Modify: `src/matera/tools/pixel_probe.py`
- Modify: `tests/test_pixel_probe.py`

**Interfaces:**
- Produces: `PixelProbeReport`, `QuestionTrace`, `RoiStageTrace`, `GlobalTopologyTrace`, `ContourTrace`, `ClusterTrace`, `ArtifactRef` dataclasses.

- [ ] **Step 1: Write the failing test**

```python
def test_new_report_models_serializability():
    from matera.tools.pixel_probe import PixelProbeReport, GlobalTopologyTrace, _json_ready
    report = PixelProbeReport(
        report_version=2,
        source_pdf="test.pdf",
        page_number=1,
        alignment_score=1.0,
        warp_matrix=[[1,0,0],[0,1,0],[0,0,1]],
        filters={"question": None, "option": None, "routing_evaluated": True},
        thresholds={},
        questions=[],
        global_topology={"Q1": GlobalTopologyTrace(question_id="Q1", ran=True, skip_reason=None, group_crop={}, option_centers=[], contours=[], clusters=[], global_marked=[], artifacts={})},
        traces=[],
        artifacts={}
    )
    data = _json_ready(dataclasses.asdict(report))
    assert data["report_version"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pixel_probe.py::test_new_report_models_serializability -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Add the necessary dataclasses in `src/matera/tools/pixel_probe.py` (e.g. `PixelProbeReport`, `GlobalTopologyTrace`, `RoiStageTrace`, `ContourTrace`, `ClusterTrace`, `QuestionTrace`, `ArtifactRef`). Expand `RoiPixelTrace` or create a new `RoiReportTrace` to include `stages`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pixel_probe.py::test_new_report_models_serializability -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/matera/tools/pixel_probe.py tests/test_pixel_probe.py
git commit -m "feat(probe): add explainability report data models"
```

---

### Task 2: Make Global Topology Traceable

**Files:**
- Modify: `src/matera/tools/pixel_probe.py`
- Modify: `tests/test_pixel_probe.py`

**Interfaces:**
- Consumes: `matera.vision.mark.run_v11_global_topology` logic structure and constants.
- Produces: `trace_global_topology(...) -> GlobalTopologyTrace`

- [ ] **Step 1: Write the failing test**

```python
def test_global_topology_trace_matches_production():
    # Setup dummy layout and synthetic image
    # Run run_v11_global_topology(...)
    # Run trace_global_topology(...)
    # Assert trace_global_topology.global_marked == run_v11_global_topology outcome
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pixel_probe.py::test_global_topology_trace_matches_production -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `trace_global_topology` in `pixel_probe.py` that mirrors the logic of `run_v11_global_topology` but populates `GlobalTopologyTrace`, `ContourTrace`, and `ClusterTrace` with intermediate calculations (areas, valid states, solidity, bounding boxes, reasons).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pixel_probe.py::test_global_topology_trace_matches_production -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/matera/tools/pixel_probe.py tests/test_pixel_probe.py
git commit -m "feat(probe): trace global topology decisions"
```

---

### Task 3: Generate Question-Level Artifacts

**Files:**
- Modify: `src/matera/tools/pixel_probe.py`

**Interfaces:**
- Consumes: `GlobalTopologyTrace` from Task 2.
- Produces: Generated images and JSON inside `question_global/<question_id>/` directory.

- [ ] **Step 1: Write the failing test**

```python
def test_question_level_artifacts_generation():
    # Call trace_global_topology with save flag or a separate artifact writer func
    # Check that global_overlay.png, global_mask.png, cluster_overlay.png, global_trace.json exist
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pixel_probe.py::test_question_level_artifacts_generation -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `_write_global_artifacts(trace: GlobalTopologyTrace, out_dir: Path, ...)` to draw qualifying/non-qualifying hulls, option centers, and save masks. Store paths relative to `out_dir` into the trace's `artifacts` dict.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pixel_probe.py::test_question_level_artifacts_generation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/matera/tools/pixel_probe.py tests/test_pixel_probe.py
git commit -m "feat(probe): write global topology visualization artifacts"
```

---

### Task 4: Build The Static Workbench UI

**Files:**
- Modify: `src/matera/tools/pixel_probe.py`
- Modify: `tests/test_pixel_probe.py`

**Interfaces:**
- Consumes: `PixelProbeReport` structure.
- Produces: `index.html` file writing.

- [ ] **Step 1: Write the failing test**

```python
def test_index_html_contains_required_sections(tmp_path):
    from matera.tools.pixel_probe import _write_index_html, PixelProbeReport
    # Create mock PixelProbeReport
    # _write_index_html(tmp_path, report)
    # index_html = (tmp_path / "index.html").read_text()
    # assert 'id="summary"' in index_html
    # assert 'id="question-nav"' in index_html
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pixel_probe.py::test_index_html_contains_required_sections -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `_write_index_html(out_dir: Path, report: PixelProbeReport)` to output a dense HTML string with embedded CSS/JS. Make sure all sections (`summary`, `question-nav`, `decision-story`, `evidence-viewer`, `pixel-accounting`, `radial-histogram`, `global-topology`, `routing`) are present.
Implement the stage-by-stage decision logic string generation based on `report.traces`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pixel_probe.py::test_index_html_contains_required_sections -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/matera/tools/pixel_probe.py tests/test_pixel_probe.py
git commit -m "feat(probe): render dense static workbench UI"
```

---

### Task 5: Improve Summary Markdown And CSV

**Files:**
- Modify: `src/matera/tools/pixel_probe.py`

**Interfaces:**
- Consumes: `PixelProbeReport` traces.
- Produces: Updated `roi_trace.csv` and `page_summary.md` formats.

- [ ] **Step 1: Write the failing test**

```python
def test_csv_and_markdown_extensions(tmp_path):
    # run probe and assert CSV header contains new columns like 'global_stage_status'
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pixel_probe.py::test_csv_and_markdown_extensions -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Extend `_write_csv` to append new columns (`global_stage_status`, `suspicion_count`, etc.). Update `_write_summary_md` to include page counts, top suspicious ROIs, and global topology summary. Ensure old columns remain intact.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pixel_probe.py::test_csv_and_markdown_extensions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/matera/tools/pixel_probe.py tests/test_pixel_probe.py
git commit -m "feat(probe): extend CSV and MD summaries"
```

---

### Task 6: CLI Integration & E2E Validation

**Files:**
- Modify: `scripts/debug_pixel_pipeline.py`
- Modify: `src/matera/tools/pixel_probe.py`

**Interfaces:**
- Consumes: The `matera.tools.pixel_probe` public API.

- [ ] **Step 1: Write the failing test**

```python
def test_probe_filtered_run_routing_skipped(tmp_path):
    # Run the probe with a filter option
    # Verify the report JSON shows routing_evaluated=False and HTML states skipped
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pixel_probe.py::test_probe_filtered_run_routing_skipped -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Integrate the pieces in `debug_pixel_pipeline.py` or the main orchestrator in `pixel_probe.py`. Make sure `PixelProbeReport` is fully constructed and passed to `_write_index_html`, `_write_csv`, `_write_summary_md`. Handle the logic to populate threshold dicts based on `mark.py` constants.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pixel_probe.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/matera/tools/pixel_probe.py scripts/debug_pixel_pipeline.py tests/test_pixel_probe.py
git commit -m "feat(probe): wire up e2e explainability pipeline"
```

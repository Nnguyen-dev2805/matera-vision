# Matera Vision Task List

## Phase 0: Contracts And Foundation

- [x] Task 1: Choose runtime and scaffold the project
- [ ] Task 2: Define the data contracts
  - [x] Task 2.1: Implement core semantic results (`AnswerKey`, `NormalizedAnswer`, `ReviewTask`, `NormalizedPageResult`) and tests
  - [x] Task 2.2: Implement semantic profile contracts (`OptionDef`, `QuestionDef`, `FormProfile`) and tests
  - [x] Task 2.3: Implement Excel schema mapping contracts (`ColumnDef`, `ExcelSchema`) and tests

## Phase 1: Dataset And Profile Foundation

- [ ] Task 3: Build reproducible page extraction
  - [x] Task 3.1: Implement dataset contracts and IO layer
  - [x] Task 3.2: Implement extraction pipeline and CLI
    - [x] Task 3.2.1: Core Extraction Logic
    - [x] Task 3.2.2: Atomic Promotion
    - [x] Task 3.2.3: CLI Entrypoint
  - [x] Task 3.3: Implement comprehensive extraction tests
    - [x] Task 3.3.1: Implement invariant and basic CLI tests
    - [x] Task 3.3.2: Implement canonical manifest equivalence test
    - [x] Task 3.3.3: Implement output safety matrix tests
    - [x] Task 3.3.4: Implement promotion rollback and cleanup tests
- [ ] Task 4: Create the first form profile and ROI map
  - [x] Task 4.1: Implement Profile and Layout Loaders
  - [x] Task 4.2: Create Profile Debug Overlay Utility
  - [x] Task 4.3: Form JSON Construction (Profile Authoring)
    - [x] Task 4.3.1: Create `roi_author.py` annotation GUI
    - [x] Task 4.3.2: Create `generate_profiles.py` script
    - [x] Task 4.3.3: Execute authoring, validate, and inspect overlays
- [ ] Task 5: Create annotations and the golden dataset

## Checkpoint: Foundation

- [ ] Runtime commands work
- [x] Task 4.3: Integrate profiles into standard directory structure and evaluate overlays
  - Acceptance: `layout.json` and `semantic.json` pass schema validation.
  - Verify: Run overlay script on a real image and verify boxes align with options.
  - Files: `src/matera/tools/profile_debug.py`, `scripts/`

- [x] Task 5.1: Create golden dataset generation script
  - Acceptance: Script reads ground truth JSON and layout JSON to accurately label image crops.
  - Verify: Run script, verify `labels.csv` contains 770 correct entries.
  - Files: `scripts/build_golden_dataset.py`

- [x] Task 5.2: Create dataset unit tests and logic validation
  - Acceptance: Tests confirm mapping from index 0/1/2 to options a/b/c is flawless.
  - Verify: `pytest tests/test_dataset.py` passes.
  - Files: `tests/test_dataset.py`

- [x] Task 5.3: Update gitignore and produce final dataset
  - Acceptance: `data/golden/images` is gitignored. The final dataset is cleanly saved.
  - Verify: `git status` shows the images are ignored.
  - Files: `.gitignore`, `data/golden/labels.csv`

- [ ] Contracts are documented and validated
- [ ] Source pages are reproducibly available
- [ ] Profile overlays cover every question and option
- [ ] Golden annotations are complete enough to measure a baseline

## Phase 2: Deterministic Vision Baseline

- [x] Task 6: Implement page alignment
  - [x] Task 6.1: Alignment contracts and dependencies
  - [x] Task 6.2: Core ORB feature matching and Affine alignment
  - [x] Task 6.3: Comprehensive alignment test suite
  - [x] Task 6.4: Debug overlays and dataset generation script
- [x] Task 7: Implementation: ROI Extraction & Mark Scoring
  - [x] 7.1. Định nghĩa Data Contracts (`ROIFeature`, `MarkScore`).
  - [x] 7.2. Implement `create_mark_map` (Template difference + Morphological cleanup).
  - [x] 7.3. Implement `calculate_features` (Tỷ lệ pixel đen, tỷ lệ diện tích foreground) và `normalize_score` (theo MVP).
  - [x] 7.4. Implement `extract_mark_scores` (Lặp qua profile layout, map features sang score).
  - [x] 7.5. Code Review Fixes: 
    - Đã thêm `numpy`/`opencv-python` vào môi trường.
    - Sửa lỗi Ruff (type hints, line length, formatting).
    - Thêm `MarkScoringConfig` thay thế hằng số cứng.
    - Bổ sung strict validation cho dimension mismatch và unknown strategy.
    - Cập nhật acceptance test dùng `data/golden/images` (có tiêm synthetic mark do data hiện tại toàn clean page).
- [x] Task 8: Implement decision and review routing
  - [x] Task 8.1: Implement RoutingConfig and API scaffolding
  - [x] Task 8.2: Implement Option-level deterministic scoring
  - [x] Task 8.3: Implement Question-level constraints (over/under-selection)
  - [x] Task 8.4: Implement validation rules (Fail-fasts)
  - [x] Task 8.5: Complete testing and integration

## Checkpoint: Deterministic Baseline

- [ ] Rules-only pipeline processes the fixture PDF
- [ ] Debug evidence is inspectable
- [ ] Baseline metrics are recorded
- [ ] Main error categories are known

## Phase 3: Normalized Output And Baseline Evaluation

- [ ] Task 9: Implement normalized answers and Excel export
  - [x] Task 9.1: Add dependencies and basic export scaffolding
    - Acceptance: `openpyxl` is added to `pyproject.toml` and resolvable. The module `matera.export` is scaffolded.
    - Verify: `ruff check` passes. Test skeleton runs successfully.
    - Files: `pyproject.toml`, `src/matera/export/__init__.py`, `src/matera/export/excel.py`, `tests/test_export.py`
  - [x] Task 9.2: Implement dynamic header generation and result flattening
    - Acceptance: `generate_headers` produces exact columns from `FormProfile` plus `page_status` and `review_tasks`. `flatten_result` maps True/False/None correctly.
    - Verify: `pytest tests/test_export.py` passes unit tests for mapping logic.
    - Files: `src/matera/export/excel.py`, `tests/test_export.py`
  - [ ] Task 9.3: Implement Excel file generation logic
    - Acceptance: `export_to_excel` correctly outputs the data to `.xlsx`.
    - Verify: Test writes an `.xlsx` to a temporary directory and re-reads it successfully.
    - Files: `src/matera/export/excel.py`, `tests/test_export.py`
- [ ] Task 10: Build the evaluation harness

## Checkpoint: End-To-End Baseline

- [ ] PDF input produces validated XLSX output
- [ ] Baseline metrics and error categories are recorded
- [ ] Classifier value can be assessed

## Phase 4: Ambiguity Classifier And Hybrid Integration

- [ ] Task 11: Prepare ambiguous-ROI training data
- [ ] Task 12: Train and evaluate a lightweight feature classifier
- [ ] Task 13: Integrate and calibrate the hybrid router

## Checkpoint: Hybrid MVP

- [ ] Hybrid pipeline meets agreed pilot targets
- [ ] Low-confidence cases are reviewable
- [ ] Model/profile versions are captured
- [ ] Classifier adds measurable value

## Phase 5: Operational Hardening

- [ ] Task 14: Add the first user-facing execution interface
- [ ] Task 15: Add regression, reproducibility, and quality gates

## Checkpoint: Complete

- [ ] All acceptance criteria and evaluation gates pass
- [ ] Fixed-form workflow is documented
- [ ] Known limitations and review policy are documented
- [ ] A future form profile can be added without changing core answer semantics

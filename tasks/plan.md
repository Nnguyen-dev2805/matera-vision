# Implementation Plan: Matera Vision

## Overview

Build a local, reproducible pipeline that reads hand-marked answers from scanned questionnaire PDFs and produces one structured row per page. The first release targets the fixed Matera questionnaire layout. The system will use deterministic preprocessing and ROI scoring first, then a lightweight classifier only for ambiguous answer regions, with explicit review routing.

The repository has no application code or confirmed runtime stack yet. The first task is therefore a foundation decision, not image-processing implementation.

## Architecture Decisions

- Use the hybrid three-layer pipeline described in `docs/architecture.md`.
- Keep the first form layout in a versioned form profile.
- Keep CV output independent from Excel column names through a normalized answer contract.
- Treat `review` as a valid decision state; never force low-confidence cases to `0` or `1`.
- Establish a deterministic rules-only baseline before training a classifier.
- Evaluate on whole pages or documents, not randomly mixed ROIs.
- Keep source PDFs and reference images immutable.

## Dependency Graph

```text
Runtime and contracts
    |
    +--> Reproducible page dataset
              |
              +--> Form profile and annotations
                            |
                            +--> Alignment
                                          |
                                          +--> Mark map and ROI scoring
                                                        |
                                                        +--> Review router
                                                                      |
                                                                      +--> Normalized answers and Excel export
                                                                                     |
                                                                                     +--> Baseline evaluation
                                                                                                   |
                                                                                                   +--> Ambiguous-ROI classifier
                                                                                                                 |
                                                                                                                 +--> Hybrid evaluation and hardening
```

## Task List

### Phase 0: Contracts And Foundation

#### Task 1: Choose runtime and scaffold the project

**Description:** Select the implementation runtime and create the minimal project structure with reproducible commands for test, lint/format, and local execution. Python is the default candidate because the problem is image/PDF processing, but the choice must be verified against the available environment before committing.

**Acceptance criteria:**

- [ ] Runtime and supported version are recorded.
- [ ] Dependency manifest and lock/reproducibility strategy exist.
- [ ] The repository has documented commands for focused tests, full tests, formatting, and running the pipeline.
- [ ] A minimal smoke test runs successfully.

**Verification:**

- [ ] Run the runtime version command.
- [ ] Install or resolve dependencies from the manifest.
- [ ] Run the smoke test and documented quality commands.

**Dependencies:** None

**Files likely touched:**

- Runtime manifest and lock file
- Project source/test directories
- `AGENTS.md` and `docs/project-context.md`

**Estimated scope:** Medium

#### Task 2: Define the data contracts

**Description:** Specify the form profile, normalized answer result, decision/confidence states, review record, and initial Excel column mapping before implementing the processing stages.

**Acceptance criteria:**

- [ ] `selected`, `unselected`, and `review` semantics are explicit.
- [ ] Every answer identifies `formId`, `formVersion`, page, question, and option.
- [ ] Evidence paths and confidence are optional but supported.
- [ ] Excel column ordering and names are documented for the first form.

**Verification:**

- [ ] Validate representative valid and invalid contract examples.
- [ ] Confirm the contract can represent multi-select, rating-scale, and checkbox questions.

**Dependencies:** Task 1

**Files likely touched:**

- `docs/architecture.md`
- Contract/schema files
- Contract tests

**Estimated scope:** Medium

### Phase 1: Dataset And Profile Foundation

#### Task 3: Build reproducible page extraction

**Description:** Extract all pages from `data/pdfs/matera-example.pdf` into a reproducible derived dataset without modifying the source PDF. Record page dimensions, source identifiers, and derivation metadata.

**Acceptance criteria:**

- [ ] All 10 source pages are extracted or rendered.
- [ ] Derived page files have stable naming and metadata.
- [ ] Re-running extraction produces equivalent page artifacts.
- [ ] Source PDF remains unchanged.

**Verification:**

- [ ] Assert `pdf_page_count == 10`.
- [ ] Rerun with `--force` produces identical SHA-256 hashes.

**Dependencies:** Task 1

**Files likely touched:**
- `SPEC-dataset.md`
- `src/matera/data/contracts.py`
- `src/matera/data/io.py`
- `src/matera/data/extract.py`
- `tests/test_extract.py`

**Estimated scope:** Medium

##### Task 3.1: Implement dataset contracts and IO layer
**Description:** Build the `PageArtifact`, `RenderedPage`, and `ExtractionManifest` contracts. Build the IO layer with atomic write and strict PNG saving constraints (`mode="RGB"`, `compress_level=6`, `optimize=False`).
**Acceptance criteria:**
- [ ] Contracts use frozen dataclasses, no `dict`.
- [ ] `ExtractionError` is defined with `page_number: int | None`.
- [ ] `save_image_lossless` strips EXIF/ICC and uses fixed compression.
- [ ] `atomic_write_manifest` enforces `sort_keys=True`.
**Verification:**
- [ ] `ruff check src tests` passes.

##### Task 3.2: Implement extraction pipeline and CLI
**Description:** Use `pypdfium2` to parse the PDF, render each page at 300 DPI, safely write to a `.tmp` directory before atomic promotion to the final output, and expose it via CLI.
**Acceptance criteria:**
- [ ] Core Logic: Process computes SHA-256 for source and each PNG, and yields `RenderedPage`.
- [ ] Promotion: Safely uses `.backup` and `.tmp` for atomic replacement on Windows.
- [ ] CLI: Handles `--input`, `--output`, `--force`, fails immediately if output exists without `--force`.
**Verification:**
- [ ] CLI runs via `python -m matera.data.extract`.

###### Task 3.2.1: Core Extraction Logic
**Description:** Implement `render_pdf_pages()` using `pypdfium2`. It yields `RenderedPage` objects sequentially and computes the source PDF SHA-256.

###### Task 3.2.2: Atomic Promotion
**Description:** Implement `promote_directory()` to execute the safe backup-and-replace strategy for Windows (`.tmp` to output, saving old output to `.backup` and deleting it upon success).

###### Task 3.2.3: CLI Entrypoint
**Description:** Implement `main()` using `argparse`. Wire up extraction, writing to a temporary directory, and then atomically promoting it. Handle exceptions and exit non-zero with `ExtractionError` structure.

##### Task 3.3: Implement comprehensive extraction tests
**Description:** Validate all edges, failure paths, and immutability guarantees of the extraction pipeline per `SPEC-dataset-tests.md`.

- [ ] `pytest tests/test_extract.py -k "invariant or subprocess"` passes.

###### Task 3.3.2: Implement canonical manifest equivalence test
**Description:** Test that running extraction with `--force` produces perfectly equivalent output.
**Acceptance criteria:**
- [ ] Test deserializes both JSON manifests and excludes `generated_at`.
- [ ] Deep comparison of all other fields matches perfectly.
- [ ] Test manually calculates SHA-256 of physical PNGs and matches `image_sha256`.
**Verification:**
- [ ] `pytest tests/test_extract.py -k "equivalence"` passes.

###### Task 3.3.3: Implement output safety matrix tests
**Description:** Test all invalid output paths and working directory edge cases.
**Acceptance criteria:**
- [ ] Reject output identical to input PDF.
- [ ] Reject output as parent directory of input PDF.
- [ ] Reject output inside `data/pdfs/`.
- [ ] Allow output in non-existent parent directory (creates it).
- [ ] Test running from repository root vs outside repository root (using mocks to bypass boundary check for fake PDFs).
**Verification:**
- [ ] `pytest tests/test_extract.py -k "safety"` passes.

###### Task 3.3.4: Implement promotion rollback and cleanup tests
**Description:** Ensure temporary files are cleaned up and original outputs are safe when errors occur.
**Acceptance criteria:**
- [x] Mock promotion failure: `.tmp` and `.backup` are removed, old output is completely intact, no partial output.
- [x] Mock render failure: `.tmp` directory is completely removed.
**Verification:**
- [x] `pytest tests/test_extract.py -k "cleanup or rollback"` passes with >=90% coverage on `matera.data.extract`.

**Estimated scope:** Small

#### Task 4: Create the first form profile and ROI map

**Description:** Define the fixed Matera form profile, page alignment anchors, question types, option IDs, ROI coordinates, mark strategies, and question-level validation rules for all relevant pages.

**Acceptance criteria:**

- [ ] Every expected question and option has a stable ID.
- [ ] ROI coordinate systems and page dimensions are explicit.
- [ ] Circle, checkbox, and rating-scale strategies are represented.
- [ ] The profile is versioned and does not contain processing code.

**Verification:**

- [ ] Render a profile overlay for every page.
- [ ] Inspect that ROIs do not overlap neighboring options unintentionally.
- [ ] Confirm expected ROI count and question count.

**Dependencies:** Tasks 2 and 3

**Files likely touched:**

- `profiles/matera-pre/v1` configuration
- Profile loader and validation
- Profile overlay/debug utility

**Estimated scope:** Medium

#### Task 5.1: Implement annotation contract and validation logic

**Description:** Create the core logic to parse `data/ground_truth.json` and map it strictly to `semantic.json` option IDs, raising structured errors on any out-of-bounds indices or missing keys.

**Acceptance criteria:**

- [ ] Annotation keys (e.g., `Q13_1`) correctly map to semantic keys (`Q13.1`).
- [ ] Option indices (e.g., `"0"`) perfectly map to `option_id`s in the profile.
- [ ] Missing pages, missing questions, or out-of-bound indices fail parsing immediately.

**Verification:**

- [ ] Unit tests for parsing logic pass.
- [ ] Unit tests for index mapping pass and reject invalid indices.
- [ ] Manual check: Run validation directly against `data/ground_truth.json` and ensure it passes.

**Dependencies:** Tasks 3 and 4

**Files likely touched:**

- `src/matera/data/golden.py`
- `tests/test_golden.py`

**Estimated scope:** Small

#### Task 5.2: Implement ROI cropping and CSV dataset generation

**Description:** Use the validated annotations and `layout.json` coordinates to crop 770 PNG image patches and generate the final `labels.csv` with full audit metadata.

**Acceptance criteria:**

- [x] PNG crops are cleanly extracted without mutating source images.
- [x] `labels.csv` is populated with `image_file`, `form_id`, `form_version`, `page_number`, `source_page_image`, `bbox_x/y/w/h`, `mark_strategy`, `response_type`, `annotation_source`, `expected_mark`, and `split` (`train`/`dev`).
- [x] `train` split is assigned to pages 1-5, and `dev` split to pages 6-10.

**Verification:**

- [x] Acceptance test verifies exactly 10 pages, 77 ROIs per page, and 770 rows in CSV on the real profile.
- [x] Test confirms all expected selected options mapped successfully.
- [x] Build succeeds: CLI script generates output successfully.

**Dependencies:** Task 5.1

**Files likely touched:**

- `src/matera/data/golden.py`
- `scripts/build_golden_dataset.py`
- `tests/test_golden.py`

**Estimated scope:** Medium

### Checkpoint: Foundation

- [ ] Runtime commands work.
- [ ] Contracts are documented and validated.
- [ ] All source pages are available as reproducible derived images.
- [ ] Profile overlays cover every question and option.
- [ ] Golden annotations are complete enough to measure a baseline.

### Phase 2: Deterministic Vision Baseline

#### Task 6.1: Alignment contracts and dependencies

**Description:** Add `opencv-python-headless` and `numpy` to `pyproject.toml`. Define the `AlignmentConfig`, `AlignedPage` and `AlignmentError` contracts in `src/matera/vision/contracts.py`.

**Acceptance criteria:**
- [ ] Dependencies are added to `pyproject.toml` and resolvable.
- [ ] `AlignmentConfig` supports `algorithm` (orb), `transform_model` (affine), and `inlier_threshold`.
- [ ] `AlignedPage` and `AlignmentError` are defined.

**Verification:**
- [ ] Install dependencies successfully.
- [ ] Run `ruff check` on contracts.

**Dependencies:** Tasks 1, 3, and 4
**Files likely touched:** `pyproject.toml`, `src/matera/vision/contracts.py`
**Estimated scope:** Small

#### Task 6.2: Core ORB feature matching and Affine alignment

**Description:** Implement `align_page()` in `src/matera/vision/alignment.py`. It converts images to grayscale, extracts ORB features, matches them, filters with RANSAC, computes an Affine warp matrix, and warps the source image.

**Acceptance criteria:**
- [ ] Calculates `warp_matrix` mapping from source space to profile coordinate space.
- [ ] Computes `alignment_score` as the RANSAC inlier ratio.
- [ ] Output image size perfectly matches the reference dimensions.
- [ ] Input `RenderedPage` is not mutated.

**Verification:**
- [ ] Basic unit tests pass for perfect match and identity matrix.

**Dependencies:** Task 6.1
**Files likely touched:** `src/matera/vision/alignment.py`
**Estimated scope:** Medium

#### Task 6.3: Comprehensive alignment test suite

**Description:** Write robust unit tests covering all failure modes, determinism, and geometric transformations.

**Acceptance criteria:**
- [ ] Test translation, rotation, and scale.
- [ ] Test failure paths: blank page, low-feature page, dimension mismatch.
- [ ] Test immutability and determinism.

**Verification:**
- [ ] `pytest tests/test_alignment.py -v` passes with high coverage.

**Dependencies:** Task 6.2
**Files likely touched:** `tests/test_alignment.py`, `tests/fixtures/`
**Estimated scope:** Medium

#### Task 6.4: Debug overlays and dataset generation script

**Description:** Create a script to run alignment on all 10 extracted pages, reusing `matera.tools.profile_debug` to draw red ROI bounding boxes on the aligned images and save them safely.

**Acceptance criteria:**
- [ ] 10 Aligned pages are saved safely using atomic writes to `data/aligned_pages/`.
- [ ] Red ROI bounding boxes correctly frame the options based on `layout.json`.
- [ ] Original `data/pages` are never overwritten.

**Verification:**
- [ ] Run script and manually inspect the 10 generated debug images.

**Dependencies:** Task 6.3
**Files likely touched:** `scripts/build_aligned_dataset.py`, `src/matera/vision/debug.py` (if needed)
**Estimated scope:** Small

#### Task 7: Implement mark maps, ROI extraction, and deterministic scores

**Description:** Build color/grayscale/template-difference evidence maps, extract configured ROIs, and calculate interpretable features and mark scores for circle, checkbox, and rating-scale strategies.

**Acceptance criteria:**

- [ ] Each expected ROI is visited exactly once.
- [ ] Mark evidence excludes as much printed text and background noise as practical.
- [ ] Scores expose their units/features and are reproducible.
- [ ] Evidence crops and mark maps can be saved to an explicit debug path.

**Verification:**

- [ ] Unit-test feature and threshold calculations.
- [ ] Inspect clear, blank, noisy, and ambiguous ROIs.
- [ ] Confirm source fixtures are not modified.

**Dependencies:** Task 6

**Files likely touched:**

- Mark map module
- ROI/feature module
- Image fixtures and focused tests

**Estimated scope:** Medium

##### Task 7.1: Define `ROIFeature` and `MarkScore` contracts
**Description:** Create frozen dataclasses for `ROIFeature` (numeric features) and `MarkScore` (normalized scores with metadata). Ensure they do not store mutable PIL images.
**Acceptance criteria:**
- Contracts are strictly typed and immutable.
- Image evidence paths are supported via `evidence_path` instead of in-memory objects.

##### Task 7.2: Implement template difference and morphology core
**Description:** Create the core image processing function that subtracts the `aligned_page` from the `reference_template` and cleans up noise using morphological operations.
**Acceptance criteria:**
- `aligned_page == reference_image` produces a near-zero mask.
- Original images are not mutated.

##### Task 7.3: Implement feature extraction and normalized scoring
**Description:** Given a cleaned mark map and raw crop, compute `dark_pixel_ratio`, `foreground_area_ratio`, `contour_count`, etc., and map them to a normalized 0.0-1.0 `score`.
**Acceptance criteria:**
- Features are deterministic.
- Clear marks produce scores distinct from blanks.

##### Task 7.4: Implement API resolver and golden dataset tests
**Description:** Implement `extract_mark_scores` which joins `FormProfile` and `PageLayout` to resolve `mark_strategy` per ROI. Test against `data/golden/labels.csv`.
**Acceptance criteria:**
- All 77 ROIs per page are accurately resolved.
- Golden evaluation test proves score separation (expected 1 vs 0).

#### Task 8: Implement decision and review routing

**Description:** Convert deterministic scores into selected/unselected/review outcomes, enforce question-level validation, and produce structured evidence for every decision, strictly following the NormalizedPageResult contract.

**Acceptance criteria:**

- [ ] High-confidence and low-confidence thresholds are configurable via `RoutingConfig`.
- [ ] Middle-range cases correctly generate `ReviewTask` objects.
- [ ] Over-selection and under-selection logic properly overrides option statuses.
- [ ] Invalid data (missing, duplicate) fails fast immediately.
- [ ] API signature strictly uses standard contracts from `matera.core.contracts`.

**Verification:**

- [ ] Unit tests pass for threshold boundaries.
- [ ] Unit tests pass for multi-select, rating-scale, over-selection, and under-selection.
- [ ] Missing score validation is tested.

**Dependencies:** Task 7

**Files likely touched:**

- `src/matera/vision/routing.py`
- `tests/test_routing.py`

**Estimated scope:** Medium

##### Task 8.1: Implement RoutingConfig and API scaffolding
**Description:** Define `RoutingConfig` with threshold validation, and create the skeleton for `route_page` returning `NormalizedPageResult`.
**Acceptance criteria:**
- [x] `RoutingConfig` validates `0 <= low < high <= 1`.
- [x] `route_page` signature matches the spec.

##### Task 8.2: Implement option-level deterministic scoring
**Description:** Iterate through `MarkScore`s and evaluate them against the thresholds to determine initial `selected` and `resolution_status`.
**Acceptance criteria:**
- Scores < low -> `selected=False`.
- Scores >= high -> `selected=True`.
- Scores between low and high -> `selected=None`, status="needs_review", and a pending `ReviewTask` is created.

##### Task 8.3: Implement question-level constraints
**Description:** Apply over-selection (`> max_selections`) and under-selection (`< min_selections`) constraints per question.
**Acceptance criteria:**
- Over-selection sets all `>high` options to `needs_review` + `ReviewTask`.
- Under-selection sets highest-score option to `needs_review` + `ReviewTask`.
- Page status bubbles up to `"review_required"` if any review task exists.

##### Task 8.4: Implement validation rules (Fail-fasts)
**Description:** Ensure missing/duplicate/unknown scores or metadata raise errors before building the page result.
**Acceptance criteria:**
- Missing `MarkScore` for profile option raises an error.
- Unknown/extra `MarkScore` raises an error.
- Missing `evidence_path` raises an error.

##### Task 8.5: Complete testing and integration
**Description:** Develop comprehensive unit tests covering all exact boundaries, strategies, and fail-fast scenarios.
**Acceptance criteria:**
- Test suite covers all scenarios outlined in the spec's Testing Strategy.
- Test coverage for `routing.py` is high.

### Checkpoint: Deterministic Baseline

- [ ] The rules-only pipeline processes the fixture PDF.
- [ ] Debug evidence is inspectable for each page and ROI.
- [ ] Baseline metrics are recorded.
- [ ] The main error categories are known before adding a classifier.

### Phase 3: Normalized Output And Baseline Evaluation

#### Task 9: Implement normalized answers and Excel export

**Description:** Convert page decisions into the normalized answer contract and export one row per page using the profile-specific column mapping.

**Acceptance criteria:**

- [ ] Exactly one output row is produced per input page.
- [ ] Output values are only `0` or `1`.
- [ ] Column names and order match the first profile.
- [ ] Review and confidence metadata are preserved outside the binary answer columns.

**Verification:**

- [ ] Validate row count, column count, ordering, and binary values.
- [ ] Open/re-read the generated workbook with a spreadsheet library.
- [ ] Test a page with multiple selected answers and a page with no marks.

**Dependencies:** Tasks 2 and 8

**Files likely touched:**

- Normalization module
- Export module
- Output validation tests

**Estimated scope:** Medium

#### Task 10: Build the evaluation harness

**Description:** Automate option-level, question-level, page-level, review-rate, coverage, and risk metrics against the golden dataset.

**Acceptance criteria:**

- [ ] Rules-only baseline metrics are reproducible.
- [ ] Metrics are reported by question type and difficulty.
- [ ] False positives, false negatives, and review cases are visible.
- [ ] A baseline report can be regenerated after each change.

**Verification:**

- [ ] Run the harness on a known fixture with expected metric values.
- [ ] Verify train/evaluation separation.
- [ ] Inspect a report and representative error evidence.

**Dependencies:** Task 5 and Task 8

**Files likely touched:**

- Evaluation module
- Metric tests
- Baseline report configuration

**Estimated scope:** Medium

### Checkpoint: End-To-End Baseline

- [ ] PDF input produces a validated XLSX output.
- [ ] Baseline metrics and error categories are recorded.
- [ ] Rules-only performance is good enough to identify the value of a classifier.

### Phase 4: Ambiguity Classifier And Hybrid Integration

#### Task 11: Prepare ambiguous-ROI training data

**Description:** Collect reviewed ambiguous ROIs, assign selected/unselected labels, extract features, and define a page/document-level train/validation split.

**Acceptance criteria:**

- [ ] Training examples come from reviewed evidence.
- [ ] Both selected and unselected ambiguous examples exist.
- [ ] Split policy prevents page/document leakage.
- [ ] Dataset version and label provenance are recorded.

**Verification:**

- [ ] Validate class balance and annotation completeness.
- [ ] Inspect representative training examples.
- [ ] Confirm evaluation pages are excluded from training.

**Dependencies:** Checkpoint: Deterministic Baseline

**Files likely touched:**

- Training-data preparation module
- Dataset metadata
- Training-data validation tests

**Estimated scope:** Medium

#### Task 12: Train and evaluate a lightweight feature classifier

**Description:** Train the first classifier on engineered ROI features, record model version and training metadata, and compare it with the deterministic baseline.

**Acceptance criteria:**

- [ ] Model training is reproducible from a versioned dataset and configuration.
- [ ] Model output includes confidence and model version.
- [ ] Classifier performance is reported on unseen pages/documents.
- [ ] The classifier is not used for clear deterministic cases.

**Verification:**

- [ ] Run training twice with equivalent configuration and compare metrics.
- [ ] Evaluate precision, recall, F1, false-positive rate, and calibration.
- [ ] Confirm a poor-confidence input can produce `review`.

**Dependencies:** Task 11

**Files likely touched:**

- Classifier module
- Training/evaluation script
- Model configuration and metadata

**Estimated scope:** Medium

#### Task 13: Integrate and calibrate the hybrid router

**Description:** Route only ambiguous ROIs to the classifier, calibrate thresholds, and preserve review routing for cases that remain uncertain.

**Acceptance criteria:**

- [ ] Clear ROIs bypass the classifier.
- [ ] Ambiguous ROIs receive classifier decisions with confidence.
- [ ] Low-confidence classifier outputs route to review.
- [ ] Hybrid results improve the difficult subset without materially regressing the easy subset.

**Verification:**

- [ ] Compare rules-only and hybrid metrics on the same hidden test set.
- [ ] Measure coverage, risk, and review rate.
- [ ] Inspect errors by category and save regression cases.

**Dependencies:** Tasks 10 and 12

**Files likely touched:**

- Ambiguity router
- Classifier integration
- Threshold/calibration tests

**Estimated scope:** Medium

### Checkpoint: Hybrid MVP

- [ ] Hybrid pipeline meets the agreed pilot acceptance targets.
- [ ] Low-confidence cases are reviewable with evidence.
- [ ] Model and profile versions are included in results.
- [ ] Baseline comparison shows the classifier adds measurable value.

### Phase 5: Operational Hardening

#### Task 14: Add the first user-facing execution interface

**Description:** Provide a local, simple entry point that accepts a PDF and profile, writes an XLSX and evidence output, and reports failures clearly. The interface may be CLI or local web UI after the runtime decision.

**Acceptance criteria:**

- [ ] User can provide an input PDF and receive the output artifact.
- [ ] Output and review evidence have deterministic locations.
- [ ] Invalid input and failed processing have actionable messages.
- [ ] Source input is never overwritten.

**Verification:**

- [ ] Run the complete flow on the fixture PDF.
- [ ] Test invalid PDF, unknown profile, alignment failure, and review cases.
- [ ] Confirm output workbook can be opened and re-read.

**Dependencies:** Checkpoint: Hybrid MVP

**Files likely touched:**

- CLI or application entry point
- Input/output handling
- Integration tests

**Estimated scope:** Medium

#### Task 15: Add regression, reproducibility, and quality gates

**Description:** Add automated checks for fixtures, metrics, profile validation, formatting, type checking, and reproducible environment setup according to the chosen runtime.

**Acceptance criteria:**

- [ ] Focused and full test commands are documented.
- [ ] Regression fixtures cover fixed bugs and difficult marks.
- [ ] Quality checks fail on contract, profile, or output regressions.
- [ ] Model/profile/data versions are captured in evaluation output.

**Verification:**

- [ ] Run all documented quality commands.
- [ ] Introduce a controlled regression and confirm the gate detects it.
- [ ] Review the final diff for accidental generated artifacts.

**Dependencies:** Task 14

**Files likely touched:**

- Test/quality configuration
- Regression fixtures
- CI or local verification configuration

**Estimated scope:** Medium

### Checkpoint: Complete

- [ ] All acceptance criteria and evaluation gates pass.
- [ ] The fixed-form workflow is documented end-to-end.
- [ ] Known limitations and manual-review policy are documented.
- [ ] The next profile can be added without changing core answer semantics.

## Parallelization Opportunities

After Task 1, Tasks 2 and the initial contract work in Task 2 can proceed in parallel. After page extraction, profile authoring and annotation tooling can proceed in parallel, but final annotations depend on the profile. Evaluation-harness work can proceed alongside deterministic implementation once the normalized contract is stable.

The following must remain sequential: profile before ROI extraction, deterministic baseline before classifier training, and classifier evaluation before changing production thresholds.

## Risks And Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Alignment is unreliable across scans | High | Use anchors/template registration, fail clearly, and inspect overlays before ROI scoring |
| Printed text is mistaken for handwriting | High | Use narrow mark corridors, template/color evidence, and false-positive-focused evaluation |
| Too few labeled pages for a classifier | High | Ship rules-only baseline first and collect reviewed ambiguous ROIs |
| ROI coordinates drift between profile revisions | High | Version profiles, validate overlays, and keep profile validation tests |
| Classifier leakage inflates metrics | High | Split by whole page/document and keep a hidden evaluation set |
| Source PDFs contain sensitive information | High | Process locally, avoid raw-content logging, and keep derived artifacts controlled |
| Excel mapping changes independently of CV | Medium | Keep normalized answers and profile-aware export as separate contracts |
| Manual review becomes a hidden bottleneck | Medium | Measure review rate and risk; improve thresholds only with evidence |

## Open Questions

- Which runtime and interface should be used for the first implementation?
- What is the exact Excel column naming and ordering contract?
- Are all 10 pages from the sample PDF representative of the production questionnaire?
- What additional real scans and mark styles are available for evaluation?
- What maximum automatic-decision risk and review rate are acceptable?
- Is manual review expected to be CLI-based, a local web screen, or an external workflow?

## Ready-To-Start Gate

Implementation should begin only after:

- [ ] Task 1 runtime decision is approved.
- [ ] Task 2 normalized answer and Excel contracts are approved.
- [ ] The first golden-data annotation policy is agreed.
- [ ] The user approves this task sequence or explicitly changes its scope.

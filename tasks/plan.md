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
**Description:** Use `pypdfium2` to parse the PDF, render each page at 300 DPI, and safely write to a `.tmp` directory before atomic promotion to the final output. Implement the `--force` flag.
**Acceptance criteria:**
- [ ] Process computes SHA-256 for source and each PNG.
- [ ] Without `--force`, fails immediately if output exists.
- [ ] With `--force`, safely uses `.backup` and `.tmp` for atomic promotion.
- [ ] Any failure cleans up `.tmp` and raises `ExtractionError`.
**Verification:**
- [ ] CLI runs via `python -m matera.data.extract`.

##### Task 3.3: Implement comprehensive extraction tests
**Description:** Validate all edges of the extraction logic.
**Acceptance criteria:**
- [ ] Test reproducing the dataset yields exact matching PNG hashes.
- [ ] Test failure paths (no force, corrupted).
- [ ] Test exact canonical manifest equivalence.
- [ ] Test source PDF hash remains perfectly identical before and after.
**Verification:**
- [ ] `pytest tests/test_extract.py` passes with >=90% coverage on `matera.data`.

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

#### Task 5: Create annotations and the golden dataset

**Description:** Manually label selected/unselected options for the available pages and record mark type, difficulty, and reviewer notes. Keep annotation files separate from source and derived images.

**Acceptance criteria:**

- [ ] Every expected option on the available pages has a label.
- [ ] Ambiguous examples are explicitly identified.
- [ ] Labels are tied to form profile version and page number.
- [ ] Dataset split policy is recorded.

**Verification:**

- [ ] Validate annotation completeness against the profile.
- [ ] Review a sample of annotations independently.
- [ ] Confirm no page from the evaluation split is used to tune thresholds.

**Dependencies:** Tasks 3 and 4

**Files likely touched:**

- Annotation files
- Annotation validator
- `docs/evaluation-plan.md`

**Estimated scope:** Medium

### Checkpoint: Foundation

- [ ] Runtime commands work.
- [ ] Contracts are documented and validated.
- [ ] All source pages are available as reproducible derived images.
- [ ] Profile overlays cover every question and option.
- [ ] Golden annotations are complete enough to measure a baseline.

### Phase 2: Deterministic Vision Baseline

#### Task 6: Implement page alignment

**Description:** Align each page to the form profile using configured anchors or template registration, then emit alignment quality and debug overlays.

**Acceptance criteria:**

- [ ] Aligned pages share the profile coordinate system.
- [ ] Alignment quality is measured and has a failure threshold.
- [ ] Failed alignment returns a structured error instead of using invalid ROIs.

**Verification:**

- [ ] Test translation, rotation, scale, and representative scan variation.
- [ ] Inspect overlays for all golden pages.
- [ ] Test the failed-alignment path.

**Dependencies:** Tasks 1, 3, and 4

**Files likely touched:**

- Alignment module
- Alignment tests
- Profile/debug utilities

**Estimated scope:** Medium

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

#### Task 8: Implement decision and review routing

**Description:** Convert deterministic scores into selected/unselected/review outcomes, enforce question-level validation, and produce structured evidence for every decision.

**Acceptance criteria:**

- [ ] High-confidence and low-confidence thresholds are configurable.
- [ ] Middle-range cases route to review or classifier input.
- [ ] No low-confidence case is silently forced to `0` or `1`.
- [ ] Contradictory or invalid question states are reported.

**Verification:**

- [ ] Test threshold boundaries and abstention behavior.
- [ ] Test multi-select, rating-scale, and checkbox rules.
- [ ] Compare decisions with golden annotations.

**Dependencies:** Task 7

**Files likely touched:**

- Decision/router module
- Validation module
- Decision and review tests

**Estimated scope:** Medium

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

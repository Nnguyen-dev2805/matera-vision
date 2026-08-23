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

- [ ] Task 6: Implement page alignment
- [ ] Task 7: Implement mark maps, ROI extraction, and deterministic scores
- [ ] Task 8: Implement decision and review routing

## Checkpoint: Deterministic Baseline

- [ ] Rules-only pipeline processes the fixture PDF
- [ ] Debug evidence is inspectable
- [ ] Baseline metrics are recorded
- [ ] Main error categories are known

## Phase 3: Normalized Output And Baseline Evaluation

- [ ] Task 9: Implement normalized answers and Excel export
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

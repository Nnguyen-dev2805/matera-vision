# Matera Vision Task List

## Phase 0: Contracts And Foundation

- [x] Task 1: Choose runtime and scaffold the project
- [ ] Task 2: Define the data contracts
  - [x] Task 2.1: Implement core semantic results (`AnswerKey`, `NormalizedAnswer`, `ReviewTask`, `NormalizedPageResult`) and tests
  - [x] Task 2.2: Implement semantic profile contracts (`OptionDef`, `QuestionDef`, `FormProfile`) and tests
  - [ ] Task 2.3: Implement Excel schema mapping contracts (`ColumnDef`, `ExcelSchema`) and tests

## Phase 1: Dataset And Profile Foundation

- [ ] Task 3: Build reproducible page extraction
- [ ] Task 4: Create the first form profile and ROI map
- [ ] Task 5: Create annotations and the golden dataset

## Checkpoint: Foundation

- [ ] Runtime commands work
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

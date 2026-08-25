# Implementation Tasks: Refactor V17 into src/matera

## Phase 1: Foundation (AI Classifier)

### Task 1: Migrate HOG+SVM Model Loader
**Description:** Move the loading of `shape_classifier.pkl` and `extract_hog_features` from `scratch/eval_v17_final.py` into `src/matera/classifier/model.py`.
**Acceptance criteria:**
- [x] `AmbiguityClassifier` class initializes the SVM model instead of the old 5-feature model.
- [x] `extract_hog_features` is correctly implemented and exposed.
**Verification:**
- [x] Manual check: Load the model via python interactive shell and run a dummy prediction.
**Dependencies:** None
**Files likely touched:** `src/matera/classifier/model.py`
**Estimated scope:** Small

## Phase 2: Core Vision Pipeline (`mark.py`)

### Task 2: Implement Global Enclosure & Local Radial
**Description:** Port `run_v11_global_topology` and `get_local_roi_crops` into `mark.py`.
**Acceptance criteria:**
- [x] Helper functions for Tầng 1 (Convex Hull) and Tầng 2 (Radial Angular Coverage) are cleanly available.
- [x] All geometric constants (e.g., `180` degrees, padding) are preserved.
**Verification:**
- [x] Build succeeds: `python -m py_compile src/matera/vision/mark.py`
**Dependencies:** None
**Files likely touched:** `src/matera/vision/mark.py`
**Estimated scope:** Medium

### Task 3: Implement Layer 1/2/3 Fallback for Q14 and Ambiguous
**Description:** Port `get_local_roi_crops_hsv` and `process_roi_hsv_ai` into `mark.py`, integrating the classifier from Task 1.
**Acceptance criteria:**
- [x] HSV thresholding and pixel counting logic is ported.
- [x] Function calls `AmbiguityClassifier` to get HOG+SVM prediction for ambiguous cases.
**Verification:**
- [x] Build succeeds: `python -m py_compile src/matera/vision/mark.py`
**Dependencies:** Task 1, Task 2
**Files likely touched:** `src/matera/vision/mark.py`
**Estimated scope:** Medium

### Task 4: Integrate Pipeline into `extract_mark_scores`
**Description:** Wire the global, local, and fallback functions inside the main `extract_mark_scores` pipeline.
**Acceptance criteria:**
- [x] `extract_mark_scores` executes Global Hull -> Local Radial -> AI Fallback based on question types (Q1-Q13 vs Q14).
- [x] Output is mapped correctly to `list[MarkScore]` objects with `decision_source` populated.
**Verification:**
- [x] Build succeeds: `python -m py_compile src/matera/vision/mark.py`
**Dependencies:** Task 2, Task 3
**Files likely touched:** `src/matera/vision/mark.py`
**Estimated scope:** Medium

## Checkpoint 1: Core Pipeline
- [x] Core vision functions are ported and compile successfully.
- [x] Classifier is correctly integrated.

## Phase 3: Clean up & Integration

### Task 5: Remove Obsolete Router
**Description:** Delete `hybrid_routing.py` since AI is now directly inside `mark.py`.
**Acceptance criteria:**
- [x] `src/matera/vision/hybrid_routing.py` is removed.
- [x] No import errors from other modules.
**Verification:**
- [ ] Build succeeds: `pytest` or Python syntax check on `src/matera/`.
**Dependencies:** Task 4
**Files likely touched:** `src/matera/vision/hybrid_routing.py`, `src/matera/main.py`
**Estimated scope:** XS

### Task 6: Refactor Streamlit App
**Description:** Port `scratch/streamlit_app_v17.py` to `src/matera/ui/dashboard.py` calling the clean `mark.py` API.
**Acceptance criteria:**
- [x] Dashboard logic uses `extract_mark_scores` instead of duplicating CV functions.
- [x] All visual debugging components work as before.
**Verification:**
- [ ] Manual check: Run `python -m streamlit run src/matera/ui/dashboard.py` and verify it loads the 10-page dataset.
**Dependencies:** Task 4
**Files likely touched:** `src/matera/ui/dashboard.py`, `scratch/streamlit_app_v17.py`
**Estimated scope:** Medium

## Checkpoint 2: End-to-end UI
- [ ] UI runs and extracts marks correctly using the official package.

## Phase 4: Verification

### Task 7: Test & Validation
**Description:** Run pipeline on matera-example.pdf to verify end-to-end functionality.
**Acceptance criteria:**
- [x] Execution script runs without crashing.
- [x] Output Excel and Walkthrough artifact are generated successfully.
**Verification:**
- [x] Run the script: `python src/matera/main.py process`
**Dependencies:** Task 6
**Files likely touched:** `src/matera/evaluation/evaluate_baseline.py`
**Estimated scope:** Small

## Checkpoint 3: Complete
- [ ] Accuracy is 94.68%.
- [ ] Ready for review.

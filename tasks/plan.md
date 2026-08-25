# Spec & Plan: Refactor V17 into src/matera

## 1. Specification

### 1.1 Objective
Move the proven V17 OMR pipeline from the experimental script (`scratch/eval_v17_final.py`) into the official `src/matera/` package structure without breaking its validated 94.68% accuracy. Replace the outdated V11 code currently residing in `src/matera/vision/mark.py`.

### 1.2 Tech Stack
- Python 3.12
- OpenCV (`cv2`), `scikit-image`, `scikit-learn`, `numpy`, `Pillow`
- Streamlit (for the debug UI)

### 1.3 Commands
- Test: `pytest`
- Run UI: `python -m streamlit run src/matera/ui/dashboard.py`

### 1.4 Project Structure Changes
- `src/matera/vision/mark.py`: Will contain the core mark detection pipeline (Global, Local Radial, L1/L2/L3 Fallback).
- `src/matera/classifier/model.py`: Will load the HOG+SVM `shape_classifier.pkl` instead of the old 5-feature classifier.
- `src/matera/ui/dashboard.py`: Will house the Streamlit debugging interface (moved from `scratch/streamlit_app_v17.py`).

### 1.5 Code Style
Follow PEP 8, enforce type hints (`-> list[MarkScore]`), keep geometric constants at the top of the module, and preserve the original mathematical logic exactly as implemented in V17.

### 1.6 Testing Strategy
- The refactored code must be tested against the 10 fixture pages and `data/ground_truth.json`.
- It must yield exactly the same TP/TN/FP/FN/Ambiguous metrics as the original `scratch/eval_v17_final.py`.

### 1.7 Boundaries
- **Always:** Wrap the final output of the computer vision logic in the standard `MarkScore` object to respect the data contract with the Excel Exporter.
- **Ask first:** Before altering any geometric parameters (e.g., `180` degrees threshold, `>1000` Hull Area).
- **Never:** Try to optimize the V17 algorithm during this refactoring phase. This phase is strictly for moving code, not changing behavior.

### 1.8 Success Criteria
- `src/matera/vision/mark.py` exposes `extract_mark_scores` running the full V17 pipeline.
- The Streamlit app runs from `src/matera/ui/dashboard.py` without import errors.
- Running the pipeline against the dataset matches the baseline output exactly.

---

## 2. Implementation Plan

| Phase | Description | Files Touched |
|---|---|---|
| **Phase 1** | Migrate AI Classifier | `src/matera/classifier/model.py` |
| **Phase 2** | Rewrite `mark.py` with V17 logic | `src/matera/vision/mark.py` |
| **Phase 3** | Delete obsolete router | `src/matera/vision/hybrid_routing.py` |
| **Phase 4** | Refactor Streamlit App | `src/matera/ui/dashboard.py` |
| **Phase 5** | Verify results | (Verification) |

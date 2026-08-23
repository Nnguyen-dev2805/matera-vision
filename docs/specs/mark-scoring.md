# Spec: Mark Maps and ROI Extraction (Task 7)

## Objective
Extract individual regions of interest (ROIs) from an `AlignedPage`, compute computer-vision features (e.g., dark pixel ratio, contour count), and calculate a normalized mark score (0.0 to 1.0). These deterministic scores will serve as the input for rule-based decision routing (Task 8) and eventually the ambiguity classifier (Phase 4).

The core challenge is separating handwriting from printed text/boxes. The primary method is **template difference**, using a pristine `reference_template` to subtract printed background and accurately measure only the respondent's marks.

## Tech Stack
- **OpenCV (`opencv-python-headless`)**: For image processing (thresholding, morphology, contour detection, absolute difference).
- **NumPy (`numpy`)**: For efficient array manipulation and pixel counting.
- **Pillow (`Pillow`)**: For image I/O and format bridging.

## Commands
```bash
# Run unit tests for mark scoring
pytest tests/test_mark.py -v

# Run linters and formatters
python -m ruff check src/matera/vision/mark.py src/matera/vision/contracts.py tests/test_mark.py
python -m ruff format src/matera/vision/mark.py src/matera/vision/contracts.py tests/test_mark.py
```

## Project Structure
```text
src/matera/vision/
  ├── mark.py          -> Core logic for mark map generation and scoring pipeline
  └── contracts.py     -> (Updated) Add ROIFeature and MarkScore contracts
tests/
  ├── test_mark.py     -> Unit tests for template difference and feature calculation
  └── fixtures/
      └── rois/        -> Synthetic ROI patches for threshold testing
```

## Code Style
Keep feature calculation pure and deterministic. Pipeline order: `resolved ROI -> crop aligned/reference -> template-diff mark map -> morphology cleanup -> features -> normalized MarkScore`. Do not mutate PIL images; image evidence is saved via explicit debug output paths rather than stored in frozen dataclasses.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ROIFeature:
    # Pure interpretable measurements
    dark_pixel_ratio: float
    foreground_area_ratio: float
    contour_count: int
    largest_component_ratio: float
    bbox_fill_ratio: float

@dataclass(frozen=True)
class MarkScore:
    question_id: str
    option_id: str
    # Normalized score between 0.0 and 1.0
    score: float
    # E.g., 'circle', 'checkbox', 'rating'
    strategy: str
    # Method used to compute the mark map, e.g., 'template_difference'
    method: str
    features: ROIFeature
    # Optional path to saved visual evidence (crop/map)
    evidence_path: str | None

def extract_mark_scores(
    aligned_page: AlignedPage,
    profile: FormProfile,
    layout: PageLayout, 
    reference_image: Image.Image,
    debug_dir: str | None = None
) -> list[MarkScore]:
    """
    Extracts features and normalizes a score for every ROI.
    Resolves the mark strategy using both the FormProfile and PageLayout.
    """
    pass
```

## Testing Strategy
- **Framework:** `pytest`
- **Synthetic Unit Tests:**
  - `test_mark_map_suppression`: Assert that feeding an identical reference patch (no handwriting) produces a mostly blank mark map (e.g., `dark_pixel_ratio <= 0.005`).
  - `test_mark_feature_checked`: Assert that a clearly checked synthetic mark produces a `dark_pixel_ratio >= 0.03` (or depending on fixture thresholds).
  - `test_roi_iteration`: Ensure exactly `len(layout.rois)` scores are returned, mapping semantic `mark_strategy` accurately.
- **Golden Dataset Acceptance Test:**
  - Load a subset from `data/golden/labels.csv` (e.g., one page's worth of 77 ROIs).
  - Run `extract_mark_scores`.
  - Assert that feature generation is deterministic.
  - Assert that score distributions separate logically: for a given `mark_strategy`, options marked `expected_mark=1` generally have higher `MarkScore.score` values than options marked `0`.
- **Immutability:** Assert that input `AlignedPage` and `reference_image` are strictly not mutated.

## Boundaries
- **Always:** Use template difference with pristine `reference_template` as the primary mark map. Keep boundaries separate: Task 7 computes features/scores; Task 8 makes the final `selected`/`unselected` decision.
- **Ask first:** Before adding machine-learning feature extractors (e.g., CNN embeddings) or adaptive thresholding as the primary logic instead of fallback.
- **Never:** Put mutable PIL `Image` instances into frozen dataclasses. Never overwrite source fixtures.

## Success Criteria
- **Suppression Invariant:** When `aligned_page == reference_image`, the foreground metrics (dark pixels, area) approach 0 for all ROIs, avoiding false positive printed boxes/text.
- **API Resolution:** Module correctly joins `FormProfile` and `PageLayout` to determine the correct `mark_strategy` (e.g., circle vs rating).
- **Golden Evaluation:** The calculated `score` across a golden page shows clear separation between selected (1) and unselected (0) labels.
- Source fixtures and `AlignedPage` are never mutated.


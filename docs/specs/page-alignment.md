# Spec: Page Alignment (Task 6)

## Objective
To align scanned questionnaire pages to a canonical coordinate system (the layout profile) before extracting Regions of Interest (ROIs). 

Because hand-drawn marks are captured on scanned paper, the images will have translation, rotation, and potentially minor scale variations compared to the original template. If we crop ROIs blindly, the answer marks might fall outside the crop, or we might crop printed text instead of handwriting.

**Success Criteria:**
- **Alignment:** Given an unaligned `RenderedPage` and a pristine reference template, produce an `AlignedPage` where the coordinate system matches the absolute pixels in `layout.json`.
- **Reference Image:** The reference must be a pristine, unmarked template (`reference_template.png`), provided separately. We will not use `page_1.png` because its handwritten marks would bias the feature matching towards one specific respondent's handwriting.
- **Output Invariants:** 
  - The `warp_matrix` transforms coordinates from **source image space -> profile coordinate space**.
  - Output images are sized exactly to `layout.pages[0].width_px` by `layout.pages[0].height_px`.
  - Emits 10 aligned images/overlays into an explicit output directory (e.g., `data/aligned_pages/`), never mutating `data/pages`.
  - Atomic writes: failures do not leave partial artifacts behind.
- **Quality Metric:** Produce an `alignment_score` based on the **inlier ratio** (number of RANSAC inlier matches / total good matches).
- **Explicit Failure:** If the inlier ratio falls below a configurable threshold (e.g., `0.6`), return an explicit `AlignmentError`. **Do not** silently return an incorrectly aligned image.
- **Debug Overlays:** Output a debug image reusing `matera.tools.profile_debug` to draw the bounding boxes.

## Tech Stack
- **OpenCV (`opencv-python-headless`)**: For feature extraction (ORB) and matching. Added to `pyproject.toml` because it provides robust, fast, and standard image registration algorithms.
- **NumPy (`numpy`)**: For geometric matrix operations (`warpAffine`). Added to `pyproject.toml`.
- **Pillow (`Pillow`)**: For bridging image formats and writing standard output.

## Commands
```bash
# Add OpenCV and Numpy to dependencies (in pyproject.toml)
# Note: Use `pip install -e .` or the repo's dependency manager after updating pyproject.toml

# Run alignment tests
pytest tests/test_alignment.py -v

# Run linters
ruff check src/matera/vision tests/test_alignment.py
ruff format src/matera/vision tests/test_alignment.py
```

## Project Structure
```text
src/matera/vision/
  ├── __init__.py
  ├── alignment.py     -> Core logic (ORB feature matching, RANSAC, warpAffine)
  └── contracts.py     -> AlignedPage, AlignmentError, AlignmentConfig
tests/
  ├── test_alignment.py
  └── fixtures/
      ├── reference_template.png
      └── unaligned_pages/ -> Translated, rotated, and scaled mock pages
```
*(Note: Debug overlay logic will reuse `src/matera/tools/profile_debug.py` rather than duplicating drawing code).*

## Code Style
Prefer deterministic functions over stateful classes unless caching feature descriptors of the reference template. Fail fast with custom exceptions.

```python
from typing import Literal

@dataclass(frozen=True)
class AlignmentConfig:
    algorithm: Literal["orb"] = "orb"
    transform_model: Literal["similarity", "affine"] = "affine"
    inlier_threshold: float = 0.6
    max_features: int = 5000
    reference_dpi: int = 300
    target_width: int = 0
    target_height: int = 0

@dataclass(frozen=True)
class AlignedPage:
    page_number: int
    image: Image.Image
    profile_form_id: str
    profile_version: str
    reference_dpi: int
    warp_matrix: np.ndarray
    alignment_score: float
    debug_evidence_path: Path | None = None

def align_page(
    source_page: RenderedPage, 
    reference_image: Image.Image, 
    config: AlignmentConfig
) -> AlignedPage:
    """
    Aligns the source page to the reference image using feature matching.
    The warp_matrix transforms from source space to reference space.
    Raises AlignmentError if the alignment_score (inlier ratio) < threshold.
    """
    pass
```

## Testing Strategy
- **Framework:** `pytest`
- **Unit Tests:**
  - `test_alignment_perfect_match`: Same image, matrix should be identity, score 1.0.
  - `test_alignment_translation`: Shift image by (50, -20) pixels, assert matrix corrects it, score > threshold.
  - `test_alignment_rotation`: Rotate image by 2 degrees, assert matrix corrects it, score > threshold.
  - `test_alignment_scale`: Scale by 1.02, assert matrix corrects it, score > threshold.
  - `test_alignment_blank_page`: Blank image, assert `AlignmentError`.
  - `test_alignment_low_feature`: Image with almost no features (e.g. solid white), assert `AlignmentError`.
  - `test_alignment_dimension_mismatch`: Input images with completely different aspect ratios/resolutions, assert `AlignmentError`.
  - `test_alignment_immutability`: Assert source `RenderedPage` is identical byte-for-byte after alignment.
  - `test_alignment_determinism`: Run alignment 3 times on the same input, assert identical `warp_matrix` and outputs.
- **Acceptance:**
  - Generate 10 aligned images with debug overlays from the real dataset into `data/aligned_pages/`.
  - Manually inspect overlays to ensure the red ROI boxes perfectly frame the answer choices.

## Boundaries
- **Always:** Return `AlignmentError` on failure. Preserve RGB mode for output images. Use `opencv-python-headless` to avoid GUI dependencies on the server. Write output atomically.
- **Ask first:** Before adding heavier Deep Learning dependencies (e.g., PyTorch). Traditional CV should be sufficient for fixed layouts.
- **Never:** Use a handwritten page (like `page_1.png`) as the reference anchor. Never silently fall back to unaligned images if registration fails. Never overwrite `data/pages`.

## Design Decisions
1. **Reference Image:** Resolved to require a pristine, clean reference template (`reference_template.png`).
2. **Transform Model:** For the MVP, we will use **Affine** (or Similarity) transform rather than Homography. Since forms are typically scanned relatively flat, full perspective homography can introduce severe warping errors if feature matches are noisy in one corner. Affine is safer and more constrained.
3. **Score Metric:** The `alignment_score` will strictly be the **inlier ratio** of the RANSAC filtering step.

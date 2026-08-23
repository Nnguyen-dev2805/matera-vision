# Spec: Form Profile and ROI Map (Task 4)

## Objective
We need to create the first versioned form profile and ROI (Region of Interest) map for the fixed Matera questionnaire template. 
To keep layout recognition and semantic data structuring decoupled, the profile must be split into two read-only configurations:
1. **Semantic Profile**: Defines questions, options, and response types.
2. **Layout Profile**: Defines page dimensions, anchors, coordinate systems, and explicit ROI boundaries mapped to the semantic options.

A robust loader will validate both configurations independently and verify cross-field invariants. A CLI utility will also be built to render these ROIs onto reference page images to visually verify accuracy across all 10 pages.

## Tech Stack
- **Language**: Python 3.12.x
- **Data Format**: JSON
- **Validation**: Python `dataclasses` + standard `json` (Zero external dependencies).
- **Imaging**: `Pillow` (PIL) for drawing the debug overlay.

## Project Structure
```text
profiles/
  matera-pre/
    v1/
      semantic.json             → Semantic questions, options, mark strategies
      layout.json               → Page dimensions, anchors, ROIs
src/matera/core/
  profile.py                    → Semantic dataclasses and validation
  layout.py                     → Layout dataclasses and validation
  errors.py                     → Structured error definitions
src/matera/tools/
  profile_debug.py              → Utility script to draw bounding boxes and anchors
tests/
  test_profile.py               → Tests for semantic loading and cross-validation
  test_layout.py                → Tests for layout loading and boundary checks
```

## Contracts & Code Style

### 1. Semantic Contract (`semantic.json`)
```json
{
  "form_id": "matera-pre",
  "form_version": "v1",
  "questions": [
    {
      "question_id": "Q1",
      "response_type": "single_choice",
      "mark_strategy": "checkbox",
      "options": ["a", "b", "c", "d"]
    }
  ]
}
```

### 2. Layout Contract (`layout.json`)
```json
{
  "form_id": "matera-pre",
  "form_version": "v1",
  "coordinate_space": "absolute_pixel",
  "reference_dpi": 200,
  "pages": [
    {
      "page_number": 1,
      "width_px": 1651,
      "height_px": 2334,
      "anchors": [
        {
          "anchor_id": "top_left_qr",
          "anchor_type": "qr_code",
          "bbox": {"x": 100, "y": 100, "w": 200, "h": 200}
        }
      ],
      "rois": [
        {
          "question_id": "Q1",
          "option_id": "a",
          "bbox": {"x": 300, "y": 450, "w": 40, "h": 40},
          "mark_strategy_override": null
        }
      ]
    }
  ]
}
```

### 3. Coordinate System Definition
- **Origin**: `(0, 0)` is the top-left corner of the aligned page image.
- **Bounding Box (`bbox`)**: Defined as `{"x": int, "y": int, "w": int, "h": int}`. The region is `[x, x+w)` and `[y, y+h)`.
- **Rounding**: All runtime math rounds to integer pixels immediately.
- **Scaling**: `layout.json` uses a `reference_dpi` (e.g., 200 DPI for `page_1.png`). If the extraction pipeline aligns at 300 DPI, coordinates MUST be scaled by `target_dpi / reference_dpi` at runtime before cropping.
- **Page Numbers**: 1-indexed (1 to N).

### 4. Structured Error Contract
Invalid configurations must raise a structured `ProfileValidationError`:
```python
@dataclass
class ProfileValidationError(Exception):
    profile_path: str
    field_path: str       # e.g., "pages[0].rois[3].bbox.w"
    error_code: str       # e.g., "NEGATIVE_DIMENSION", "MISSING_SEMANTIC_REF"
    reason: str           # e.g., "Width must be > 0"
```

## Testing Strategy
- **Framework**: `pytest`
- **Location**: `tests/test_profile.py`, `tests/test_layout.py`
- **Coverage Requirement**:
```powershell
pytest tests/test_profile.py tests/test_layout.py -v -o addopts="" `
  --cov=matera.core.profile `
  --cov=matera.core.layout `
  --cov-report=term-missing `
  --cov-fail-under=90
```
- **Validation Layers**:
  1. **Schema Check**: Type validation and missing fields.
  2. **Geometry Check**: Width/Height > 0, bounds within `width_px`/`height_px`.
  3. **Cross-Field Invariants**:
     - `form_id` and `form_version` exactly match between semantic and layout profiles.
     - Page numbers fall inside `1..page_count` sequentially.
     - Every ROI references a valid `question_id` and `option_id` that exists in the semantic profile.
     - No duplicate `(page_number, question_id, option_id)` tuples.
     - No orphaned options (every semantic option must have exactly one ROI defined in the layout).

## Debug Overlay Output Contract
The utility `matera.tools.profile_debug` visually verifies the profile:
- **Output Directory**: Created automatically (e.g., `debug/overlays/`).
- **Targeting**: Can run on all pages or a specific page via `--page-number <N>`.
- **Naming**: Stable filenames (`overlay_page_001.png`).
- **Visuals**:
  - Image size and mode remain identical to the input.
  - Page bounds drawn as a green border.
  - Anchors drawn as blue boxes.
  - ROIs drawn as red boxes with text labels `"{question_id}/{option_id}"`.
- **Failure**: A profile validation failure aborts the overlay process with a `ProfileValidationError` and produces no partial image.

## Boundaries
- **Always**:
  - Separate semantic definitions from layout configurations.
  - Load coordinates from JSON; NEVER hardcode Matera coordinates in Python.
- **Ask first**:
  - Adding a schema library like `pydantic`. (Current plan: pure dataclasses + dict iteration).
- **Never**:
  - Never allow a layout profile to map an ROI to a non-existent question.
  - Never silently fallback to defaults if a field is missing.

## Success Criteria
- [ ] `semantic.json` and `layout.json` are fully specified for all 10 pages and all target options.
- [ ] Strict type checking, bound checking, and cross-reference validation logic passes testing.
- [ ] The debug overlay successfully runs across all 10 pages (`page_1.png` to `page_10.png`), outputting perfectly aligned debug images.
- [ ] `pytest` coverage strictly exceeds 90% for the loader modules.

## Commands
```powershell
# Run the validation and debug overlay generator for page 1
python -m matera.tools.profile_debug --semantic profiles/matera-pre/v1/semantic.json --layout profiles/matera-pre/v1/layout.json --images-dir data/pages --output-dir debug/overlays --page-number 1

# Run tests
pytest tests/test_profile.py tests/test_layout.py -v -o addopts="" --cov=matera.core.profile --cov=matera.core.layout --cov-report=term-missing --cov-fail-under=90
```

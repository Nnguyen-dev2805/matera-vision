# Spec: Phase 1 Task 5 - Golden Dataset Generation

## Objective
Generate a "Golden Dataset" of 770 cropped ROI images (image patches) and their corresponding labels. 
This dataset serves as the standard evaluation benchmark for Phase 2's computer vision heuristics and Phase 3's ML classifier.
The script will use `layout.json` to accurately crop the 10 source page images (`data/pages/*.png`), and parse `data/ground_truth.json` (a human-verified annotation file) to assign labels. The dataset is derived from human ground truth, not automatically inferred.

## Tech Stack
- **Language**: Python 3.12
- **Image Library**: `Pillow==10.3.0`
- **Data handling**: standard `json`, `csv`, `pathlib`

## Commands
**Generate Dataset:**
```powershell
python scripts/build_golden_dataset.py --pages-dir data/pages --semantic profiles/matera-pre/v1/semantic.json --layout profiles/matera-pre/v1/layout.json --ground-truth data/ground_truth.json --output-dir data/golden
```

**Testing & QA:**
```powershell
pytest tests/test_golden.py
ruff check src/matera/data/golden.py scripts/build_golden_dataset.py tests/test_golden.py
ruff format --check src/matera/data/golden.py scripts tests
```

## Project Structure
- `src/matera/data/golden.py`: Core logic for parsing ground truth, validating annotations, and cropping datasets.
- `scripts/build_golden_dataset.py`: Thin CLI wrapper for the core logic.
- `data/golden/images/`: Directory to store the cropped PNG patches.
- `data/golden/labels.csv`: Output CSV containing detailed metadata and labels.
- `tests/test_golden.py`: Unit and acceptance tests.

## Code Style & Contracts
Use typed objects and strong input validation. Output CSV structure must include full traceability:
- `image_file`: e.g. `page_1_Q2_a.png`
- `form_id`: e.g. `matera-pre`
- `form_version`: e.g. `v1`
- `page_number`: e.g. `1`
- `source_page_image`: e.g. `page_1.png`
- `question_id`: e.g. `Q2`
- `option_id`: e.g. `a`
- `response_type`: e.g. `single_select`
- `mark_strategy`: e.g. `circle`
- `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`: Source crop coordinates
- `annotation_source`: e.g. `human_ground_truth.json`
- `expected_mark`: `1` (marked) or `0` (unmarked)
- `split`: `train` (pages 1-5) or `dev` (pages 6-10)

**Policy on Dataset Splits**: Threshold tuning and heuristic optimization may only use the `train` and `dev` splits. A future `test` split will be reserved for unseen data.

### Annotation Contract
Before cropping, the `ground_truth.json` file MUST be validated against the `semantic.json` profile:
- **Page Keys**: Must match `page_X`.
- **Question Keys**: Must map directly to `semantic.json` (e.g. converting `Q13_1` to `Q13.1`).
- **Option Mapping**: `ground_truth.json` uses numeric strings (`"0"`, `"1"`). These must be strictly mapped using the option's index in `semantic.json` (`question.options[int(idx)].option_id`).
- **Validation**: Any index out of bounds, missing page, or unmapped question must immediately fail dataset generation.

## Testing Strategy
- Unit tests verifying the parsing logic of `ground_truth.json`.
- Unit tests for index mapping (`"0" -> "a"`), raising clear errors on out-of-bounds indices.
- **Acceptance Tests**: Must run on the REAL profile and mock ground truth to assert:
  - Exactly 10 pages are processed.
  - Exactly 77 ROIs are processed per page.
  - Exactly 770 rows are generated in the CSV.
  - Every expected selected option maps successfully to `semantic.json` and `layout.json` without any unknown options.

## Boundaries
- **Always do**: Output clean PNG patches without mutating the source images. Validate the Annotation Contract before any files are written. Put core logic in `src/matera/data/golden.py`.
- **Ask first**: Changing the CSV schema or adding external dependencies (e.g., pandas) since standard library `csv` is sufficient.
- **Never do**: Overwrite the source `ground_truth.json` or `layout.json`. Never use `test` split for threshold tuning.

## Success Criteria
1. Core dataset logic is cleanly separated into `src/matera/data/golden.py`.
2. The script executes successfully and produces a populated `data/golden/images` folder.
3. `labels.csv` is created with exactly 770 rows containing all required audit metadata.
4. Acceptance tests pass using the real profile, confirming 100% annotation mapping coverage.

## Open Questions
- None.

# Spec: Form JSON Construction (Task 4.3)

## Objective
Construct the complete, production-ready `semantic.json` and `layout.json` files for the `matera-pre` form (version `v1`).
**Crucial Architectural Invariant:** The form template itself is only **1 page long**. The 10-page sample PDF represents 10 different scanned instances (respondents) of this single-page form layout. Therefore, the `layout.json` will describe exactly 1 page template, and the overlay debug tool will run this 1-page layout against all 10 extracted fixtures to verify alignment and stability.

## Tech Stack
- **Data Format**: JSON (Strictly adhering to `matera.core.profile` and `matera.core.layout` contracts).
- **Tools**: Python scripts (if a generator is needed) + `profile_debug.py` for visual verification.

## Commands
```powershell
# Validate the constructed JSONs directly against the loaders
python -c "from pathlib import Path; from matera.core.profile import load_semantic_profile; from matera.core.layout import load_layout_profile; s=load_semantic_profile(Path('profiles/matera-pre/v1/semantic.json')); load_layout_profile(Path('profiles/matera-pre/v1/layout.json'), s); print('Validation Passed!')"

# Run visual debug overlay for all pages to verify coordinates
python -m matera.tools.profile_debug --semantic profiles/matera-pre/v1/semantic.json --layout profiles/matera-pre/v1/layout.json --images-dir data/pages --output-dir debug/overlays
```

## Project Structure
```text
profiles/
  matera-pre/
    v1/
      semantic.json             → Final semantic definitions for 10 pages
      layout.json               → Final geometric ROIs and anchors for 10 pages
scripts/
  generate_profiles.py          → (Optional) Script to generate JSON from a simpler tabular format/CSV if coordinates are provided externally.
```

## Boundaries
- **Always**:
  - Run the `load_layout_profile` validator to catch any orphaned options or duplicate ROIs immediately after saving.
  - Verify visually using `profile_debug.py`.
- **Ask first**:
  - Inventing question IDs or mark strategies without human confirmation.
- **Never**:
  - Never commit placeholder coordinates `[0, 0, 0, 0]` as the final dataset. The success of this task requires true coordinates.

## Success Criteria
- [ ] `semantic.json` contains every question and option of the template form version.
- [ ] `layout.json` contains a single page layout with `reference_dpi: 200` and all bounding boxes for anchors and ROIs.
- [ ] The cross-field validator passes directly on the newly created profile files with 0 orphaned options.
- [ ] `profile_debug.py` successfully generates 10 overlays (applying the 1-page layout to 10 fixture images).
- [ ] **Visual Verification**: Manual inspection of all 10 overlays confirms ROIs are perfectly aligned, do not overlap neighboring options, and every expected option has exactly 1 ROI.

## Coordinate and Data Sourcing (Input Checklist)
Before proceeding to implementation, we must close these dependencies:
- [ ] **Form Structure Details**: We need the exact list of Question IDs (e.g. Q1-Q10), Option IDs, `response_type` (single/multi), and `mark_strategy` (circle/checkbox).
- [ ] **Coordinate Measurement Workflow**: Since AI cannot measure exact pixels from the image, we must agree on a workflow. (Options: provide a CSV with coordinates, provide a JSON dictionary, or the AI creates a script to help the human click-and-extract coordinates). 

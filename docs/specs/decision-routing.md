# Spec: Decision & Review Routing (Task 8)

## Objective
Implement the **AmbiguityRouter** and **NormalizedAnswerBuilder** as defined in the architecture. This module sits between the computer vision layer (Task 7) and the exporter. It receives a list of raw `MarkScore` objects for a page, applies deterministic thresholds and logical rules (based on `FormProfile` constraints like `min_selections`), and outputs a structured `NormalizedPageResult` contract.

Success looks like a robust pipeline that correctly groups options by question, detects contradictions (e.g., multiple selections exceeding `max_selections`), strictly generates `ReviewTask` objects for ambiguous cases, and fails fast on invalid or missing data.

## Tech Stack
- Python 3.12
- Standard library (dataclasses, typing, json)
- `pytest` for unit testing

## Commands
```bash
# Run tests for this module
python -m pytest tests/test_routing.py -v

# Run type checking and linting
python -m ruff check src/matera/vision/routing.py
python -m ruff format src/matera/vision/routing.py
```

## Project Structure
```text
src/matera/vision/routing.py   → Core AmbiguityRouter and NormalizedAnswerBuilder logic
src/matera/core/contracts.py   → Update/Add NormalizedAnswer data contracts
tests/test_routing.py          → Unit tests for routing rules and edge cases
```

## Code Style
Do not invent new models. Reuse the standard contracts from `matera.core.contracts`.

```python
from matera.core.contracts import NormalizedAnswer, ReviewTask, NormalizedPageResult
from matera.core.profile import FormProfile
from matera.vision.contracts import MarkScore
```

API Signature:
```python
def route_page(
    mark_scores: list[MarkScore], 
    profile: FormProfile, 
    page_number: int, 
    config: RoutingConfig | None = None
) -> NormalizedPageResult:
    pass
```

## Routing Policy (MVP Rules)
The pipeline uses a configurable `RoutingConfig(low_threshold=0.2, high_threshold=0.6)`. Must validate `0 <= low < high <= 1`.

For each option, evaluate the raw score:
1. **Low confidence:** Score `< low_threshold` → `selected: False`, `resolution_status: "resolved"`
2. **High confidence:** Score `>= high_threshold` → `selected: True`, `resolution_status: "resolved"`
3. **Ambiguous:** Score `low_threshold <= score < high_threshold` → `selected: None`, `resolution_status: "needs_review"`, and generate an option-level `ReviewTask`.

After option-level scoring, apply question-level constraints using `min_selections` and `max_selections` from `QuestionDef`:
4. **Contradictions (Over-selection):** If the number of `selected=True` options exceeds `max_selections` (e.g., >1 for single choice):
   - Change all those over-selected options to `selected: None` and `resolution_status: "needs_review"`.
   - Generate a `ReviewTask` for each of them.
5. **Missing Answers (Under-selection):** If the number of `selected=True` options is less than `min_selections` (e.g., 0 for a mandatory question):
   - Find the option(s) with the highest score in that question.
   - Change its status to `needs_review`, `selected: None`, and generate a `ReviewTask`. (This directs human attention to the most likely intended mark, or just forces a review of the blank question).
6. **Page Status:** If ANY option on the page is `needs_review`, the `NormalizedPageResult.page_status` must be set to `"review_required"`. Otherwise, `"resolved"`.

## Testing Strategy
- **Framework**: `pytest`.
- **Test Levels**: Unit tests purely on the data transformation layer. Mock `MarkScore` objects in memory.
- **Cases to cover**:
  - Clear selection (1 option > 0.6, others < 0.2).
  - Exact boundary thresholds (score exactly equal to `low_threshold` or `high_threshold`).
  - Ambiguous mark creating a `ReviewTask`.
  - Rating question correctly handling `min_selections=1, max_selections=1`.
  - Checkbox multi-select correctly handling `min_selections=0, max_selections=N`.
  - Contradiction (2 options > 0.6 when `max_selections=1`), ensuring both get `ReviewTask`.
  - Under-selection (0 options > 0.6 when `min_selections=1`), ensuring a `ReviewTask` is generated.
  - Page status correctly resolves to `"review_required"` when ambiguities exist.
  - Fail-fast validations (see Boundaries).

## Boundaries
- **Always do**: Group `MarkScore`s strictly by `question_id`. Create a `NormalizedAnswer` for every `MarkScore`. Set `decision_source="deterministic"`, `deterministic_score=score`, and `confidence=score`.
- **Always Fail-fast (Raise Error)**:
  - Missing `MarkScore` for an option defined in the `FormProfile`.
  - Duplicate `MarkScore` for the same option.
  - Unknown `question_id` or `option_id` not in the `FormProfile`.
  - Extra `MarkScore` that shouldn't be there.
  - Missing `evidence_path` metadata in the `MarkScore` (required for review tasks).
- **Ask first**: Adding any ML/Classifier logic. MVP is strictly deterministic threshold-based.
- **Never do**: Silently guess or force a `0`/`1` for a score in the ambiguous middle range.

## Success Criteria
- Given a list of `MarkScore` objects and a `FormProfile`, the module returns a valid `NormalizedPageResult`.
- Every `needs_review` answer strictly corresponds to a pending `ReviewTask`.
- Validation errors are raised correctly to prevent dirty data from reaching the exporter.
- Unit tests pass covering all rules defined in the Routing Policy and Testing Strategy.

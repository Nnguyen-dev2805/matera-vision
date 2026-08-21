# Spec: Phase 0 Task 2 - Data Contracts

## Objective
Specify the data contracts for the Matera Vision pipeline to guarantee clean boundaries before image processing begins. This separates machine vision semantic output (`NormalizedAnswer`) from human operational workflow (`ReviewTask`), and isolates the business logic from explicit file layouts (`LayoutProfile` - Task 4) and export formatting (`ExcelSchema`).

## Tech Stack
- **Language**: Python 3.12 (dataclasses, typing)
- **Testing**: `pytest` for validation boundaries

## Commands
- **Test**: `pytest tests/`
- **Lint**: `ruff check src tests`
- **Format**: `ruff format --check src tests`

## Project Structure
- `src/matera/core/contracts.py`: Semantic outcomes (`NormalizedAnswer`, `NormalizedPageResult`, `AnswerKey`) and workflow states (`ReviewTask`).
- `src/matera/core/profile.py`: Semantic question structures (`FormProfile`, `QuestionDef`, `OptionDef`).
- `src/matera/export/schema.py`: Explicit definitions for column naming and ordering.
- `tests/test_contracts.py`: Tests validating the structure boundaries and immutability.
- `tests/test_profile.py`: Tests validating profile constraints.

## Code Style
```python
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AnswerKey:
    form_id: str
    form_version: str
    page_number: int
    question_id: str
    option_id: str


@dataclass(frozen=True)
class NormalizedAnswer:
    answer_key: AnswerKey
    selected: bool | None
    resolution_status: Literal["resolved", "needs_review"]
    decision_source: Literal["deterministic", "classifier", "human"]
    confidence: float | None = None
    evidence_path: str | None = None
    deterministic_score: float | None = None  # Normalized to [0.0, 1.0] if present
    model_version: str | None = None
```
- Immutable dataclasses (`frozen=True`, use `tuple` instead of `list` for collections).
- Validate invariants via `__post_init__` (e.g. `resolution_status == "needs_review" -> selected is None`).

## Testing Strategy
- Unit tests focusing on boundary validation via `__post_init__` across all contracts.
- Ensure `ReviewTask` represents an independent workflow state.
- Validate `FormProfile` rejects invalid constraints (e.g., duplicate IDs, missing options, contradictory single_select configurations).

## Boundaries
- **Always do**: Keep `ReviewTask` linked only by `AnswerKey`. Ensure all answers emit a strict `0/1` binary mapping for the `ExcelSchema` via one-column-per-option.
- **Ask first**: Merging or removing `decision_source` fields.
- **Never do**: Add absolute pixel coordinates (ROI) into the semantic `FormProfile`. Never force `selected=None` to `0` silently during export.

## Success Criteria
1. `NormalizedAnswer` defines `selected`, `resolution_status`, and `decision_source` and enforces consistency in `__post_init__`.
2. `ReviewTask` defines a separate operational state, linked strictly by `AnswerKey`.
3. `NormalizedPageResult` groups answers and derives `page_status` automatically (returning `review_required` if any answer needs review).
4. `FormProfile` separates `response_type` and `mark_strategy`. It uses `OptionDef` to preserve option IDs and sequence. Strict validations are applied for constraints.
5. `ExcelSchema` defines the one-column-per-option mapping rule ensuring purely binary output for all question types.

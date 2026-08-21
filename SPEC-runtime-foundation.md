# Spec: Runtime & Foundation (Phase 0 Task 1)

## Objective
Establish the foundational runtime and project scaffolding for the Matera Vision pipeline. Since this pipeline will rely heavily on image processing and computer vision libraries, Python is the selected runtime. The objective is to set up a reproducible environment, dependency management, project structure, and basic commands for testing and linting, ensuring subsequent tasks start on solid ground.

## Tech Stack
- **Supported runtime**: Python 3.12.x
- **Supported OS**: Windows 10/11
- **Reference Runtime**: Python `3.12.3` (but any `3.12.x` is supported)
- **Dependency Management**: Native `venv` + `python -m pip`
- **Project Configuration**: `pyproject.toml` to define the package, dependencies, and tools configuration.
- **Build Strategy**: The project requires network access during the initial `pip install -e .[dev]` to download build isolation tools (`setuptools`, `wheel`) and pinned runtime packages.
- **Lock Strategy**: Exact versions are pinned directly in `pyproject.toml` (e.g. `pytest==9.1.1`). We do not use separate lockfiles to simplify the workflow and ensure `pip install -e .[dev]` uses the exact versions.
- **Testing**: `pytest`
- **Linting & Formatting**: `ruff`

## Commands
```powershell
# Setup environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package in editable mode with development dependencies
python -m pip install -e .[dev]

# Quality Gates (CI / Verification)
pytest tests/
ruff check .
ruff format --check .

# Auto-fix (Local Development)
ruff check --fix .
ruff format .

# Run Smoke Test
python -m matera.main
```

## Project Structure
```text
matera-vision/
├── src/                → Application source code
│   └── matera/         → Main package
│       ├── __init__.py → Package marker
│       ├── core/       → Domain models and contracts
│       └── main.py     → Application entry point (smoke test)
├── tests/              → Unit and integration tests
├── data/               → Source PDF fixtures and derived pages
├── profiles/           → Versioned form profile configurations
├── pyproject.toml      → Project metadata, exact pinned dependencies, ruff config
└── .gitignore          → Exclude .venv, cache, logs, debug images, extracted pages, model artifacts, output Excel
```

## Code Style
We follow standard PEP 8, enforced strictly by `ruff` configured via `pyproject.toml` (e.g., specific line length, rule set, exclude dirs).

```python
# src/matera/core/contracts.py

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class NormalizedAnswer:
    """Represents a single parsed answer from a form."""

    form_id: str
    form_version: str
    page_number: int
    question_id: str
    option_id: str
    decision: Literal["selected", "unselected", "review"]
    confidence: float | None = None
    evidence_path: str | None = None
    model_version: str | None = None
```

- **Naming**: `snake_case` for variables/functions, `PascalCase` for classes.
- **Typing**: Strict type hints for all function signatures.
- **Docstrings**: Required for all public interfaces and complex logic.

## Testing Strategy
- **Framework**: `pytest`
- **Location**: `tests/` directory mirroring the `src/matera/` structure.
- **Coverage Target**: Core contract tests must reach >= 90% line coverage.
- **Levels**: Unit tests for independent logic, integration tests for end-to-end flows in later phases.

## Boundaries
- **Always**: Type hint public methods. Ensure `ruff check` and `pytest` pass before committing. 
- **Ask first**: Before adding large binary dependencies (e.g., PyTorch, TensorFlow) if OpenCV/scikit-learn is sufficient.
- **Never**: Commit actual extracted PII/real production forms to `data/` or `tests/`. Only use approved development fixtures (e.g., `matera-example.pdf`).

## Success Criteria
- [ ] Python 3.12 runtime is confirmed.
- [ ] A virtual environment can be created and dependencies installed via `pip install -e .[dev]`.
- [ ] `pytest` runs and executes a minimal dummy test successfully, reporting coverage.
- [ ] `ruff check .` and `ruff format --check .` execute successfully.
- [ ] `pyproject.toml` is created with package metadata, dependencies, and Ruff configuration.
- [ ] The `src/` and `tests/` directories are physically scaffolded with proper `__init__.py`.
- [ ] `python -m matera.main` runs without crashing (smoke test).
- [ ] `.gitignore` is populated with all specified exclude patterns.

## Open Questions
- None. (Pending approval to begin Task 1).

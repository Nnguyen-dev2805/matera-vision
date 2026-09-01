# Matera Vision Test Suite

This directory contains the automated test suite for the Matera Vision project, organized to separate production verification from exploratory experiments.

## Directory Structure

### `unit/`
Contains fast, isolated tests for individual components and functions.
- **`vision/`**: Tests for vision processing, scoring algorithms, adaptive cropping, alignment, and internal contracts.
- **`core/`**: Tests for the core domain models (profiles, layouts, schemas, and IO).
- **`evaluation/`**: Tests for reporting, metrics calculation, and output formatting.
- **`vlm/`**: Tests for VLM integration, prompting, parsing, and resolving logic.

### `integration/`
Contains end-to-end tests that stitch multiple components together.
- **`production_pipeline/`**: Tests the main PDF → alignment → scoring → routing path. Includes golden mark tests, extraction end-to-end tests, and architecture boundary validations. These tests assert on production behavior.
- **`evaluation_pipeline/`**: Tests that run evaluation scripts and report generation end-to-end, validating the ground truth and diagnostic reporting outputs.

### `experiments/`
Contains tests for diagnostic tooling, legacy systems, and experimental features. **Important:** Tests in this directory are not proof that a feature is production-enabled.
- **`diagnostics/`**: Tests for diagnostic-only tools like the global shape validator, pixel probe, debug UI, and reference cleaner tools.
- **`legacy/`**: Tests for removed or deprecated code (e.g., legacy classifier) that we intentionally keep around for historical behavior verification or backwards compatibility.

## Running Tests

To run the full test suite:
```bash
pytest tests/
```

To run a specific test suite, e.g. unit tests:
```bash
pytest tests/unit/
```

# Spec: Comprehensive Extraction Tests (Task 3.3)

## Objective
Validate all edge cases, failure paths, and immutability guarantees of the PDF extraction pipeline (`matera.data.extract`). This ensures the dataset foundation remains perfectly reproducible, does not modify source data, and cleans up after itself on failure.

## Tech Stack
- **Language**: Python 3.12
- **Testing Framework**: `pytest`
- **Coverage**: `pytest-cov`
- **Mocking**: `pytest.MonkeyPatch`, `pytest.CaptureFixture`, `subprocess`

## Commands
```powershell
# Run only extraction tests with isolated coverage for matera.data.extract
pytest tests/test_extract.py -v -o addopts="" `
  --cov=matera.data.extract `
  --cov-report=term-missing `
  --cov-fail-under=90

# Run a real smoke test via subprocess to verify packaging and entrypoint
python -m matera.data.extract --input data/pdfs/matera-example.pdf --output data/pages/generated/matera-smoke-test --force
```

## Project Structure
- `tests/test_extract.py`: The single file containing all extraction pipeline tests.
- `src/matera/data/extract.py`: The target module under test.

## Code Style
Use `pytest` fixtures for temp paths.
- **Fixture paths**: Real integration tests MUST use `data/pdfs/matera-example.pdf` as the input fixture and only write output to `tmp_path`.
- **Mocking**: For simulated errors (corrupted PDFs, zero pages, render failures), mock `extract_pages()` or write dummy PDFs inside a valid repo-relative structure (e.g. by mocking `Path.cwd()`) to properly bypass the repository boundary check and test the actual failure.
- **Subprocess execution**: Most tests mock `sys.argv` and call `main()`, but at least one test must run `subprocess.run(["python", "-m", "matera.data.extract", ...])` to verify the CLI entrypoint directly.

## Testing Strategy
- **Coverage**: Must maintain >= 90% coverage across the `matera.data.extract` module.
- **Test Levels**: Unit tests for small utilities (`get_file_sha256`, `promote_directory`), integration tests for the CLI `main()` and `extract_pages()` generator.
- **Scope**: Third-party internals (`pypdfium2`, `Pillow`) are strictly out of scope. Use mock/fakes for third-party failures.

## Boundaries
- **Always**: Use `tmp_path` fixture for all output operations. Validate output directory state after failure (must not exist, or `.tmp` and `.backup` must be cleaned).
- **Ask first**: Adding new test dependencies or test data files to the repository.
- **Never**: Modify `data/pdfs/matera-example.pdf` during tests. 

## Success Criteria

### 1. Invariants
- The golden PDF has exactly 10 pages.
- Output contains exactly 10 PNGs.
- Filenames are strictly `page-0001.png` to `page-0010.png`.
- `page_number` is contiguous from 1 to `page_count`.
- `page_count` in manifest matches actual file count, and every listed filename exists.
- No stale or extra page artifacts exist in the directory.

### 2. Output Safety Matrix
Test all matrix combinations:
- Output is the exact same as input PDF.
- Output is a parent directory of the input PDF.
- Output resides inside `data/pdfs/`.
- Output is in a non-existent parent directory (should create it).
- Running command from repository root vs running from a different working directory.

### 3. Canonical Manifest Equivalence
- Deserialize both JSON manifests.
- Exclude **only** the `generated_at` field.
- Deeply compare all other fields (page order, filename, dimensions, hashes, renderer, Pillow version, render config).
- Manually calculate SHA-256 from the physical PNG bytes and ensure it perfectly matches `image_sha256` in the manifest.

### 4. Promotion Rollback & Cleanup
- Test when old output exists, `--force` is used, but promotion from `.tmp` to final fails:
  - Old output remains completely intact.
  - `.tmp` is completely removed.
  - `.backup` is completely removed.
  - No partial output exists.
- Test cleanup on render error:
  - When `extract_pages` fails midway, `.tmp` directory is completely removed.

## Open Questions
- (Closed) Task 3.3 is complete only when all acceptance tests defined in this SPEC pass. Third-party behaviors are out of scope. Use real fixtures for integration paths and mocks for edge cases.

# Spec: Phase 1 Task 3 - Reproducible Page Extraction

## Objective
Extract all pages from a source PDF into a reproducible, deterministic image dataset (PNG format) without modifying the original PDF. This dataset serves as the immutable foundation for all downstream computer vision tasks, golden annotation, and evaluation. Traceability and environment metadata must be fully recorded in `manifest.json` to guarantee precise reproducibility.

## Tech Stack
- **Language**: Python 3.12
- **PDF Renderer**: `pypdfium2==4.30.0`
- **Image Library**: `Pillow==10.3.0`
- **Hashing**: `hashlib` (SHA-256)

*Dependencies have been explicitly pinned in `pyproject.toml`.*

## Commands
**Extract Pages (Powershell):**
```powershell
# Fails if output exists
python -m matera.data.extract --input data/pdfs/matera-example.pdf --output data/pages/generated/matera-example

# Forces overwrite using a safe backup-and-replace strategy
python -m matera.data.extract --input data/pdfs/matera-example.pdf --output data/pages/generated/matera-example --force
```

**Testing & QA:**
```powershell
pytest tests/test_extract.py
ruff check src tests
ruff format --check src tests
```

## Project Structure
- `src/matera/data/extract.py`: CLI and core logic for PDF reading, rendering, and atomic promotions.
- `src/matera/data/contracts.py`: Dataclasses modeling the extraction pipeline results.
- `tests/test_extract.py`: Tests to guarantee extraction logic does not mutate the source, handles errors, and ensures reproducibility.

## Code Style & Contracts
Contracts cleanly separate in-memory processing (`RenderedPage`) from serializable state (`PageArtifact` and `ExtractionManifest`). `dict` is strictly avoided in favor of typed objects.

```python
from dataclasses import dataclass
from typing import Literal
from PIL import Image


@dataclass(frozen=True)
class RenderedPage:
    page_number: int  # 1-indexed
    width_px: int
    height_px: int
    pdf_width_pt: float
    pdf_height_pt: float
    image: Image.Image


@dataclass(frozen=True)
class PageArtifact:
    page_number: int
    filename: str
    image_sha256: str
    width_px: int
    height_px: int
    pdf_width_pt: float
    pdf_height_pt: float


@dataclass(frozen=True)
class RendererInfo:
    name: str
    version: str


@dataclass(frozen=True)
class ImageLibraryInfo:
    name: str
    version: str


@dataclass(frozen=True)
class RenderConfig:
    dpi: int
    mode: str
    compress_level: int
    optimize: bool
    strip_metadata: bool


@dataclass(frozen=True)
class ExtractionManifest:
    manifest_version: Literal["1"]
    source_path: str  # Must be repository-relative, e.g. "data/pdfs/matera-example.pdf"
    source_sha256: str
    page_count: int
    pages: tuple[PageArtifact, ...]
    renderer: RendererInfo
    image_library: ImageLibraryInfo
    render_config: RenderConfig
    generated_at: str  # Volatile timestamp (ISO 8601)
```

## Reproducibility & Metadata
- **Deterministic Checksums**: Both the source PDF and output PNGs must be hashed using `SHA-256`.
- **PNG Encoding Policy**: To ensure hashes match exactly, images must be saved with strictly controlled parameters:
  - `mode`: `"RGB"`
  - `compress_level`: `6`
  - `optimize`: `False`
  - EXIF, ICC, and DPI chunks MUST be completely stripped (`exif=b''`, `icc_profile=None`).
- *(Note: Cross-platform pixel-perfect equivalence is difficult to guarantee. The current equivalence boundary guarantees reproducibility within the exact same OS, runtime, and dependency versions.)*
- **Equivalence Check**: Equivalence between two runs is determined by comparing the **entire** deserialized `manifest.json` (source path/hash, filenames, dimensions, versions, config, and page ordering), completely excluding only the `generated_at` timestamp.
- **Canonical JSON**: `manifest.json` must be serialized with `sort_keys=True` and consistent indentation.

## Failure Behavior & Atomic Writes
- **Atomic Promotion on Windows (`--force`)**: Windows directory replacement requires a specific strategy to prevent partial states. 
  1. Render into a temporary directory (`.tmp`).
  2. If all pages and manifests render and validate successfully:
  3. Rename the existing output directory to a backup name (`.backup`).
  4. Rename the temporary directory to the final output name.
  5. Delete the backup directory.
  6. *(If step 4 fails, restore `.backup` back to the output name).*
  - If output exists and `--force` is omitted, the process **must fail immediately** without doing any work.
- **Fail-Fast**: Corrupt PDFs, zero-page PDFs, or rendering errors must clean up the temp directory and exit non-zero immediately.
- **Structured Errors**: Raise `ExtractionError(source_path, page_number: int | None, error_code, reason)`. Document-level errors (like corrupt PDF) use `page_number=None`. Do not crash blindly.

## Boundaries
- **Always do**: Render at exactly **300 DPI**, RGB mode, compress level 6, optimize false. Save as lossless `.png` with stable names `page-0001.png`, `page-0002.png`, etc. Use repository-relative paths for `source_path`. Use strong typed contracts.
- **Ask first**: Changing the version of `pypdfium2` or `Pillow`.
- **Never do**: Overwrite or alter the source PDF. Allow partial outputs. Write absolute paths into `manifest.json`.

## Success Criteria & Test Strategy
1. The script outputs exactly 10 PNG images (assert `pdf_page_count == 10`) and a `manifest.json`.
2. Tests must explicitly read back `manifest.json`, verify the physical files exist matching `filename`, and manually recalculate the SHA-256 of the PNG files to verify correctness.
3. Tests must verify `page_number` is contiguous from 1 to `page_count`.
4. Tests must verify the source hash is strictly unchanged.
5. Re-running without `--force` raises an error. Re-running with `--force` produces perfectly matching image hashes.
6. A visual QA check (manual) confirms `page-0001.png` correctly matches the visual intent of the original PDF.

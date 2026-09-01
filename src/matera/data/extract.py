import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pypdfium2 as pdfium
from PIL import __version__ as pil_version

from matera.data.contracts import (
    ExtractionError,
    ExtractionManifest,
    ImageLibraryInfo,
    PageArtifact,
    RenderConfig,
    RenderedPage,
    RendererInfo,
)
from matera.data.io import atomic_write_manifest, save_image_lossless

try:
    from pypdfium2.version import V_PYPDFIUM2
except ImportError:
    V_PYPDFIUM2 = "unknown"


def get_file_sha256(path: Path) -> str:
    """Computes the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_pages(pdf_path: Path, dpi: int = 300) -> Iterator[RenderedPage]:
    """Extracts pages from a PDF and yields RenderedPage objects."""
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception as e:
        raise ExtractionError(
            source_path=str(pdf_path),
            page_number=None,
            error_code="PDF_LOAD_FAILED",
            reason=f"Failed to load PDF: {e}"
        ) from e

    try:
        for i, page in enumerate(pdf):
            page_num = i + 1
            try:
                # Render to PIL. pypdfium2 default scale is 72 dpi.
                scale = dpi / 72.0
                pil_image = page.render(scale=scale).to_pil()

                # Dimensions in pt (72 dpi)
                width_pt, height_pt = page.get_size()

                # Ensure it's RGB
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                width_px, height_px = pil_image.size
                yield RenderedPage(
                    page_number=page_num,
                    width_px=width_px,
                    height_px=height_px,
                    pdf_width_pt=float(width_pt),
                    pdf_height_pt=float(height_pt),
                    image=pil_image,
                )
            except Exception as e:
                raise ExtractionError(
                    source_path=str(pdf_path),
                    page_number=page_num,
                    error_code="PAGE_PROCESS_FAILED",
                    reason=f"Failed to process page {page_num}: {e}"
                ) from e
    finally:
        pdf.close()


def promote_directory(tmp_dir: Path, target_dir: Path) -> None:
    """
    Atomically renames tmp_dir to target_dir. 
    If target_dir exists, safely renames it to .backup first.
    """
    if target_dir.exists():
        backup_name = f"{target_dir.name}.backup_{uuid.uuid4().hex[:8]}"
        backup_dir = target_dir.with_name(backup_name)
            
        os.rename(target_dir, backup_dir)
        try:
            os.rename(tmp_dir, target_dir)
            shutil.rmtree(backup_dir)
        except Exception:
            # Rollback
            if target_dir.exists():
                shutil.rmtree(target_dir)
            os.rename(backup_dir, target_dir)
            raise
    else:
        os.rename(tmp_dir, target_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract pages from a PDF into a directory.")
    parser.add_argument("--input", type=Path, required=True, help="Path to source PDF")
    parser.add_argument("--output", type=Path, required=True, help="Target output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    in_path: Path = args.input
    out_path: Path = args.output

    if not in_path.exists():
        print(f"Error: Input file {in_path} does not exist", file=sys.stderr)
        return 1

    try:
        in_res = in_path.resolve(strict=True)
        out_res = out_path.resolve()
        if in_res == out_res or out_res in in_res.parents:
            print("Error: Output directory cannot be the same as or a parent of the input file.", file=sys.stderr)  # noqa: E501
            return 1
            
        # Protect data/pdfs
        if "data/pdfs" in out_res.as_posix():
            print("Error: Output directory cannot be inside the source PDFs directory.", file=sys.stderr)  # noqa: E501
            return 1
    except Exception as e:
        print(f"Error validating paths: {e}", file=sys.stderr)
        return 1

    if out_path.exists() and not args.force:
        print(f"Error: Output directory {out_path} already exists. Use --force to overwrite.", file=sys.stderr)  # noqa: E501
        return 1

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_prefix = f"{out_path.name}.tmp_"
        tmp_dir_str = tempfile.mkdtemp(prefix=tmp_prefix, dir=out_path.parent)
        tmp_dir = Path(tmp_dir_str)
    except Exception as e:
        print(f"Error creating temporary directory: {e}", file=sys.stderr)
        return 1

    try:
        source_hash = get_file_sha256(in_path)
        artifacts = []
        for rendered in extract_pages(in_path):
            filename = f"page-{rendered.page_number:04d}.png"
            out_img = tmp_dir / filename

            save_image_lossless(rendered.image, out_img)
            img_hash = get_file_sha256(out_img)

            artifacts.append(PageArtifact(
                page_number=rendered.page_number,
                filename=filename,
                image_sha256=img_hash,
                width_px=rendered.width_px,
                height_px=rendered.height_px,
                pdf_width_pt=rendered.pdf_width_pt,
                pdf_height_pt=rendered.pdf_height_pt,
            ))

        if not artifacts:
            raise ExtractionError(
                source_path=in_path.as_posix(),
                page_number=None,
                error_code="PDF_EMPTY",
                reason="PDF contains no pages",
            )

        try:
            # Determine repo root dynamically
            repo_root = Path.cwd().resolve(strict=True)
            current = repo_root
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    repo_root = current
                    break
                current = current.parent
            
            in_res = in_path.resolve(strict=True)
            rel_in_path = in_res.relative_to(repo_root).as_posix()
        except ValueError:
            print(f"Error: Input file must be within the repository root", file=sys.stderr)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            return 1

        manifest = ExtractionManifest(
            manifest_version="1",
            source_path=rel_in_path,
            source_sha256=source_hash,
            page_count=len(artifacts),
            pages=tuple(artifacts),
            renderer=RendererInfo("pypdfium2", V_PYPDFIUM2),
            image_library=ImageLibraryInfo("pillow", pil_version),
            render_config=RenderConfig(300, "RGB", 6, False, True),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        
        atomic_write_manifest(manifest, tmp_dir / "manifest.json")
        promote_directory(tmp_dir, out_path)
        print(f"Successfully extracted {len(artifacts)} pages to {out_path}")
        return 0

    except ExtractionError as e:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        print(f"ExtractionError: {e.error_code} at {e.source_path} (page {e.page_number}): {e.reason}", file=sys.stderr)  # noqa: E501
        return 1
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        print("An unexpected error occurred. See traceback below.", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

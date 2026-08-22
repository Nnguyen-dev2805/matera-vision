import json
from pathlib import Path

import pytest
from PIL import Image

from matera.data.contracts import (
    ExtractionManifest,
    ImageLibraryInfo,
    PageArtifact,
    RenderConfig,
    RendererInfo,
)
from matera.data.io import atomic_write_manifest, save_image_lossless


def test_save_image_lossless(tmp_path: Path):
    out_file = tmp_path / "test.png"
    # Create a non-RGB image
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))

    save_image_lossless(img, out_file)

    assert out_file.exists()

    # Read it back to verify RGB mode and no extra info
    saved_img = Image.open(out_file)
    assert saved_img.mode == "RGB"
    assert saved_img.format == "PNG"

    # Information like EXIF and ICC should be stripped
    assert "exif" not in saved_img.info or not saved_img.info["exif"]
    assert "icc_profile" not in saved_img.info or not saved_img.info["icc_profile"]


def test_atomic_write_manifest(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"

    manifest = ExtractionManifest(
        manifest_version="1",
        source_path="data/pdfs/doc.pdf",
        source_sha256="a" * 64,
        page_count=1,
        pages=(
            PageArtifact(
                page_number=1,
                filename="page-0001.png",
                image_sha256="b" * 64,
                width_px=100,
                height_px=100,
                pdf_width_pt=50.0,
                pdf_height_pt=50.0,
            ),
        ),
        renderer=RendererInfo("pypdfium2", "1"),
        image_library=ImageLibraryInfo("pillow", "1"),
        render_config=RenderConfig(300, "RGB", 6, False, True),
        generated_at="2026-01-01T00:00:00Z",
    )

    atomic_write_manifest(manifest, manifest_path)

    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if pretty formatted and sorted
    assert content.endswith("\n")
    data = json.loads(content)
    assert data["manifest_version"] == "1"
    assert data["pages"][0]["filename"] == "page-0001.png"
    assert data["renderer"]["name"] == "pypdfium2"

    # Check order of keys (sort_keys=True)
    # The first key in sorted order should be 'generated_at'
    lines = content.splitlines()
    assert '"generated_at":' in lines[1]


def test_atomic_write_manifest_overwrite(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("old")

    manifest = ExtractionManifest(
        manifest_version="1",
        source_path="data/pdfs/doc.pdf",
        source_sha256="a" * 64,
        page_count=1,
        pages=(
            PageArtifact(
                page_number=1,
                filename="page-0001.png",
                image_sha256="b" * 64,
                width_px=100,
                height_px=100,
                pdf_width_pt=50.0,
                pdf_height_pt=50.0,
            ),
        ),
        renderer=RendererInfo("pypdfium2", "1"),
        image_library=ImageLibraryInfo("pillow", "1"),
        render_config=RenderConfig(300, "RGB", 6, False, True),
        generated_at="2026-01-01T00:00:00Z",
    )

    atomic_write_manifest(manifest, manifest_path)
    assert manifest_path.read_text(encoding="utf-8").startswith("{")


def test_atomic_write_manifest_cleanup_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import shutil

    manifest_path = tmp_path / "manifest.json"

    def mock_move(*args, **kwargs):
        raise OSError("Simulated failure")

    monkeypatch.setattr(shutil, "move", mock_move)

    manifest = ExtractionManifest(
        manifest_version="1",
        source_path="data/pdfs/doc.pdf",
        source_sha256="a" * 64,
        page_count=1,
        pages=(
            PageArtifact(
                page_number=1,
                filename="page-0001.png",
                image_sha256="b" * 64,
                width_px=100,
                height_px=100,
                pdf_width_pt=50.0,
                pdf_height_pt=50.0,
            ),
        ),
        renderer=RendererInfo("pypdfium2", "1"),
        image_library=ImageLibraryInfo("pillow", "1"),
        render_config=RenderConfig(300, "RGB", 6, False, True),
        generated_at="2026-01-01T00:00:00Z",
    )

    with pytest.raises(OSError, match="Simulated failure"):
        atomic_write_manifest(manifest, manifest_path)

    # Verify no tmp files left
    assert len(list(tmp_path.glob("*.tmp"))) == 0


def test_save_image_lossless_hash(tmp_path: Path):
    import hashlib

    out_file = tmp_path / "test_hash.png"
    img = Image.new("RGB", (10, 10), color="white")

    save_image_lossless(img, out_file)

    # Verify we can hash the file and it doesn't change on re-save
    first_hash = hashlib.sha256(out_file.read_bytes()).hexdigest()

    # Save again to same or different path
    out_file2 = tmp_path / "test_hash2.png"
    save_image_lossless(img, out_file2)

    second_hash = hashlib.sha256(out_file2.read_bytes()).hexdigest()
    assert first_hash == second_hash

import os
import sys
from pathlib import Path

import pytest
from PIL import Image

from matera.data.contracts import ExtractionError
from matera.data.extract import (
    extract_pages,
    get_file_sha256,
    main,
    promote_directory,
)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = Path("data/example/matera-example.pdf")
    if not pdf_path.exists():
        pytest.fail(f"Required test fixture {pdf_path.resolve()} not found")
    return pdf_path


def test_get_file_sha256(tmp_path: Path):
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"hello world")
    # echo -n "hello world" | sha256sum
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert get_file_sha256(test_file) == expected


def test_extract_pages_corrupt_pdf(tmp_path: Path):
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"12345")
    with pytest.raises(ExtractionError) as exc:
        list(extract_pages(corrupt_pdf))
    assert exc.value.error_code == "PDF_LOAD_FAILED"


def test_extract_pages_happy_path(sample_pdf: Path):
    pages = list(extract_pages(sample_pdf))
    assert len(pages) == 10

    first_page = pages[0]
    assert first_page.page_number == 1
    assert first_page.width_px > 0
    assert first_page.height_px > 0
    assert first_page.pdf_width_pt > 0
    assert first_page.pdf_height_pt > 0
    assert isinstance(first_page.image, Image.Image)
    assert first_page.image.mode == "RGB"


def test_promote_directory_simple(tmp_path: Path):
    tmp_dir = tmp_path / "tmp"
    target_dir = tmp_path / "target"

    tmp_dir.mkdir()
    (tmp_dir / "file.txt").write_text("hello")

    promote_directory(tmp_dir, target_dir)

    assert target_dir.exists()
    assert (target_dir / "file.txt").exists()
    assert not tmp_dir.exists()


def test_promote_directory_overwrite(tmp_path: Path):
    tmp_dir = tmp_path / "tmp"
    target_dir = tmp_path / "target"

    target_dir.mkdir()
    (target_dir / "old.txt").write_text("old")

    tmp_dir.mkdir()
    (tmp_dir / "new.txt").write_text("new")

    promote_directory(tmp_dir, target_dir)

    assert target_dir.exists()
    assert (target_dir / "new.txt").exists()
    assert not (target_dir / "old.txt").exists()
    assert not tmp_dir.exists()
    # Check no backups remain
    assert not list(tmp_path.glob("target.backup_*"))


def test_promote_directory_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tmp_dir = tmp_path / "tmp"
    target_dir = tmp_path / "target"

    target_dir.mkdir()
    (target_dir / "old.txt").write_text("old")

    tmp_dir.mkdir()
    (tmp_dir / "new.txt").write_text("new")

    # Mock os.rename to fail ONLY when renaming tmp_dir to target_dir
    original_rename = os.rename

    def mock_rename(src, dst):
        if Path(src) == tmp_dir and Path(dst) == target_dir:
            raise OSError("Simulated failure during promotion")
        return original_rename(src, dst)

    monkeypatch.setattr(os, "rename", mock_rename)

    with pytest.raises(OSError, match="Simulated failure"):
        promote_directory(tmp_dir, target_dir)

    # Target dir should be rolled back to its original state
    assert target_dir.exists()
    assert (target_dir / "old.txt").exists()
    assert not (target_dir / "new.txt").exists()


def test_cli_input_missing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", "doesnotexist.pdf", "--output", "out"]
    )
    assert main() == 1
    out, err = capsys.readouterr()
    assert "does not exist" in err


def test_cli_output_exists_no_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    in_pdf = tmp_path / "in.pdf"
    in_pdf.write_bytes(b"dummy")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", str(in_pdf), "--output", str(out_dir)]
    )
    assert main() == 1
    out, err = capsys.readouterr()
    assert "already exists. Use --force to overwrite" in err


def test_cli_output_protects_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    in_pdf = tmp_path / "in.pdf"
    in_pdf.write_bytes(b"dummy")
    out_dir = tmp_path

    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", str(in_pdf), "--output", str(out_dir), "--force"]
    )
    assert main() == 1
    out, err = capsys.readouterr()
    assert "Output directory cannot be the same as or a parent of the input file" in err


def test_cli_happy_path(tmp_path: Path, sample_pdf: Path, monkeypatch: pytest.MonkeyPatch):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", str(sample_pdf), "--output", str(out_dir)]
    )

    source_hash_before = get_file_sha256(sample_pdf)
    assert main() == 0
    source_hash_after = get_file_sha256(sample_pdf)

    assert source_hash_after == source_hash_before
    assert out_dir.exists()
    assert (out_dir / "manifest.json").exists()

    png_files = list(out_dir.glob("*.png"))
    assert len(png_files) == 10

    # Check exact filenames
    expected_filenames = {f"page-{i:04d}.png" for i in range(1, 11)}
    actual_filenames = {p.name for p in png_files}
    assert actual_filenames == expected_filenames

    # Check for no stale artifacts
    all_files = list(out_dir.glob("*"))
    assert len(all_files) == 11  # 10 PNGs + 1 manifest

    # Check manifest relative path
    import json

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["source_path"] == "data/example/matera-example.pdf"
    assert manifest["source_sha256"] == source_hash_before
    assert manifest["page_count"] == 10
    assert manifest["page_count"] == len(manifest["pages"])

    # Check PNG hashes and page numbers
    for i, page in enumerate(manifest["pages"], start=1):
        assert page["page_number"] == i
        png_path = out_dir / page["filename"]
        assert get_file_sha256(png_path) == page["image_sha256"]


def test_cli_subprocess_smoke(tmp_path: Path, sample_pdf: Path):
    import subprocess

    out_dir = tmp_path / "smoke_out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "matera.data.extract",
            "--input",
            str(sample_pdf),
            "--output",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert out_dir.exists()
    assert (out_dir / "manifest.json").exists()


def test_cli_parent_directory_creation(
    tmp_path: Path, sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
):
    out_dir = tmp_path / "deeply" / "nested" / "out"
    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", str(sample_pdf), "--output", str(out_dir)]
    )

    assert main() == 0
    assert out_dir.exists()
    assert (out_dir / "manifest.json").exists()


def test_cli_force_equivalence(tmp_path: Path, sample_pdf: Path, monkeypatch: pytest.MonkeyPatch):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", str(sample_pdf), "--output", str(out_dir)]
    )

    assert main() == 0
    import json

    manifest1 = json.loads((out_dir / "manifest.json").read_text())

    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", str(sample_pdf), "--output", str(out_dir), "--force"]
    )
    assert main() == 0
    manifest2 = json.loads((out_dir / "manifest.json").read_text())

    # Exclude generated_at from equivalence check
    manifest1.pop("generated_at")
    manifest2.pop("generated_at")

    # Deeply compare all other fields
    assert manifest1 == manifest2

    # Manually calculate SHA-256 from the physical PNG bytes
    for page in manifest2["pages"]:
        png_path = out_dir / page["filename"]
        # Ensure it perfectly matches image_sha256 in the manifest
        assert get_file_sha256(png_path) == page["image_sha256"]


def test_cli_zero_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    in_pdf = tmp_path / "empty.pdf"
    in_pdf.write_bytes(b"dummy")
    out_dir = tmp_path / "out"

    # Mock extract_pages to yield nothing
    from matera.data import extract

    monkeypatch.setattr(extract, "extract_pages", lambda *args, **kwargs: iter([]))

    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", str(in_pdf), "--output", str(out_dir)]
    )

    assert main() == 1
    out, err = capsys.readouterr()
    assert "PDF_EMPTY" in err
    assert not out_dir.exists()


def test_cli_cleanup_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    in_pdf = tmp_path / "error.pdf"
    in_pdf.write_bytes(b"dummy")
    out_dir = tmp_path / "out"

    # Mock extract_pages to yield one page then raise an error
    def mock_extract_pages(*args, **kwargs):
        from PIL import Image

        from matera.data import extract

        yield extract.RenderedPage(1, 10, 10, 10.0, 10.0, Image.new("RGB", (10, 10)))
        raise ExtractionError("error.pdf", 2, "PAGE_PROCESS_FAILED", "Simulated error")

    from matera.data import extract

    monkeypatch.setattr(extract, "extract_pages", mock_extract_pages)

    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", str(in_pdf), "--output", str(out_dir)]
    )

    assert main() == 1
    out, err = capsys.readouterr()
    assert "PAGE_PROCESS_FAILED" in err

    # Temporary directory should be cleaned up
    assert not list(tmp_path.glob("out.tmp_*"))
    assert not out_dir.exists()


def test_cli_promotion_rollback(
    tmp_path: Path, sample_pdf: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    out_dir = tmp_path / "out"

    # First create some old output
    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", str(sample_pdf.resolve()), "--output", str(out_dir)]
    )
    assert main() == 0
    (out_dir / "old_marker.txt").write_text("old")

    old_files = {p.name: p.read_bytes() for p in out_dir.iterdir() if p.is_file()}
    assert len(old_files) > 0

    # Now mock os.rename to fail ONLY when renaming tmp_dir to target_dir during promotion
    import os

    original_rename = os.rename

    def mock_rename(src, dst):
        if Path(dst) == out_dir and ".tmp_" in Path(src).name:
            raise OSError("Simulated failure during promotion")
        return original_rename(src, dst)

    monkeypatch.setattr(os, "rename", mock_rename)

    # Run with --force
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract.py",
            "--input",
            str(sample_pdf.resolve()),
            "--output",
            str(out_dir),
            "--force",
        ],
    )
    assert main() == 1
    out, err = capsys.readouterr()
    assert "Simulated failure during promotion" in err

    # Verify old output is completely intact
    new_files = {p.name: p.read_bytes() for p in out_dir.iterdir() if p.is_file()}
    assert new_files == old_files

    # Verify .tmp and .backup are completely removed
    assert not list(out_dir.parent.glob("out.tmp_*"))
    assert not list(out_dir.parent.glob("out.backup_*"))


def test_cli_protects_data_pdfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    in_pdf = tmp_path / "in.pdf"
    in_pdf.write_bytes(b"dummy")
    # Output cannot be exactly the input file
    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", str(in_pdf), "--output", str(in_pdf), "--force"]
    )
    assert main() == 1
    out, err = capsys.readouterr()
    assert "Output directory cannot be the same as or a parent of the input file" in err

    # Check true repository data/pdfs path
    out_dir = Path("data/pdfs/out")

    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", str(in_pdf), "--output", str(out_dir), "--force"]
    )
    assert main() == 1
    out, err = capsys.readouterr()
    assert "Output directory cannot be inside the source PDFs directory" in err


def test_cli_repo_boundary_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    # Setup fake repo and outside file
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "pyproject.toml").touch()

    in_pdf = tmp_path / "outside_in.pdf"
    in_pdf.touch()

    out_dir = tmp_path / "outside_out"

    # Mock extract_pages so we don't fail PDF loading before the boundary check
    from PIL import Image

    from matera.data import extract

    dummy_page = extract.RenderedPage(1, 10, 10, 10.0, 10.0, Image.new("RGB", (10, 10)))
    monkeypatch.setattr(extract, "extract_pages", lambda *args, **kwargs: iter([dummy_page]))

    # Mock Path.cwd to return our fake_repo
    # Need to be careful to only patch cwd on pathlib.Path
    original_cwd = Path.cwd
    monkeypatch.setattr(Path, "cwd", lambda: fake_repo)

    monkeypatch.setattr(
        sys, "argv", ["extract.py", "--input", str(in_pdf), "--output", str(out_dir)]
    )

    try:
        assert main() == 1
        out, err = capsys.readouterr()
        assert "Input file must be within the repository root" in err
    finally:
        # Restore cwd to avoid messing up other tests
        monkeypatch.setattr(Path, "cwd", original_cwd)


def test_cli_run_from_different_cwd(
    tmp_path: Path, sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
):
    out_dir = tmp_path / "out"

    # Change working directory to a subfolder of the repository (e.g. data/pdfs)
    import os

    abs_in = sample_pdf.resolve()
    original_cwd = os.getcwd()
    try:
        os.chdir(sample_pdf.parent)
        # We need absolute paths because we changed cwd
        monkeypatch.setattr(
            sys, "argv", ["extract.py", "--input", str(abs_in), "--output", str(out_dir)]
        )
        assert main() == 0
        assert out_dir.exists()
    finally:
        os.chdir(original_cwd)

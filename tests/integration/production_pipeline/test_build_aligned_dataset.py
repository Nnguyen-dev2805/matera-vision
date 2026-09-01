import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_build_aligned_dataset_atomic_rollback(tmp_path: Path):
    """
    Test that if an error occurs during alignment processing (e.g. invalid config or missing file),
    the output directory is not partially written, and the temporary directory is cleaned up.
    """
    # Create mock directories
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    # Create 9 valid pages, but missing the 10th page to force a failure midway
    for i in range(1, 10):
        img = Image.new("RGB", (300, 300), color="white")
        img.save(pages_dir / f"page_{i}.png")

    ref_path = tmp_path / "reference.png"
    Image.new("RGB", (300, 300), color="white").save(ref_path)

    # Use actual profiles from repo
    repo_root = Path(__file__).parents[3]
    semantic_path = repo_root / "profiles/semantic.json"
    layout_path = repo_root / "profiles/layout.json"

    out_dir = tmp_path / "aligned_out"

    script_path = repo_root / "scripts/dataset_tools/build_aligned_dataset.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--pages-dir",
        str(pages_dir),
        "--reference",
        str(ref_path),
        "--semantic",
        str(semantic_path),
        "--layout",
        str(layout_path),
        "--output-dir",
        str(out_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # The script should fail because page_10.png is missing
    assert result.returncode != 0
    assert "Source page image not found" in result.stderr

    # The output directory should NOT exist (atomic rollback)
    assert not out_dir.exists()

    # Also verify no temporary directories starting with 'aligned_' were left behind
    tmp_dirs = list(tmp_path.glob("aligned_*"))
    assert len(tmp_dirs) == 0


def test_build_aligned_dataset_smoke(tmp_path: Path):
    """
    Smoke test the dataset generation script with dummy data to ensure
    it produces the 10 aligned pages and 10 debug overlays atomically.
    """
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    # Create 10 valid pages (dummy random noise to satisfy ORB)
    import numpy as np

    np.random.seed(42)
    img_arr = np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)
    img_arr[300:500, 300:500] = 255  # some strong features

    for i in range(1, 11):
        Image.fromarray(img_arr, mode="RGB").save(pages_dir / f"page_{i}.png")

    ref_path = tmp_path / "reference.png"
    Image.fromarray(img_arr, mode="RGB").save(ref_path)

    repo_root = Path(__file__).parents[3]
    semantic_path = repo_root / "profiles/semantic.json"
    layout_path = repo_root / "profiles/layout.json"

    out_dir = tmp_path / "aligned_out"

    script_path = repo_root / "scripts/dataset_tools/build_aligned_dataset.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--pages-dir",
        str(pages_dir),
        "--reference",
        str(ref_path),
        "--semantic",
        str(semantic_path),
        "--layout",
        str(layout_path),
        "--output-dir",
        str(out_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # The script should succeed
    assert result.returncode == 0

    # The output directory should exist
    assert out_dir.exists()

    # It should contain exactly 20 files (10 aligned, 10 debug)
    output_files = list(out_dir.glob("*.png"))
    assert len(output_files) == 20

    aligned_files = list(out_dir.glob("aligned_page_*.png"))
    assert len(aligned_files) == 10

    debug_files = list(out_dir.glob("debug_aligned_page_*.png"))
    assert len(debug_files) == 10

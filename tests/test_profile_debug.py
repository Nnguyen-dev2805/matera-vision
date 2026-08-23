import json
from pathlib import Path

from PIL import Image

from matera.tools.profile_debug import main


def create_mock_profiles(tmp_path: Path):
    sem_path = tmp_path / "semantic.json"
    lay_path = tmp_path / "layout.json"

    sem_path.write_text(
        json.dumps(
            {
                "form_id": "f",
                "form_version": "v",
                "questions": [
                    {
                        "question_id": "q1",
                        "response_type": "single_select",
                        "mark_strategy": "circle",
                        "options": [{"option_id": "o1"}],
                        "max_selections": 1,
                    }
                ],
            }
        )
    )

    lay_path.write_text(
        json.dumps(
            {
                "form_id": "f",
                "form_version": "v",
                "coordinate_space": "absolute_pixel",
                "reference_dpi": 200,
                "pages": [
                    {
                        "page_number": 1,
                        "width_px": 500,
                        "height_px": 500,
                        "anchors": [
                            {
                                "anchor_id": "a1",
                                "anchor_type": "qr",
                                "bbox": {"x": 10, "y": 10, "w": 50, "h": 50},
                            }
                        ],
                        "rois": [
                            {
                                "question_id": "q1",
                                "option_id": "o1",
                                "bbox": {"x": 100, "y": 100, "w": 20, "h": 20},
                            }
                        ],
                    },
                    {
                        "page_number": 2,
                        "width_px": 500,
                        "height_px": 500,
                        "anchors": [],
                        "rois": [
                            {
                                "question_id": "q1",
                                "option_id": "o1",
                                "bbox": {"x": 100, "y": 100, "w": 20, "h": 20},
                            }
                        ],
                    },
                ],
            }
        )
    )

    return sem_path, lay_path


def test_profile_debug_all_pages(tmp_path: Path, monkeypatch):
    sem_path, lay_path = create_mock_profiles(tmp_path)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        [
            "profile_debug.py",
            "--semantic",
            str(sem_path),
            "--layout",
            str(lay_path),
            "--output-dir",
            str(out_dir),
        ],
    )

    assert main() == 0
    assert (out_dir / "overlay_page_001.png").exists()
    assert (out_dir / "overlay_page_002.png").exists()


def test_profile_debug_specific_page(tmp_path: Path, monkeypatch):
    sem_path, lay_path = create_mock_profiles(tmp_path)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        [
            "profile_debug.py",
            "--semantic",
            str(sem_path),
            "--layout",
            str(lay_path),
            "--output-dir",
            str(out_dir),
            "--page-number",
            "2",
        ],
    )

    assert main() == 0
    assert not (out_dir / "overlay_page_001.png").exists()
    assert (out_dir / "overlay_page_002.png").exists()


def test_profile_debug_with_background_image(tmp_path: Path, monkeypatch):
    sem_path, lay_path = create_mock_profiles(tmp_path)
    out_dir = tmp_path / "out"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create a dummy background image with L mode (grayscale)
    bg_path = img_dir / "page_1.png"
    img = Image.new("L", (1000, 1000), color=128)
    img.save(bg_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "profile_debug.py",
            "--semantic",
            str(sem_path),
            "--layout",
            str(lay_path),
            "--output-dir",
            str(out_dir),
            "--images-dir",
            str(img_dir),
            "--page-number",
            "1",
        ],
    )

    assert main() == 0

    out_path = out_dir / "overlay_page_001.png"
    assert out_path.exists()

    # Assert output mode and size matches background
    out_img = Image.open(out_path)
    assert out_img.mode == "L"
    assert out_img.size == (1000, 1000)


def test_profile_debug_invalid_profile(tmp_path: Path, monkeypatch, capsys):
    sem_path, lay_path = create_mock_profiles(tmp_path)
    out_dir = tmp_path / "out"

    # Break layout JSON
    lay_path.write_text("{}")

    monkeypatch.setattr(
        "sys.argv",
        [
            "profile_debug.py",
            "--semantic",
            str(sem_path),
            "--layout",
            str(lay_path),
            "--output-dir",
            str(out_dir),
        ],
    )

    assert main() == 1
    assert not out_dir.exists()

    captured = capsys.readouterr()
    assert "Error loading layout profile" in captured.err


def test_profile_debug_colors(tmp_path: Path, monkeypatch):
    sem_path, lay_path = create_mock_profiles(tmp_path)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        [
            "profile_debug.py",
            "--semantic",
            str(sem_path),
            "--layout",
            str(lay_path),
            "--output-dir",
            str(out_dir),
            "--page-number",
            "1",
        ],
    )

    assert main() == 0
    out_path = out_dir / "overlay_page_001.png"
    assert out_path.exists()

    img = Image.open(out_path).convert("RGB")
    # Verify page bounds (green) at top-left edge
    assert (
        img.getpixel((0, 0)) == (0, 128, 0)
        or img.getpixel((0, 0)) == (0, 255, 0)
        or img.getpixel((0, 0))[:3] == (0, 128, 0)
    )

    # Verify anchor (blue) on its bbox outline (e.g. at x=10, y=10)
    assert img.getpixel((10, 10)) == (0, 0, 255)

    # Verify ROI (red) on its bbox outline (e.g. at x=100, y=100)
    assert img.getpixel((100, 100)) == (255, 0, 0)


def test_profile_debug_missing_image(tmp_path: Path, monkeypatch, capsys):
    sem_path, lay_path = create_mock_profiles(tmp_path)
    out_dir = tmp_path / "out"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    monkeypatch.setattr(
        "sys.argv",
        [
            "profile_debug.py",
            "--semantic",
            str(sem_path),
            "--layout",
            str(lay_path),
            "--output-dir",
            str(out_dir),
            "--images-dir",
            str(img_dir),
        ],
    )

    assert main() == 1
    captured = capsys.readouterr()
    assert "Error: Background image not found" in captured.err

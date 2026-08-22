import pytest
from PIL import Image

from matera.data.contracts import (
    ExtractionError,
    ExtractionManifest,
    ImageLibraryInfo,
    PageArtifact,
    RenderConfig,
    RenderedPage,
    RendererInfo,
)

VALID_HASH = "a" * 64


def test_rendered_page_valid():
    img = Image.new("RGB", (100, 100))
    page = RenderedPage(1, 100, 100, 50.0, 50.0, img)
    assert page.page_number == 1


def test_rendered_page_invalid():
    with pytest.raises(ValueError, match="page_number must be a positive int"):
        RenderedPage("1", 100, 100, 50.0, 50.0, Image.new("RGB", (100, 100)))  # type: ignore
    with pytest.raises(ValueError, match="page_number must be a positive int"):
        RenderedPage(0, 100, 100, 50.0, 50.0, Image.new("RGB", (100, 100)))
    with pytest.raises(ValueError, match="image must be a PIL Image"):
        RenderedPage(1, 100, 100, 50.0, 50.0, "not_an_image")  # type: ignore
    with pytest.raises(ValueError, match="image dimensions must match width_px and height_px"):
        RenderedPage(1, 200, 100, 50.0, 50.0, Image.new("RGB", (100, 100)))


def test_page_artifact_invalid():
    with pytest.raises(ValueError, match="filename must be a non-empty string"):
        PageArtifact(1, "", VALID_HASH, 100, 100, 50.0, 50.0)
    with pytest.raises(ValueError, match="filename must not contain directory separators"):
        PageArtifact(1, "foo/bar.png", VALID_HASH, 100, 100, 50.0, 50.0)
    with pytest.raises(
        ValueError, match="image_sha256 must be a 64-character lowercase hex string"
    ):
        PageArtifact(1, "file.png", "not_a_hash", 100, 100, 50.0, 50.0)
    with pytest.raises(ValueError, match="width_px must be a positive int"):
        PageArtifact(1, "file.png", VALID_HASH, 0, 100, 50.0, 50.0)


def test_render_config_invalid():
    with pytest.raises(ValueError, match="dpi must be 300"):
        RenderConfig(150, "RGB", 6, False, True)
    with pytest.raises(ValueError, match="mode must be 'RGB'"):
        RenderConfig(300, "L", 6, False, True)
    with pytest.raises(ValueError, match="compress_level must be 6"):
        RenderConfig(300, "RGB", 10, False, True)
    with pytest.raises(ValueError, match="optimize must be False"):
        RenderConfig(300, "RGB", 6, True, True)
    with pytest.raises(ValueError, match="strip_metadata must be True"):
        RenderConfig(300, "RGB", 6, False, False)

def test_metadata_info_invalid():
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        RendererInfo("", "1")
    with pytest.raises(ValueError, match="version must be a non-empty string"):
        ImageLibraryInfo("pillow", "")


def test_extraction_manifest_invalid():
    artifact = PageArtifact(1, "f.png", VALID_HASH, 100, 100, 50.0, 50.0)
    artifact2 = PageArtifact(2, "f2.png", VALID_HASH, 100, 100, 50.0, 50.0)
    renderer = RendererInfo("a", "1")
    lib = ImageLibraryInfo("b", "1")
    config = RenderConfig(300, "RGB", 6, False, True)

    with pytest.raises(ValueError, match="manifest_version must be '1'"):
        ExtractionManifest("2", "src", VALID_HASH, 1, (artifact,), renderer, lib, config, "time")  # type: ignore

    with pytest.raises(ValueError, match="source_path must be repository-relative and safe"):
        ExtractionManifest(
            "1", "/absolute/path", VALID_HASH, 1, (artifact,), renderer, lib, config, "time"
        )

    with pytest.raises(
        ValueError, match="source_sha256 must be a 64-character lowercase hex string"
    ):
        ExtractionManifest("1", "src", "bad", 1, (artifact,), renderer, lib, config, "time")

    with pytest.raises(ValueError, match="All elements in pages must be instances of PageArtifact"):
        ExtractionManifest(
            "1", "src", VALID_HASH, 1, ("not_artifact",), renderer, lib, config, "time"
        )  # type: ignore

    with pytest.raises(ValueError, match=r"length of pages \(0\) must match page_count \(1\)"):
        ExtractionManifest("1", "src", VALID_HASH, 1, (), renderer, lib, config, "time")

    with pytest.raises(
        ValueError, match="Expected page_number 1, got 2. Pages must be contiguous and sorted."
    ):
        ExtractionManifest(
            "1", "src", VALID_HASH, 2, (artifact2, artifact), renderer, lib, config, "2026-01-01T00:00:00Z"
        )

    with pytest.raises(ValueError, match="generated_at must be a valid ISO-8601 string"):
        ExtractionManifest("1", "src", VALID_HASH, 1, (artifact,), renderer, lib, config, "not-a-time")


def test_extraction_error():
    err = ExtractionError("doc.pdf", None, "CORRUPT", "Bad PDF")
    assert str(err) == "doc.pdf - CORRUPT: Bad PDF"
    assert err.page_number is None

    err2 = ExtractionError("doc.pdf", 5, "RENDER_FAIL", "OOM")
    assert str(err2) == "doc.pdf (page 5) - RENDER_FAIL: OOM"

import os
import re
import math
from datetime import datetime
from dataclasses import dataclass
from typing import Literal
from PIL import Image

SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    width_px: int
    height_px: int
    pdf_width_pt: float
    pdf_height_pt: float
    image: Image.Image

    def __post_init__(self) -> None:
        if type(self.page_number) is not int or self.page_number <= 0:
            raise ValueError("page_number must be a positive int")
        if type(self.width_px) is not int or self.width_px <= 0:
            raise ValueError("width_px must be a positive int")
        if type(self.height_px) is not int or self.height_px <= 0:
            raise ValueError("height_px must be a positive int")
        if type(self.pdf_width_pt) is not float or not math.isfinite(self.pdf_width_pt) or self.pdf_width_pt <= 0:
            raise ValueError("pdf dimensions must be positive finite floats")
        if type(self.pdf_height_pt) is not float or not math.isfinite(self.pdf_height_pt) or self.pdf_height_pt <= 0:
            raise ValueError("pdf dimensions must be positive finite floats")
        if not isinstance(self.image, Image.Image):
            raise ValueError("image must be a PIL Image")
        if self.image.size != (self.width_px, self.height_px):
            raise ValueError("image dimensions must match width_px and height_px")


@dataclass(frozen=True)
class PageArtifact:
    page_number: int
    filename: str
    image_sha256: str
    width_px: int
    height_px: int
    pdf_width_pt: float
    pdf_height_pt: float

    def __post_init__(self) -> None:
        if type(self.page_number) is not int or self.page_number <= 0:
            raise ValueError("page_number must be a positive int")
        if type(self.filename) is not str or not self.filename.strip():
            raise ValueError("filename must be a non-empty string")
        if "/" in self.filename or "\\" in self.filename or ".." in self.filename:
            raise ValueError(
                "filename must not contain directory separators or traversal characters"
            )
        if type(self.image_sha256) is not str or not SHA256_PATTERN.match(self.image_sha256):
            raise ValueError("image_sha256 must be a 64-character lowercase hex string")
        if type(self.width_px) is not int or self.width_px <= 0:
            raise ValueError("width_px must be a positive int")
        if type(self.height_px) is not int or self.height_px <= 0:
            raise ValueError("height_px must be a positive int")
        if type(self.pdf_width_pt) is not float or not math.isfinite(self.pdf_width_pt) or self.pdf_width_pt <= 0:
            raise ValueError("pdf dimensions must be positive finite floats")
        if type(self.pdf_height_pt) is not float or not math.isfinite(self.pdf_height_pt) or self.pdf_height_pt <= 0:
            raise ValueError("pdf dimensions must be positive finite floats")


@dataclass(frozen=True)
class RendererInfo:
    name: str
    version: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if type(self.version) is not str or not self.version.strip():
            raise ValueError("version must be a non-empty string")

@dataclass(frozen=True)
class ImageLibraryInfo:
    name: str
    version: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if type(self.version) is not str or not self.version.strip():
            raise ValueError("version must be a non-empty string")


@dataclass(frozen=True)
class RenderConfig:
    dpi: int
    mode: str
    compress_level: int
    optimize: bool
    strip_metadata: bool

    def __post_init__(self) -> None:
        if self.dpi != 300:
            raise ValueError("dpi must be 300")
        if self.mode != "RGB":
            raise ValueError("mode must be 'RGB'")
        if self.compress_level != 6:
            raise ValueError("compress_level must be 6")
        if self.optimize is not False:
            raise ValueError("optimize must be False")
        if self.strip_metadata is not True:
            raise ValueError("strip_metadata must be True")


@dataclass(frozen=True)
class ExtractionManifest:
    manifest_version: Literal["1"]
    source_path: str
    source_sha256: str
    page_count: int
    pages: tuple[PageArtifact, ...]
    renderer: RendererInfo
    image_library: ImageLibraryInfo
    render_config: RenderConfig
    generated_at: str

    def __post_init__(self) -> None:
        if self.manifest_version != "1":
            raise ValueError("manifest_version must be '1'")
        if type(self.source_path) is not str or not self.source_path.strip():
            raise ValueError("source_path must be a non-empty string")
        if os.path.isabs(self.source_path) or ".." in self.source_path:
            raise ValueError("source_path must be repository-relative and safe")
        if type(self.source_sha256) is not str or not SHA256_PATTERN.match(self.source_sha256):
            raise ValueError("source_sha256 must be a 64-character lowercase hex string")
        if type(self.page_count) is not int or self.page_count <= 0:
            raise ValueError("page_count must be a positive int")
        if not isinstance(self.pages, tuple):
            raise ValueError("pages must be a tuple")
        if len(self.pages) != self.page_count:
            raise ValueError(
                f"length of pages ({len(self.pages)}) must match page_count ({self.page_count})"
            )

        expected_page = 1
        for p in self.pages:
            if not isinstance(p, PageArtifact):
                raise ValueError("All elements in pages must be instances of PageArtifact")
            if p.page_number != expected_page:
                raise ValueError(
                    f"Expected page_number {expected_page}, got {p.page_number}. Pages must be contiguous and sorted."
                )
            expected_page += 1

        try:
            datetime.fromisoformat(self.generated_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            raise ValueError("generated_at must be a valid ISO-8601 string")


class ExtractionError(Exception):
    def __init__(self, source_path: str, page_number: int | None, error_code: str, reason: str):
        page_ctx = f" (page {page_number})" if page_number is not None else ""
        super().__init__(f"{source_path}{page_ctx} - {error_code}: {reason}")
        self.source_path = source_path
        self.page_number = page_number
        self.error_code = error_code
        self.reason = reason

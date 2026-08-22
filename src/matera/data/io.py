import dataclasses
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from matera.data.contracts import ExtractionManifest


def save_image_lossless(image: Image.Image, out_path: Path) -> None:
    """
    Saves an image as PNG with strict parameters to guarantee byte-for-byte reproducibility
    across identical OS/Dependency environments.
    """
    # Enforce RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(
        out_path,
        format="PNG",
        optimize=False,
        compress_level=6,
        exif=b"",
        icc_profile=None,
    )


class _ManifestEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)


def atomic_write_manifest(manifest: ExtractionManifest, out_path: Path) -> None:
    """
    Atomically writes the ExtractionManifest to the target path.
    Ensures canonical JSON serialization with sort_keys=True and indent=2.
    """
    # Serialize to memory first to ensure no partial writes on error
    data = json.dumps(
        manifest,
        cls=_ManifestEncoder,
        sort_keys=True,
        indent=2,
    )

    # Write to a temporary file in the same directory, then replace
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=".manifest_", suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.write("\n")  # Add trailing newline
        # Atomically replace
        shutil.move(tmp_path, out_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

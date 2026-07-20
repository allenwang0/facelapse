"""Ingest: enumerate a folder, decode to RGB (HEIC included), content-hash.

Never downscale here. Original resolution feeds the sharpness and face-size
metrics; downscaling happens only at render.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageOps

# Register HEIC/HEIF so Pillow can open iCloud exports. If pillow-heif is
# absent, HEIC files are simply skipped with a clear error rather than a
# silent decode failure.
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False


def content_hash(path: Path) -> str:
    """Hash of raw file bytes. Identity of the CACHE ENTRY, not of the image
    content -- re-saved duplicates hash differently and are caught later by
    perceptual hashing in dedup.py."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_images(folder: str, exts: tuple[str, ...]) -> Iterator[Path]:
    root = Path(folder)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def decode(path: Path) -> np.ndarray:
    """Return an HxWx3 uint8 RGB array. Applies EXIF orientation so faces
    aren't sideways. Raises on unreadable files (fail loud)."""
    if path.suffix.lower() in (".heic", ".heif") and not HEIF_OK:
        raise RuntimeError(f"HEIC file but pillow-heif not installed: {path}")
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)   # respect camera rotation flag
    img = img.convert("RGB")
    return np.asarray(img)

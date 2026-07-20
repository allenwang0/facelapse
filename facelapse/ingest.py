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


def content_hash(path: Path, max_size_mb: int = 200) -> str:
    """Hash of raw file bytes. Identity of the CACHE ENTRY, not of the image
    content -- re-saved duplicates hash differently and are caught later by
    perceptual hashing in dedup.py.

    max_size_mb: reject files larger than this to prevent memory exhaustion.
    """
    size_bytes = path.stat().st_size
    if size_bytes > max_size_mb * 1024 * 1024:
        raise ValueError(f"File too large: {size_bytes / 1024 / 1024:.1f}MB > {max_size_mb}MB limit")

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_images(folder: str, exts: tuple[str, ...]) -> Iterator[Path]:
    """Enumerate images in folder, with path traversal protection."""
    root = Path(folder).resolve()  # resolve symlinks and make absolute
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        # Verify path is within root to prevent symlink escapes
        try:
            p_resolved = p.resolve()
            p_resolved.relative_to(root)  # raises ValueError if outside root
        except (ValueError, OSError):
            continue
        if p.suffix.lower() in exts:
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

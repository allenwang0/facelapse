"""Ingest: enumerate a folder, decode to RGB (HEIC included), content-hash.

Never downscale here. Original resolution feeds the sharpness and face-size
metrics; downscaling happens only at render.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageOps

# Register HEIC/HEIF so Pillow can open iCloud exports. If pillow-heif is
# absent, HEIC files are simply skipped with a clear error rather than a
# silent decode failure.
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    HEIF_OK = True
except ImportError:
    HEIF_OK = False


MAX_SIZE_MB = 200


def too_large(size_bytes: int, max_size_mb: int = MAX_SIZE_MB) -> str | None:
    """Error message when a file is over the size limit, which guards
    against memory exhaustion; None otherwise."""
    if size_bytes > max_size_mb * 1024 * 1024:
        return f"File too large: {size_bytes / 1024 / 1024:.1f}MB > {max_size_mb}MB limit"
    return None


def content_hash(path: Path, max_size_mb: int = MAX_SIZE_MB) -> str:
    """Hash of raw file bytes. Identity of the CACHE ENTRY, not of the image
    content -- re-saved duplicates hash differently and are caught later by
    perceptual hashing in dedup.py."""
    err = too_large(path.stat().st_size, max_size_mb)
    if err:
        raise ValueError(err)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ref_key(ref: str, size: int) -> str:
    """Cache key for a file whose bytes were not hashed (over the size limit)."""
    return "unhashed:" + hashlib.sha256(f"{ref}|{size}".encode()).hexdigest()


DECODE_ERRORS = (OSError, ValueError, RuntimeError, Image.DecompressionBombError)


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


ZIP_SEP = "::"


@dataclass
class Source:
    """One image to analyze, from a folder or a Takeout zip. ref is what the
    cache stores as the path: a file path, or "archive.zip::member"."""

    ref: str
    name: str
    size: int
    mtime: float | None
    sidecar: dict[str, Any] | None

    def read(self) -> bytes:
        return _read(self.ref)


def _read(ref: str) -> bytes:
    if ZIP_SEP in ref:
        archive, member = ref.split(ZIP_SEP, 1)
        with zipfile.ZipFile(archive) as zf:
            return zf.read(member)
    return Path(ref).read_bytes()


def _open(path: Path | str) -> Path | io.BytesIO:
    ref = str(path)
    return io.BytesIO(_read(ref)) if ZIP_SEP in ref else Path(ref)


def iter_sources(paths: list[str], exts: tuple[str, ...]) -> Iterator[Source]:
    """Folders are walked (with the symlink-escape check); .zip files are
    streamed member by member without extracting, and each image is paired
    with its Takeout sidecar JSON, which may sit in another zip of the same
    export."""
    from .timestamps import load_takeout_sidecar, takeout_sidecar_names

    zips = [p for p in paths if p.lower().endswith(".zip")]
    sidecars: dict[str, tuple[str, str]] = {}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".json"):
                    sidecars[name] = (z, name)
    for p in paths:
        if p in zips:
            with zipfile.ZipFile(p) as zf:
                for info in sorted(zf.infolist(), key=lambda i: i.filename):
                    if info.is_dir() or Path(info.filename).suffix.lower() not in exts:
                        continue
                    sidecar = None
                    for cand in takeout_sidecar_names(info.filename):
                        if cand in sidecars:
                            z, member = sidecars[cand]
                            sidecar = _json_member(z, member)
                            break
                    yield Source(
                        ref=f"{p}{ZIP_SEP}{info.filename}",
                        name=Path(info.filename).name,
                        size=info.file_size,
                        mtime=datetime(*info.date_time).timestamp(),
                        sidecar=sidecar,
                    )
        else:
            for f in iter_images(p, exts):
                st = f.stat()
                yield Source(str(f), f.name, st.st_size, st.st_mtime, load_takeout_sidecar(f))


def _json_member(archive: str, member: str) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(archive) as zf:
            data = json.loads(zf.read(member))
    except (OSError, ValueError, KeyError):
        return None
    return data if isinstance(data, dict) else None


def decode(path: Path | str) -> np.ndarray:
    """Return an HxWx3 uint8 RGB array. Applies EXIF orientation so faces
    aren't sideways. Raises on unreadable files (fail loud). Accepts a file
    path or a "archive.zip::member" reference."""
    return decode_image(Image.open(_open(path)), str(path))


def decode_image(img: Image.Image, name: str) -> np.ndarray:
    if Path(name).suffix.lower() in (".heic", ".heif") and not HEIF_OK:
        raise RuntimeError(f"HEIC file but pillow-heif not installed: {name}")
    img = ImageOps.exif_transpose(img)  # respect camera rotation flag
    img = img.convert("RGB")
    return np.asarray(img)


def thumbnail(path: Path, max_w: int) -> tuple[np.ndarray, float]:
    """Decode at most max_w pixels wide without holding the full-resolution
    image. Image.draft lets the JPEG decoder downscale by 2/4/8 while
    decoding. Returns the RGB array and its scale relative to the oriented
    original, for mapping face boxes onto it."""
    img = Image.open(_open(path))
    img.draft("RGB", (max_w, max_w * 4))
    img = ImageOps.exif_transpose(img).convert("RGB")
    full_w = _oriented_width(path)
    if img.width > max_w:
        img = img.resize((max_w, max(1, round(img.height * max_w / img.width))))
    return np.asarray(img), img.width / full_w


def _oriented_width(path: Path) -> int:
    with Image.open(_open(path)) as img:
        w, h = img.size
        if img.getexif().get(0x0112) in (5, 6, 7, 8):
            return h
        return w

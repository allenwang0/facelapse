"""Shared builders for synthetic test records."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from PIL import Image

from facelapse.records import Face


def make_face(**kw: Any) -> Face:
    base: dict[str, Any] = dict(
        content_hash="h",
        face_id=0,
        bbox=np.array([0, 0, 100, 100], np.float32),
        kps=np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], np.float32),
        embedding=np.ones(512, np.float32) / np.sqrt(512),
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        det_score=0.9,
    )
    base.update(kw)
    return Face(**base)


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def embedding_with_similarity(ref: np.ndarray, sim: float, seed: int) -> np.ndarray:
    """A unit vector whose cosine similarity to unit vector ref is exactly sim."""
    # a separate stream, so the direction can never coincide with a ref
    # drawn from default_rng(seed)
    rng = np.random.default_rng([seed, 7919])
    u = rng.normal(size=ref.shape)
    u -= (u @ ref) * ref
    u /= np.linalg.norm(u)
    return unit(sim * ref + np.sqrt(max(0.0, 1 - sim**2)) * u)


def exif_with_date(when: datetime, offset: str | None = None) -> Image.Exif:
    exif = Image.Exif()
    ifd = exif.get_ifd(0x8769)
    ifd[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")  # DateTimeOriginal
    if offset:
        ifd[0x9011] = offset  # OffsetTimeOriginal
    return exif


def write_image(
    path: Any,
    seed: int,
    size: tuple[int, int] = (160, 160),
    exif_date: datetime | None = None,
) -> None:
    """Noise image distinct enough per seed that pHash never groups two.
    exif_date writes DateTimeOriginal (JPEG only)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize(size, Image.Resampling.NEAREST)
    if exif_date is not None:
        img.save(path, exif=exif_with_date(exif_date), quality=95)
    else:
        img.save(path)

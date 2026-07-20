"""Perceptual-hash dedup. Content-hashing in ingest catches byte-identical
files; this catches re-saved / re-compressed copies that carry different
bytes (and sometimes different timestamps) but the same picture, so one
physical photo can't sneak into two buckets.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
import imagehash


def phash(rgb: np.ndarray) -> imagehash.ImageHash:
    return imagehash.phash(Image.fromarray(rgb))


def group_duplicates(hashes: dict[str, imagehash.ImageHash], max_hamming: int) -> dict[str, str]:
    """Map each content_hash to a canonical representative content_hash.
    Greedy union by Hamming distance; fine for archive-scale sets."""
    items = list(hashes.items())
    canonical: dict[str, str] = {}
    reps: list[tuple[str, imagehash.ImageHash]] = []
    for ch, h in items:
        matched = None
        for rep_ch, rep_h in reps:
            if (h - rep_h) <= max_hamming:
                matched = rep_ch
                break
        if matched is None:
            reps.append((ch, h))
            canonical[ch] = ch
        else:
            canonical[ch] = matched
    return canonical

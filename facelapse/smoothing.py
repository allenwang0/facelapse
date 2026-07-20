"""Smooth the SEQUENCE of alignment parameters before warping, to remove the
one-or-two-pixel per-frame landmark-detector jitter while preserving the slow
real drift. Same idea as video stabilization, applied to the transform
parameters rather than to camera motion.

Angle is unwrapped before averaging so a wrap across +/-pi doesn't produce a
spurious spin.
"""
from __future__ import annotations

import numpy as np

from .align import SimilarityParams


def smooth_params(seq: list[SimilarityParams], window: int) -> list[SimilarityParams]:
    if window <= 1 or len(seq) < 3:
        return seq
    w = window if window % 2 == 1 else window + 1

    angles = np.unwrap(np.array([p.angle for p in seq]))
    scales = np.array([p.scale for p in seq])
    txs = np.array([p.tx for p in seq])
    tys = np.array([p.ty for p in seq])

    def ma(a: np.ndarray) -> np.ndarray:
        pad = w // 2
        ap = np.pad(a, pad, mode="edge")
        kernel = np.ones(w) / w
        return np.convolve(ap, kernel, mode="valid")

    a2, s2, x2, y2 = ma(angles), ma(scales), ma(txs), ma(tys)
    return [SimilarityParams(float(a), float(s), float(x), float(y))
            for a, s, x, y in zip(a2, s2, x2, y2)]

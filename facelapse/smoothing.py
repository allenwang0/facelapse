"""Smooth the SEQUENCE of alignments before warping, to remove the
one-or-two-pixel per-frame landmark-detector jitter while preserving the slow
real drift.

Each frame's transform maps that photo's own pixel coordinates to the canvas,
so the raw parameters are not comparable across photos. Smoothing therefore
happens in canvas space: for every frame we measure where its eye midpoint,
interocular scale and eye-line angle land relative to the canonical target
(exactly on target for a 2-point fit, off target only when the nasion term
pulls the least-squares fit), average those residuals over the window, and
compose the difference back onto the frame's own transform.

Angle residuals are unwrapped before averaging so a wrap across +/-pi doesn't
produce a spurious spin.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .align import SimilarityParams

Point = tuple[float, float]
Array = NDArray[np.float64]


def _moving_average(a: Array, window: int) -> Array:
    pad = window // 2
    padded = np.pad(a, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def _canvas_geometry(p: SimilarityParams, eyes: Array) -> tuple[Array, float, float]:
    M = p.matrix()
    pts = eyes @ M[:, :2].T + M[:, 2]
    v = pts[1] - pts[0]
    return (pts[0] + pts[1]) / 2.0, math.log(math.hypot(v[0], v[1])), math.atan2(v[1], v[0])


def smooth_params(
    seq: list[SimilarityParams],
    eyes: Sequence[NDArray[np.floating]],
    target: tuple[Point, Point],
    window: int,
) -> list[SimilarityParams]:
    """seq[i] maps frame i's source pixels to the canvas; eyes[i] is that
    frame's (2, 2) source eye centers; target is the canonical (left, right)
    eye position on the canvas."""
    if window <= 1 or len(seq) < 3:
        return seq
    w = window if window % 2 == 1 else window + 1

    t_pts = np.asarray(target, dtype=np.float64)
    t_mid, t_logscale, t_angle = _canvas_geometry(SimilarityParams(0.0, 1.0, 0.0, 0.0), t_pts)

    geo = [_canvas_geometry(p, np.asarray(e[:2], dtype=np.float64)) for p, e in zip(seq, eyes)]
    mids = np.array([g[0] for g in geo])
    logscales = np.array([g[1] for g in geo])
    angles = np.unwrap(np.array([g[2] for g in geo]))

    sm_dx = _moving_average(mids[:, 0] - t_mid[0], w)
    sm_dy = _moving_average(mids[:, 1] - t_mid[1], w)
    sm_ls = _moving_average(logscales - t_logscale, w)
    sm_an = _moving_average(angles - t_angle, w)

    out: list[SimilarityParams] = []
    for i, p in enumerate(seq):
        d_angle = float(t_angle + sm_an[i] - angles[i])
        d_scale = math.exp(float(t_logscale + sm_ls[i] - logscales[i]))
        want_mid = t_mid + np.array([sm_dx[i], sm_dy[i]])
        c, s = math.cos(d_angle) * d_scale, math.sin(d_angle) * d_scale
        R = np.array([[c, -s], [s, c]])
        t = R @ (np.array([p.tx, p.ty]) - mids[i]) + want_mid
        out.append(SimilarityParams(p.angle + d_angle, p.scale * d_scale, float(t[0]), float(t[1])))
    return out

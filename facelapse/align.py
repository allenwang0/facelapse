"""Alignment: warp each face into a fixed canonical frame so successive
frames are directly comparable. This is what makes the video read as one
face changing rather than a flipbook.

We use a SIMILARITY transform (rotation + uniform scale + translation, 4 DOF,
no shear) and align on the most temporally STABLE points only: the two eye
centers, optionally plus the nasion (nose bridge). Two eye points alone
exactly determine a 4-DOF similarity; adding the nasion overdetermines it and
least-squares stabilizes against per-point jitter.

Deliberately NOT full-face Procrustes: aligning on many points cancels real
facial change (a landmark that moved because your face genuinely changed gets
partly undone to minimize total displacement). Eyes + nasion barely move with
expression or moderate aging, so everything below them renders honestly.

Returns the transform PARAMETERS (angle, scale, tx, ty) as well as applying
them, so the sequence of parameters can be smoothed (smoothing.py) before the
final warp.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .records import Face


@dataclass
class SimilarityParams:
    angle: float   # radians
    scale: float
    tx: float
    ty: float

    def matrix(self) -> np.ndarray:
        c, s = math.cos(self.angle) * self.scale, math.sin(self.angle) * self.scale
        return np.array([[c, -s, self.tx], [s, c, self.ty]], dtype=np.float64)


def _decompose(M: np.ndarray) -> SimilarityParams:
    a, b = M[0, 0], M[1, 0]
    scale = math.hypot(a, b)
    angle = math.atan2(b, a)
    return SimilarityParams(angle=angle, scale=scale, tx=float(M[0, 2]), ty=float(M[1, 2]))


def source_points(f: Face, use_nasion: bool) -> np.ndarray:
    """Eye centers (kps[0], kps[1]); append nasion from dense landmarks if
    available and requested."""
    pts = [f.kps[0], f.kps[1]]
    if use_nasion and f.dense is not None:
        from .landmarks import nasion_point
        pts.append(nasion_point(f.dense))
    return np.asarray(pts, dtype=np.float32)


def target_points(cfg, with_nasion: bool) -> np.ndarray:
    le = np.array(cfg.left_eye_xy, dtype=np.float32)
    re = np.array(cfg.right_eye_xy, dtype=np.float32)
    pts = [le, re]
    if with_nasion:
        # canonical nasion: midpoint of eyes, raised slightly toward brow.
        mid = 0.5 * (le + re)
        eye_dist = np.linalg.norm(re - le)
        nas = mid + np.array([0.0, -0.15 * eye_dist], dtype=np.float32)
        pts.append(nas)
    return np.asarray(pts, dtype=np.float32)


def estimate(f: Face, cfg) -> SimilarityParams:
    with_nasion = cfg.use_nasion and f.dense is not None
    src = source_points(f, with_nasion)
    dst = target_points(cfg, with_nasion)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        # fall back to eyes-only closed form if the estimator fails
        M, _ = cv2.estimateAffinePartial2D(f.kps[:2].astype(np.float32),
                                           target_points(cfg, False), method=cv2.LMEDS)
    return _decompose(np.asarray(M, dtype=np.float64))


def warp(rgb: np.ndarray, params: SimilarityParams, cfg) -> np.ndarray:
    return cv2.warpAffine(
        rgb, params.matrix(), (cfg.canvas_w, cfg.canvas_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )

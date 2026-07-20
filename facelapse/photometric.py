"""Photometric normalization so a decade of wildly different white balance
and exposure doesn't make the video strobe. Two modes:

  exposure   : match each frame's luminance mean/std to a reference frame.
  hist_match : full per-channel CDF histogram matching to a reference frame
               (stronger; equalizes color cast too).

The reference is the median-luminance frame of the selected set, chosen so we
normalize toward a typical frame rather than an extreme one.
"""
from __future__ import annotations

import cv2
import numpy as np


def choose_reference(frames: list[np.ndarray]) -> np.ndarray:
    lums = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).mean() for f in frames]
    idx = int(np.argsort(lums)[len(lums) // 2])
    return frames[idx]


def _match_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match src channel histogram to ref channel via CDF lookup."""
    src_hist = np.bincount(src.ravel(), minlength=256).astype(np.float64)
    ref_hist = np.bincount(ref.ravel(), minlength=256).astype(np.float64)
    src_cdf = np.cumsum(src_hist) / max(src_hist.sum(), 1)
    ref_cdf = np.cumsum(ref_hist) / max(ref_hist.sum(), 1)
    lut = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
    return lut[src]


def normalize(frame: np.ndarray, ref: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return frame
    if mode == "exposure":
        f = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        r = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
        for c in range(3):
            fs, rs = f[..., c].std() + 1e-6, r[..., c].std()
            f[..., c] = (f[..., c] - f[..., c].mean()) / fs * rs + r[..., c].mean()
        f = np.clip(f, 0, 255).astype(np.uint8)
        return cv2.cvtColor(f, cv2.COLOR_LAB2RGB)
    if mode == "hist_match":
        out = np.empty_like(frame)
        for c in range(3):
            out[..., c] = _match_channel(frame[..., c], ref[..., c])
        return out
    raise ValueError(f"unknown photometric mode: {mode}")

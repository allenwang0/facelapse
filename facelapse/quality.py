"""Sharpness and face-size quality signals, computed on the face crop from
the original-resolution image.
"""
from __future__ import annotations

import cv2
import numpy as np

from .records import Face


def interocular_px(f: Face) -> float:
    """Distance between the two eye keypoints in image pixels. A direct proxy
    for how much real facial detail exists. kps[0], kps[1] = eyes (VERIFY)."""
    return float(np.linalg.norm(f.kps[0] - f.kps[1]))


def sharpness(rgb: np.ndarray, bbox: np.ndarray) -> float:
    """Variance of the Laplacian over the face box. Higher = sharper. Blurry
    and motion-smeared faces (festival shots) score low and lose selection."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

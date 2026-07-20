"""Pose handling. Roll is NOT filtered here; it is corrected losslessly in
alignment. Yaw and pitch ARE filtered, because they cannot be corrected
without inventing occluded geometry, and we refuse to invent it.

Two functions:
  passes_pose: hard gate on |yaw|,|pitch|.
  frontality_score: continuous 'closeness to dead-frontal', fed into the
    per-bucket composite so selection MINIMIZES pose variance across the
    chosen sequence rather than merely staying under threshold. Sub-threshold
    frames that still range widely read as a swiveling head; this term pulls
    the selected set toward yaw=pitch=0.
"""
from __future__ import annotations

import math

from .records import Face


def passes_pose(f: Face, max_yaw: float, max_pitch: float) -> bool:
    if math.isnan(f.yaw) or math.isnan(f.pitch):
        return False   # unknown pose -> fail loud, don't silently pass
    return abs(f.yaw) <= max_yaw and abs(f.pitch) <= max_pitch


def frontality_score(f: Face, max_yaw: float, max_pitch: float) -> float:
    """1.0 at dead-frontal, decaying to ~0 at the tolerance edge."""
    if math.isnan(f.yaw) or math.isnan(f.pitch):
        return 0.0
    yn = min(1.0, abs(f.yaw) / max_yaw)
    pn = min(1.0, abs(f.pitch) / max_pitch)
    # combine; quadratic falloff punishes larger deviations harder
    return max(0.0, 1.0 - 0.5 * (yn ** 2 + pn ** 2))

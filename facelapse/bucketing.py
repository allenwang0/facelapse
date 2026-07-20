"""Turn irregular photo clumps into a regular cadence: bucket faces by time,
keep the single best per bucket. This also kills burst-shot near-duplicates
for free (they share a bucket, only the sharpest survives).

The composite score is where pose-variance minimization, neutrality,
sharpness, resolution, and identity confidence trade off. Weights are a
starting point; tune against the review page.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from .records import Face
from .pose import frontality_score


def bucket_key(dt: datetime, grain: str) -> str:
    if grain == "day":
        return dt.strftime("%Y-%m-%d")
    if grain == "month":
        return dt.strftime("%Y-%m")
    # default week: ISO year-week
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def composite_score(f: Face, cfg, sharp_ref: float) -> float:
    """Blend of the quality axes, each roughly normalized to [0,1].
    sharp_ref: a high-percentile sharpness across the set, used to scale the
    unbounded Laplacian variance into [0,1]."""
    front = frontality_score(f, cfg.max_abs_yaw_deg, cfg.max_abs_pitch_deg)
    res = min(1.0, (f.interocular_px or 0.0) / max(cfg.min_interocular_px * 3, 1.0))
    sharp = min(1.0, (f.sharpness or 0.0) / sharp_ref) if sharp_ref > 0 else 0.0
    neutral = 1.0 - min(1.0, (f.neutral_penalty or 0.0) / max(cfg.neutral_max_penalty, 1e-6))
    ident = float(np.clip((f.id_sim or 0.0), 0.0, 1.0))
    eyes = 1.0 if (f.ear is None or f.ear >= cfg.ear_closed_below) else 0.0

    return (
        0.30 * front +      # pose proximity -> minimizes swivel across sequence
        0.25 * neutral +    # neutrality
        0.20 * sharp +      # sharpness
        0.15 * res +        # face resolution
        0.05 * ident +      # identity confidence (already gated; light tiebreak)
        0.05 * eyes         # eyes open
    )


def select_best_per_bucket(
    faces: list[Face], ts_of: dict[str, datetime], cfg,
) -> list[Face]:
    """Given self-faces that already passed pose+neutral gates, assign buckets
    and return the winner per bucket (marks .selected, .bucket, .composite)."""
    if not faces:
        return []
    sharp_vals = [f.sharpness for f in faces if f.sharpness is not None]
    sharp_ref = float(np.percentile(sharp_vals, 90)) if sharp_vals else 1.0

    by_bucket: dict[str, list[Face]] = {}
    for f in faces:
        dt = ts_of.get(f.content_hash)
        if dt is None:
            continue   # untrustworthy/absent date -> excluded from final cut
        f.bucket = bucket_key(dt, cfg.bucket_grain)
        f.composite = composite_score(f, cfg, sharp_ref)
        by_bucket.setdefault(f.bucket, []).append(f)

    winners: list[Face] = []
    for bucket, group in by_bucket.items():
        group.sort(key=lambda x: x.composite or 0.0, reverse=True)
        best = group[0]
        best.selected = True
        winners.append(best)
    # chronological order for the video
    winners.sort(key=lambda f: ts_of[f.content_hash])
    return winners

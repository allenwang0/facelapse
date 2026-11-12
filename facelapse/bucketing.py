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

from .config import Config
from .pose import frontality_score
from .records import Face


def bucket_key(dt: datetime, grain: str) -> str:
    if grain == "day":
        return dt.strftime("%Y-%m-%d")
    if grain == "month":
        return dt.strftime("%Y-%m")
    # default week: ISO year-week
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def composite_score(f: Face, cfg: Config, sharp_ref: float) -> float:
    """Blend of the quality axes, each roughly normalized to [0,1].
    sharp_ref: a high-percentile sharpness across the set, used to scale the
    unbounded Laplacian variance into [0,1]."""
    front = frontality_score(f, cfg.max_abs_yaw_deg, cfg.max_abs_pitch_deg)
    res = min(
        1.0,
        (f.interocular_px or 0.0) / max(cfg.min_interocular_px * cfg.resolution_scale_factor, 1.0),
    )
    sharp = min(1.0, (f.sharpness or 0.0) / sharp_ref) if sharp_ref > 0 else 0.0
    neutral = 1.0 - min(1.0, (f.neutral_penalty or 0.0) / max(cfg.neutral_max_penalty, 1e-6))
    ident = float(np.clip((f.id_sim or 0.0), 0.0, 1.0))
    eyes = 1.0 if (f.ear is None or f.ear >= cfg.ear_closed_below) else 0.0

    return float(
        cfg.score_weight_frontality * front
        + cfg.score_weight_neutral * neutral
        + cfg.score_weight_sharpness * sharp
        + cfg.score_weight_resolution * res
        + cfg.score_weight_identity * ident
        + cfg.score_weight_eyes_open * eyes
    )


def select_best_per_bucket(
    faces: list[Face],
    ts_of: dict[str, datetime],
    cfg: Config,
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
            continue  # untrustworthy/absent date -> excluded from final cut
        f.bucket = bucket_key(dt, cfg.bucket_grain)
        f.composite = composite_score(f, cfg, sharp_ref)
        by_bucket.setdefault(f.bucket, []).append(f)

    winners: list[Face] = []
    for group in by_bucket.values():
        group.sort(key=lambda x: x.composite or 0.0, reverse=True)
        best = group[0]
        best.selected = True
        winners.append(best)
    # chronological order for the video
    winners.sort(key=lambda f: ts_of[f.content_hash])
    return winners


def year_quotas(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Water-fill a budget of n frames across years. Years with fewer frames
    than the current fair share keep all of them; the rest split what is left
    as evenly as possible. Quotas sum to min(n, total) and never exceed a
    year's size. Ported from face-lapse (commit 50de507), whose ceil-based
    share overshot: years of 47, 67 and 67 frames with n=180 got 181."""
    quota = dict.fromkeys(sizes, 0)
    if sum(sizes.values()) <= n:
        return dict(sizes)
    left, active = n, sorted(sizes)
    while active:
        share = left // len(active)
        capped = [y for y in active if sizes[y] <= share]
        if not capped:
            for y in active:
                quota[y] = share
            for y in active[: left - share * len(active)]:
                quota[y] += 1
            break
        for y in capped:
            quota[y] = sizes[y]
            left -= sizes[y]
            active.remove(y)
    return quota


def thin_by_year(winners: list[Face], ts_of: dict[str, datetime], n: int) -> list[Face]:
    """Keep at most n winners, spread across years by year_quotas and, within
    a year, picked closest to evenly spaced target dates."""
    by_year: dict[str, list[Face]] = {}
    for f in sorted(winners, key=lambda f: ts_of[f.content_hash]):
        by_year.setdefault(str(ts_of[f.content_hash].year), []).append(f)
    quota = year_quotas({y: len(v) for y, v in by_year.items()}, n)
    kept: list[Face] = []
    for year, items in by_year.items():
        q = quota[year]
        if q >= len(items):
            kept.extend(items)
            continue
        times = np.array([ts_of[f.content_hash].timestamp() for f in items])
        available = set(range(len(items)))
        for t in np.linspace(times[0], times[-1], q):
            j = min(available, key=lambda k: abs(times[k] - t))
            kept.append(items[j])
            available.remove(j)
    return sorted(kept, key=lambda f: ts_of[f.content_hash])

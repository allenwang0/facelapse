"""Tests for everything that runs WITHOUT the ML model deps. Encodes WHY
each behavior matters, not just what it does (per the project's test rule)."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelapse.config import Config
from facelapse.records import Face, Timestamp, TsSource, ImageRecord
from facelapse.cache import Cache
from facelapse import align as align_mod, smoothing, bucketing, dedup, photometric, timestamps
from facelapse.align import SimilarityParams


def _face(**kw) -> Face:
    base = dict(
        content_hash="h", face_id=0, bbox=np.array([0, 0, 100, 100], np.float32),
        kps=np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], np.float32),
        embedding=np.ones(512, np.float32) / np.sqrt(512),
        yaw=0.0, pitch=0.0, roll=0.0, det_score=0.9,
    )
    base.update(kw)
    return Face(**base)


def test_timestamp_source_priority_and_confidence():
    # WHY: the temporal axis of the video must degrade to low confidence, not
    # silently trust a filename or mtime as if it were EXIF.
    assert Timestamp(datetime(2020, 1, 1), TsSource.TAKEOUT_JSON).is_trustworthy
    assert Timestamp(datetime(2020, 1, 1), TsSource.FILENAME).confidence == "medium"
    assert not Timestamp(datetime(2020, 1, 1), TsSource.MTIME).is_trustworthy


def test_filename_date_parsing():
    # WHY: phone filename stamps are a real fallback when EXIF is stripped.
    dt = timestamps._from_filename(Path("IMG_20210517_143000.jpg"))
    assert dt == datetime(2021, 5, 17, 14, 30, 0)
    assert timestamps._from_filename(Path("PXL_20191225.jpg")) == datetime(2019, 12, 25)


def test_alignment_places_eyes_at_canonical_positions():
    # WHY: the entire "comparable over time" claim rests on eyes landing at
    # fixed canonical pixels every frame. This asserts the invariant.
    cfg = Config()
    f = _face(kps=np.array([[100, 200], [180, 210], [140, 240], [110, 280], [170, 285]], np.float32))
    p = align_mod.estimate(f, cfg)
    M = p.matrix()
    le = M @ np.array([f.kps[0][0], f.kps[0][1], 1.0])
    re = M @ np.array([f.kps[1][0], f.kps[1][1], 1.0])
    assert np.allclose(le, cfg.left_eye_xy, atol=1.0)
    assert np.allclose(re, cfg.right_eye_xy, atol=1.0)


def test_alignment_corrects_roll():
    # WHY: roll must be removed (eyes horizontal) regardless of input tilt.
    cfg = Config()
    # eyes tilted 30 degrees
    ang = np.deg2rad(30)
    c, s = np.cos(ang), np.sin(ang)
    le = np.array([100, 200]); off = np.array([80 * c, 80 * s])
    f = _face(kps=np.array([le, le + off, le + [40, 60], le + [10, 90], le + [60, 92]], np.float32))
    p = align_mod.estimate(f, cfg)
    M = p.matrix()
    le2 = M @ np.array([f.kps[0][0], f.kps[0][1], 1.0])
    re2 = M @ np.array([f.kps[1][0], f.kps[1][1], 1.0])
    assert abs(le2[1] - re2[1]) < 1.0   # eyes on the same horizontal line


def test_smoothing_reduces_jitter_preserves_drift():
    # WHY: smoothing must kill high-freq jitter without erasing slow real drift.
    n = 20
    drift = np.linspace(0, 10, n)
    jitter = np.array([(-1) ** i for i in range(n)]) * 2.0
    seq = [SimilarityParams(0.0, 1.0, float(d + j), 0.0) for d, j in zip(drift, jitter)]
    out = smoothing.smooth_params(seq, window=5)
    tx = np.array([p.tx for p in out])
    # jitter variance drops
    assert np.var(np.diff(tx)) < np.var(np.diff([p.tx for p in seq]))
    # endpoints still reflect the drift direction
    assert tx[-1] > tx[0] + 5


def test_bucket_key_grains():
    dt = datetime(2021, 5, 17)
    assert bucketing.bucket_key(dt, "day") == "2021-05-17"
    assert bucketing.bucket_key(dt, "month") == "2021-05"
    assert bucketing.bucket_key(dt, "week").startswith("2021-W")


def test_bucket_selection_picks_highest_composite():
    # WHY: within a bucket the sharpest/most-frontal/most-neutral must win.
    cfg = Config()
    good = _face(content_hash="g", sharpness=500, interocular_px=120,
                 neutral_penalty=0.0, id_sim=0.6, ear=0.3, yaw=0, pitch=0)
    bad = _face(content_hash="b", sharpness=10, interocular_px=45,
                neutral_penalty=0.9, id_sim=0.4, ear=0.3, yaw=18, pitch=18)
    ts = {"g": datetime(2021, 5, 17), "b": datetime(2021, 5, 18)}  # same ISO week
    winners = bucketing.select_best_per_bucket([good, bad], ts, cfg)
    assert len(winners) == 1 and winners[0].content_hash == "g"


def test_dedup_groups_near_duplicates():
    # WHY: a re-saved copy must not occupy a second bucket slot.
    base = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    noisy = base.copy(); noisy[0, 0] = 255 - noisy[0, 0]
    hashes = {"a": dedup.phash(base), "b": dedup.phash(noisy)}
    canon = dedup.group_duplicates(hashes, max_hamming=6)
    assert canon["a"] == canon["b"]


def test_photometric_exposure_matches_luminance():
    # WHY: exposure normalization must pull a dark frame toward the reference.
    dark = np.full((32, 32, 3), 40, np.uint8)
    ref = np.full((32, 32, 3), 160, np.uint8)
    out = photometric.normalize(dark, ref, "exposure")
    assert out.mean() > dark.mean() + 50


def test_cache_roundtrip_embeddings():
    # WHY: embeddings survive the BLOB round-trip bit-exact, or identity breaks.
    import tempfile, os
    tmp = tempfile.mkdtemp()
    cache = Cache(os.path.join(tmp, "t.db"))
    f = _face(embedding=np.random.default_rng(1).random(512).astype(np.float32))
    rec = ImageRecord("h", "/x.jpg", 100, 100, Timestamp(datetime(2021, 1, 1), TsSource.EXIF_ORIGINAL), faces=[f])
    cache.upsert_image(rec)
    got = cache.all_faces()[0]
    assert np.allclose(got.embedding, f.embedding, atol=1e-6)
    cache.close()

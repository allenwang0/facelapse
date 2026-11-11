"""score and select are advertised as cheap and rerunnable while tuning. That
only holds if a rerun with stricter settings clears the flags the previous run
set; otherwise render keeps rendering the old winners."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from facelapse import pipeline
from facelapse.cache import Cache
from facelapse.config import Config
from facelapse.records import ImageRecord, Timestamp, TsSource
from tests.helpers import embedding_with_similarity, make_face, unit, write_image

ME = unit(np.random.default_rng(42).normal(size=512))

# (similarity to the seed, capture date)
PHOTOS = [
    (0.95, datetime(2020, 1, 5)),
    (0.30, datetime(2020, 1, 20)),
    (0.50, datetime(2020, 2, 10)),
    (0.92, datetime(2020, 3, 3)),
    (0.40, datetime(2020, 3, 25)),
]


def _build(tmp_path) -> Config:
    cfg = Config()
    cfg.cache_db = str(tmp_path / "cache.db")
    cfg.out_dir = str(tmp_path / "out")
    cfg.bucket_grain = "month"
    cache = Cache(cfg.cache_db)
    cache.add_seed("seed.jpg", datetime(2020, 2, 1), ME)
    for i, (sim, when) in enumerate(PHOTOS):
        path = tmp_path / f"p{i}.png"
        write_image(path, seed=i)
        face = make_face(
            content_hash=f"h{i}",
            embedding=embedding_with_similarity(ME, sim, seed=i),
            interocular_px=40.0,
            sharpness=100.0,
        )
        ts = Timestamp(when, TsSource.EXIF_ORIGINAL)
        cache.upsert_image(ImageRecord(f"h{i}", str(path), 160, 160, ts, faces=[face]))
    cache.close()
    return cfg


def test_stricter_rerun_drops_previous_winners(tmp_path):
    cfg = _build(tmp_path)
    cfg.id_t_abs = 0.2
    pipeline.score(cfg)
    pipeline.select(cfg)

    cfg.id_t_abs = 0.9
    pipeline.score(cfg)
    pipeline.select(cfg)

    cache = Cache(cfg.cache_db)
    selected = cache.all_faces(only_selected=True)
    self_faces = cache.all_faces(only_self=True)
    cache.close()
    assert selected, "the two faces above 0.9 must still be selected"
    assert all(f.id_sim is not None and f.id_sim >= 0.9 for f in selected)
    assert all(f.id_sim is not None and f.id_sim >= 0.9 for f in self_faces)
    buckets = [f.bucket for f in selected]
    assert len(buckets) == len(set(buckets))


def test_coarser_grain_rerun_keeps_one_winner_per_bucket(tmp_path):
    cfg = _build(tmp_path)
    cfg.id_t_abs = 0.2
    cfg.bucket_grain = "day"
    pipeline.score(cfg)
    pipeline.select(cfg)
    cfg.bucket_grain = "month"
    pipeline.select(cfg)

    cache = Cache(cfg.cache_db)
    selected = cache.all_faces(only_selected=True)
    cache.close()
    buckets = [f.bucket for f in selected]
    assert len(buckets) == len(set(buckets)) == 3

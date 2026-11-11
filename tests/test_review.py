"""The review page is where ambiguous two-person photos get resolved by hand.
A promotion must turn the chosen face into the subject and keep doing so on
every later score run, or tuning silently undoes the user's decisions."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from facelapse import cli, pipeline
from facelapse.cache import Cache
from facelapse.config import Config
from facelapse.records import ImageRecord, Timestamp, TsSource
from tests.helpers import embedding_with_similarity, make_face, unit, write_image

ME = unit(np.random.default_rng(5).normal(size=512))


def _ambiguous_cache(tmp_path, size=(160, 160)) -> Config:
    cfg = Config()
    cfg.cache_db = str(tmp_path / "cache.db")
    cfg.out_dir = str(tmp_path / "out")
    cache = Cache(cfg.cache_db)
    cache.add_seed("seed.jpg", None, ME)
    path = tmp_path / "pair.jpg"
    write_image(path, seed=9, size=size, exif_date=datetime(2021, 6, 1))
    me = make_face(content_hash="pair", face_id=0, embedding=embedding_with_similarity(ME, 0.62, 1))
    sib = make_face(
        content_hash="pair",
        face_id=1,
        bbox=np.array([100, 10, 150, 80], np.float32),
        embedding=embedding_with_similarity(ME, 0.60, 2),
    )
    ts = Timestamp(datetime(2021, 6, 1), TsSource.EXIF_ORIGINAL)
    cache.upsert_image(ImageRecord("pair", str(path), size[0], size[1], ts, faces=[me, sib]))
    cache.close()
    return cfg


def _self_faces(cfg):
    cache = Cache(cfg.cache_db)
    try:
        return [(f.content_hash, f.face_id) for f in cache.all_faces(only_self=True)]
    finally:
        cache.close()


def test_ambiguous_image_is_recorded_not_dropped(tmp_path):
    cfg = _ambiguous_cache(tmp_path)
    pipeline.score(cfg)
    cache = Cache(cfg.cache_db)
    assert cache.hashes_with_outcome("ambiguous") == ["pair"]
    cache.close()
    assert _self_faces(cfg) == []


def test_promotion_survives_score_reruns(tmp_path):
    cfg = _ambiguous_cache(tmp_path)
    pipeline.score(cfg)
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({"rejects": [], "promotions": [{"content_hash": "pair", "face_id": 1}]})
    )
    pipeline.apply_review(cfg, str(review))
    pipeline.score(cfg)
    assert _self_faces(cfg) == [("pair", 1)]
    cfg.id_margin = 0.5
    pipeline.score(cfg)
    assert _self_faces(cfg) == [("pair", 1)]


def test_review_thumbnails_are_small(tmp_path, monkeypatch):
    # WHY: holding every winner at full resolution is ~10 GB for 300 12 MP
    # photos; the page only needs 220 px thumbnails.
    cfg = _ambiguous_cache(tmp_path, size=(3000, 2000))
    pipeline.score(cfg)
    captured = {}

    def fake_write_review(selected, ambiguous, thumbs, dates, out_dir):
        captured.update(thumbs)
        return "review.html"

    monkeypatch.setattr("facelapse.review.write_review", fake_write_review)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"cache_db": cfg.cache_db, "out_dir": cfg.out_dir}))
    cli.main(["--config", str(cfg_path), "review"])
    assert captured, "the ambiguous image should be on the page"
    assert all(t.rgb.shape[1] <= 220 for t in captured.values())


def test_review_page_has_promotion_buttons_and_download(tmp_path):
    cfg = _ambiguous_cache(tmp_path)
    pipeline.score(cfg)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"cache_db": cfg.cache_db, "out_dir": cfg.out_dir}))
    cli.main(["--config", str(cfg_path), "review"])
    page = (tmp_path / "out" / "review.html").read_text()
    assert page.count('class="me"') == 2
    assert "review.json" in page


def test_render_rejects_accepts_review_download(tmp_path):
    p = tmp_path / "review.json"
    p.write_text(json.dumps({"rejects": ["a", "b"], "promotions": []}))
    assert cli._load_rejects(str(p)) == {"a", "b"}
    p.write_text(json.dumps(["c"]))
    assert cli._load_rejects(str(p)) == {"c"}

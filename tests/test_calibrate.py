"""calibrate must return a margin with a defined error rate: among labeled
multi-face images where the top face is not you, at most target_far of them
may clear the margin."""

from __future__ import annotations

import json

import numpy as np
import pytest

from facelapse import cli
from facelapse.cache import Cache
from facelapse.identity import calibrate
from facelapse.records import ImageRecord, Timestamp, TsSource
from tests.helpers import make_face


def test_margin_meets_target_false_accept_rate_exactly():
    # 100 images where a lookalike outranks you, margins 0.01..1.00, and 50
    # where you are on top with margin 0.995. Allowing 1% false accepts lets
    # exactly one negative through, so the margin sits just above 0.99.
    neg = [(0.5 + m, 0.5, False) for m in np.arange(1, 101) / 100]
    pos = [(0.9, 0.9 - 0.995, True)] * 50
    out = calibrate(neg + pos, target_far=0.01)
    assert out["margin"] == pytest.approx(0.99, abs=1e-6)
    assert out["margin"] > 0.99
    assert out["false_accept_rate"] == pytest.approx(0.01)
    assert out["true_accept_rate"] == pytest.approx(1.0)


def test_stricter_target_never_lowers_the_margin():
    rng = np.random.default_rng(0)
    triples = [(0.6, 0.6 - g, bool(g > 0.1)) for g in rng.uniform(0, 0.3, 400)]
    margins = [calibrate(triples, target_far=t)["margin"] for t in (0.2, 0.05, 0.01, 0.0)]
    assert margins == sorted(margins)


def test_true_accept_rate_counts_positives_at_or_above_margin():
    triples = [(0.8, 0.6, True), (0.8, 0.5, True), (0.7, 0.6, False)]
    out = calibrate(triples, target_far=0.0)
    assert 0.09 < out["margin"] < 0.2
    assert out["true_accept_rate"] == pytest.approx(1.0)


def test_calibrate_command_reads_labels_and_cache(tmp_path, capsys):
    db = tmp_path / "cache.db"
    cache = Cache(str(db))
    for i, (me_sim, other_sim) in enumerate([(0.8, 0.4), (0.5, 0.7), (0.9, 0.3)]):
        faces = [
            make_face(content_hash=f"h{i}", face_id=0, id_sim=me_sim),
            make_face(content_hash=f"h{i}", face_id=1, id_sim=other_sim),
        ]
        ts = Timestamp(None, TsSource.NONE)
        cache.upsert_image(ImageRecord(f"h{i}", f"p{i}.jpg", 10, 10, ts, faces=faces))
    cache.close()
    labels = tmp_path / "labels.csv"
    labels.write_text("content_hash,me_face_id\nh0,0\nh1,0\nh2,0\n")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"cache_db": str(db)}))
    cli.main(["--config", str(cfg), "calibrate", str(labels), "--target-far", "0"])
    out = capsys.readouterr().out
    assert '"id_margin"' in out
    assert "true accept rate 1.00" in out


def test_labels_may_name_images_by_file_name(tmp_path, capsys):
    db = tmp_path / "cache.db"
    cache = Cache(str(db))
    faces = [
        make_face(content_hash="h0", face_id=0, id_sim=0.8),
        make_face(content_hash="h0", face_id=1, id_sim=0.4),
    ]
    ts = Timestamp(None, TsSource.NONE)
    cache.upsert_image(ImageRecord("h0", "/archive/2019/pair.jpg", 10, 10, ts, faces=faces))
    cache.close()
    labels = tmp_path / "labels.csv"
    labels.write_text("path,me_face_id\npair.jpg,0\n")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"cache_db": str(db)}))
    assert cli.main(["--config", str(cfg), "calibrate", str(labels)]) == 0
    assert "1 images" in capsys.readouterr().out

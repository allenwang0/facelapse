"""Apple Vision backend: conversion from the helper's output, and the
single-person rule it depends on. The compile test runs only on macOS."""

from __future__ import annotations

import math
import sys
from datetime import datetime

import numpy as np
import pytest

from facelapse import pipeline
from facelapse.cache import Cache
from facelapse.config import Config
from facelapse.records import ImageRecord, Timestamp, TsSource
from facelapse.vision import faces_from_vision
from tests.helpers import make_face

RECORD = {
    "faces": [
        {
            "bbox": [100, 50, 300, 300],
            "confidence": 0.98,
            # Vision's "left" is the subject's left, which is image right
            "left_pupil": [[250, 150]],
            "right_pupil": [[150, 152]],
            "nose": [[195, 200], [205, 210], [200, 220]],
            "outer_lips": [[160, 260], [200, 250], [240, 262], [200, 270]],
            "yaw": math.radians(-12.0),
            "pitch": math.radians(4.0),
            "roll": math.radians(3.0),
        }
    ]
}


def test_conversion_orders_eyes_and_uses_degrees():
    (f,) = faces_from_vision(RECORD, "h")
    assert f.kps[0].tolist() == [150, 152] and f.kps[1].tolist() == [250, 150]
    assert np.allclose(f.kps[2], [200, 210])
    assert f.kps[3].tolist() == [160, 260] and f.kps[4].tolist() == [240, 262]
    assert (round(f.yaw, 6), round(f.pitch, 6), round(f.roll, 6)) == (-12.0, 4.0, 3.0)
    assert f.embedding.size == 0


def test_missing_pose_is_nan_so_the_pose_gate_rejects_it():
    rec = {"faces": [{**RECORD["faces"][0]}]}
    del rec["faces"][0]["yaw"]
    (f,) = faces_from_vision(rec, "h")
    assert math.isnan(f.yaw)


def _vision_cache(tmp_path) -> Config:
    cfg = Config()
    cfg.cache_db = str(tmp_path / "c.db")
    small = make_face(content_hash="i", face_id=0, embedding=np.zeros(0, np.float32))
    big = make_face(
        content_hash="i",
        face_id=1,
        bbox=np.array([0, 0, 300, 300], np.float32),
        embedding=np.zeros(0, np.float32),
    )
    cache = Cache(cfg.cache_db)
    ts = Timestamp(datetime(2020, 1, 1), TsSource.EXIF_ORIGINAL)
    cache.upsert_image(ImageRecord("i", "x.jpg", 400, 400, ts, faces=[small, big]))
    cache.close()
    return cfg


def test_faces_without_embeddings_require_single_person(tmp_path):
    cfg = _vision_cache(tmp_path)
    with pytest.raises(RuntimeError, match="--single-person"):
        pipeline.score(cfg)


def test_single_person_takes_the_largest_face(tmp_path):
    cfg = _vision_cache(tmp_path)
    pipeline.score(cfg, single_person=True)
    cache = Cache(cfg.cache_db)
    assert [f.face_id for f in cache.all_faces(only_self=True)] == [1]
    cache.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision is macOS only")
def test_helper_compiles_and_answers(tmp_path):
    from facelapse.vision import VisionDetector

    det = VisionDetector(tmp_path / "vision_detect")
    assert det.analyze(np.full((64, 64, 3), 128, np.uint8), "h") == []

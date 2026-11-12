"""analyze walks archives of tens of thousands of files; one bad file must be
recorded and skipped, never abort the pass."""

from __future__ import annotations

from datetime import datetime

from facelapse import pipeline
from facelapse.cache import Cache
from facelapse.config import Config
from tests.helpers import make_face, write_image


class OneFaceDetector:
    def analyze(self, rgb, content_hash):
        return [make_face(content_hash=content_hash)]


def test_oversized_file_is_recorded_and_the_next_image_analyzed(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    with open(photos / "a_huge.jpg", "wb") as fh:
        fh.truncate(201 * 1024 * 1024)  # sparse: no real disk use
    write_image(photos / "b_normal.jpg", seed=1, exif_date=datetime(2020, 1, 1))
    cfg = Config()
    cfg.cache_db = str(tmp_path / "cache.db")

    pipeline.analyze(str(photos), cfg, detector=OneFaceDetector(), mesh=None)

    cache = Cache(cfg.cache_db)
    rows = cache.db.execute("SELECT path, decoded_ok, error FROM images ORDER BY path").fetchall()
    n_faces = len(cache.all_faces())
    cache.close()
    assert [r[1] for r in rows] == [0, 1]
    assert rows[0][0].endswith("a_huge.jpg") and "too large" in rows[0][2]
    assert n_faces == 1


def test_select_reads_cached_phash_without_decoding(tmp_path, monkeypatch):
    # WHY: select is the cheap, rerun-while-tuning stage; decoding every
    # gated photo to hash it made it as slow as analyze.
    import numpy as np

    from facelapse import ingest
    from tests.helpers import unit

    me = unit(np.ones(512))
    photos = tmp_path / "photos"
    photos.mkdir()
    for i in range(3):
        write_image(photos / f"p{i}.jpg", seed=i, exif_date=datetime(2020, 1 + i, 1))

    class MeDetector:
        def analyze(self, rgb, content_hash):
            return [make_face(content_hash=content_hash, embedding=me)]

    cfg = Config()
    cfg.cache_db = str(tmp_path / "cache.db")
    cache = Cache(cfg.cache_db)
    cache.add_seed("s", None, me)
    cache.close()
    pipeline.analyze(str(photos), cfg, detector=MeDetector(), mesh=None)
    pipeline.score(cfg)

    def no_decode(path):
        raise AssertionError(f"select decoded {path}")

    monkeypatch.setattr(ingest, "decode", no_decode)
    assert len(pipeline.select(cfg)) == 3

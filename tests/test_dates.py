"""Date handling. The timeline is only as honest as its dates: a photo dated
by file mtime (screenshots, re-saves, copies) must not enter the video unless
the user opts in, and every trustworthy source must resolve to the day the
camera recorded."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from facelapse import pipeline, timestamps
from facelapse.cache import Cache
from facelapse.config import Config
from facelapse.records import TsSource
from tests.helpers import make_face, unit, write_image

ME = unit(np.random.default_rng(3).normal(size=512))


class OneFaceDetector:
    def analyze(self, rgb, content_hash):
        return [make_face(content_hash=content_hash, embedding=ME, sharpness=50.0)]


def _analyzed(tmp_path) -> Config:
    photos = tmp_path / "photos"
    photos.mkdir()
    write_image(photos / "camera.jpg", seed=1, exif_date=datetime(2020, 5, 1, 9, 0))
    write_image(photos / "screenshot.png", seed=2)
    cfg = Config()
    cfg.cache_db = str(tmp_path / "cache.db")
    cfg.out_dir = str(tmp_path / "out")
    cfg.bucket_grain = "day"
    cache = Cache(cfg.cache_db)
    cache.add_seed("seed.jpg", None, ME)
    cache.close()
    pipeline.analyze(str(photos), cfg, detector=OneFaceDetector(), mesh=None)
    pipeline.score(cfg)
    return cfg


def test_mtime_dated_photo_is_excluded_by_default(tmp_path):
    cfg = _analyzed(tmp_path)
    winners = pipeline.select(cfg)
    cache = Cache(cfg.cache_db)
    sources = [cache.image_ts(f.content_hash).source for f in winners]
    cache.close()
    assert sources == [TsSource.EXIF_ORIGINAL]


def test_allow_mtime_dates_keeps_it(tmp_path):
    cfg = _analyzed(tmp_path)
    winners = pipeline.select(cfg, allow_mtime_dates=True)
    assert len(winners) == 2


def test_exif_original_is_read_from_the_exif_sub_ifd(tmp_path):
    # Cameras write DateTimeOriginal in the Exif sub-IFD (0x8769), not IFD0.
    p = tmp_path / "camera.jpg"
    write_image(p, seed=1, exif_date=datetime(2019, 7, 4, 18, 30))
    ts = timestamps.resolve(p)
    assert ts.source == TsSource.EXIF_ORIGINAL
    assert ts.value == datetime(2019, 7, 4, 18, 30)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("IMG_20210517_143000.jpg", datetime(2021, 5, 17, 14, 30, 0)),
        ("PXL_20191225.jpg", datetime(2019, 12, 25)),
        ("2021-05-17_14-30.jpg", datetime(2021, 5, 17, 14, 30)),
        ("scan_20210517.jpg", datetime(2021, 5, 17)),
        ("scan_20211340.jpg", None),
        ("IMG_20211340_143000.jpg", None),
    ],
)
def test_filename_patterns(name, expected):
    assert timestamps._from_filename(Path(name)) == expected

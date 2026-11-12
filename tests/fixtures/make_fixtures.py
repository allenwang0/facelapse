"""Write the synthetic demo archive: faces drawn with OpenCV, no real photos.

    python tests/fixtures/make_fixtures.py [OUT_DIR]

OUT_DIR defaults to tests/fixtures and receives photos/ and seeds/. Both are
generated, never committed. Every case the pipeline filters is present, so
each funnel stage removes something: other people, a lookalike twin
(ambiguous), turned heads, smiles, closed eyes, tiny faces, a re-saved
duplicate, and screenshots with no date but mtime. One stripped photo is
dated by a Takeout sidecar and one by its filename. The subject's hair
greys over the years so the timelapse shows change. The year 2019 is
deliberately missing to show a gap in the coverage chart.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from facelapse.synthetic import FaceSpec, draw_face  # noqa: E402

W, H = 800, 600
YEARS = [2016, 2017, 2018, 2020, 2021, 2022, 2023, 2024]


def _background(rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    a, b = rng.uniform(60, 160), rng.uniform(-40, 40)
    g = a + b * (xx / W) + 15 * np.sin(yy / rng.uniform(40, 90))
    img = np.repeat(g[..., None], 3, axis=2).clip(0, 255).astype(np.uint8)
    # grey furniture so distinct photos have distinct perceptual hashes
    for _ in range(6):
        x, y = int(rng.integers(0, W - 100)), int(rng.integers(0, H - 100))
        w, h = int(rng.integers(60, 300)), int(rng.integers(60, 300))
        img[y : y + h, x : x + w] = int(rng.integers(20, 180))
    return img


def _hair(img: np.ndarray, s: FaceSpec, grey: float) -> None:
    import cv2

    v = int(40 + 150 * grey)
    center = (round(s.cx), round(s.cy - 0.55 * s.iod))
    axes = (round(1.2 * s.iod), round(0.75 * s.iod))
    cv2.ellipse(img, center, axes, s.roll, 180, 360, (v, v, v), -1)


def _photo(rng: np.random.Generator, when: datetime, **kw: object) -> np.ndarray:
    img = _background(rng)
    grey = (when.year - YEARS[0]) / (YEARS[-1] - YEARS[0])
    spec = FaceSpec(
        cx=float(kw.pop("cx", rng.uniform(300, 500))),
        cy=float(kw.pop("cy", rng.uniform(230, 280))),
        iod=float(kw.pop("iod", rng.uniform(70, 100))),
        yaw=float(kw.pop("yaw", rng.uniform(-6, 6))),
        pitch=float(kw.pop("pitch", rng.uniform(-5, 5))),
        roll=float(kw.pop("roll", rng.uniform(-8, 8))),
        **kw,  # type: ignore[arg-type]
    )
    _hair(img, spec, grey)
    draw_face(img, spec)
    return img


def _companion(img: np.ndarray, identity: str, rng: np.random.Generator) -> None:
    spec = FaceSpec(cx=float(rng.uniform(640, 680)), cy=260.0, iod=60.0, identity=identity)
    draw_face(img, spec)


def _exif(when: datetime) -> Image.Exif:
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")
    return exif


def _save_jpeg(path: Path, img: np.ndarray, when: datetime | None, quality: int = 90) -> None:
    kw = {"exif": _exif(when)} if when else {}
    Image.fromarray(img).save(path, quality=quality, **kw)


def make(out: Path) -> None:
    photos, seeds = out / "photos", out / "seeds"
    for d in (photos, seeds):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()
    rng = np.random.default_rng(2026)

    n = 0
    for year in YEARS:
        for month in (2, 5, 8, 11):
            when = datetime(year, month, int(rng.integers(1, 28)), int(rng.integers(8, 20)))
            n += 1
            _save_jpeg(photos / f"photo_{n:03d}.jpg", _photo(rng, when), when)
            # a second, weaker shot in the same month competes for the bucket
            if month == 5:
                n += 1
                later = when + timedelta(days=3)
                img = _photo(rng, later, yaw=float(rng.uniform(10, 15)))
                _save_jpeg(photos / f"photo_{n:03d}.jpg", img, later)

    cases = [
        ("turned", dict(yaw=40.0)),
        ("turned", dict(yaw=-38.0)),
        ("looking_down", dict(pitch=35.0)),
        ("smile", dict(smile=0.75)),
        ("smile", dict(smile=0.8)),
        ("blink", dict(eye_open=0.1, smile=0.15)),
        ("tiny", dict(iod=30.0)),
    ]
    for i, (name, kw) in enumerate(cases):
        when = datetime(YEARS[i % len(YEARS)], 6, 15, 12)
        _save_jpeg(photos / f"{name}_{i}.jpg", _photo(rng, when, **kw), when)

    for i in range(3):
        when = datetime(2017 + 2 * i, 9, 9, 15)
        img = _photo(rng, when, cx=330.0)
        _companion(img, "twin", rng)
        _save_jpeg(photos / f"with_twin_{i}.jpg", img, when)
    for i in range(2):
        when = datetime(2018 + 3 * i, 3, 3, 10)
        img = _photo(rng, when, cx=330.0)
        _companion(img, "stranger", rng)
        _save_jpeg(photos / f"with_friend_{i}.jpg", img, when)
    for i in range(3):
        when = datetime(2016 + 3 * i, 7, 7, 9)
        _save_jpeg(photos / f"friend_only_{i}.jpg", _photo(rng, when, identity="stranger"), when)

    # a re-saved copy: different bytes, same picture, later date
    dup_src = photos / "photo_001.jpg"
    Image.open(dup_src).save(
        photos / "photo_001_edited.jpg", quality=70, exif=_exif(datetime(2016, 2, 20))
    )

    # screenshots: PNG, no EXIF, no date in the name -> dated by mtime only
    for i in range(2):
        path = photos / f"screenshot_{i}.png"
        Image.fromarray(_photo(rng, datetime(2022, 1, 1))).save(path)
        stamp = datetime(2025, 1, 10 + i).timestamp()
        os.utime(path, (stamp, stamp))

    # stripped of EXIF but dated by a Takeout sidecar
    takeout_when = datetime(2021, 12, 24, 18, 0, tzinfo=timezone.utc)
    path = photos / "takeout_stripped.png"
    Image.fromarray(_photo(rng, takeout_when.replace(tzinfo=None))).save(path)
    sidecar = {"photoTakenTime": {"timestamp": str(int(takeout_when.timestamp()))}}
    (photos / "takeout_stripped.png.json").write_text(json.dumps(sidecar))

    # stripped of EXIF but dated by its filename
    img = _photo(rng, datetime(2023, 4, 2))
    Image.fromarray(img).save(photos / "IMG_20230402_101500.png")

    for year in (2016, 2020, 2024):
        when = datetime(year, 1, 15, 12)
        img = _photo(rng, when, yaw=0.0, pitch=0.0, roll=0.0, cx=400.0)
        _save_jpeg(seeds / f"seed_{year}.jpg", img, when)


if __name__ == "__main__":
    make(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)

"""Google Takeout exports arrive as several 2 GB zips. analyze must read
them in place, pair each photo with its sidecar JSON even when the sidecar
landed in a different zip, and leave nothing extracted on disk."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from facelapse import cli
from facelapse.cache import Cache
from facelapse.records import TsSource
from facelapse.synthetic import FaceSpec, draw_face

WHEN = datetime(2021, 7, 4, 15, 30, tzinfo=timezone.utc)


def _png(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = np.repeat(rng.integers(20, 180, (6, 8, 1), dtype=np.uint8), 3, axis=2)
    img = np.ascontiguousarray(np.kron(img, np.ones((80, 80, 1), np.uint8)))
    draw_face(img, FaceSpec(320, 220, 90))
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def test_split_takeout_zips_are_analyzed_in_place(tmp_path):
    photos = {f"Takeout/Google Photos/2021/IMG_{i}.png": _png(i) for i in range(3)}
    sidecar = {"photoTakenTime": {"timestamp": str(int(WHEN.timestamp()))}}
    with zipfile.ZipFile(tmp_path / "takeout-001.zip", "w") as z1:
        for name, data in photos.items():
            z1.writestr(name, data)
    with zipfile.ZipFile(tmp_path / "takeout-002.zip", "w") as z2:
        for name in photos:
            z2.writestr(name + ".json", json.dumps(sidecar))
    before = sorted(p.name for p in tmp_path.iterdir())

    cfg = tmp_path / "cfg.json"
    db = tmp_path / "cache.db"
    cfg.write_text(json.dumps({"cache_db": str(db), "out_dir": str(tmp_path / "out")}))
    zips = [str(tmp_path / "takeout-001.zip"), str(tmp_path / "takeout-002.zip")]
    cli.main(["--config", str(cfg), "analyze", *zips, "--backend", "synthetic"])

    cache = Cache(str(db))
    rows = cache.db.execute("SELECT content_hash, path FROM images ORDER BY path").fetchall()
    assert len(rows) == 3
    for (ch, ref), (name, data) in zip(rows, sorted(photos.items())):
        assert ref == f"{zips[0]}::{name}"
        assert ch == hashlib.sha256(data).hexdigest()
        ts = cache.image_ts(ch)
        assert ts.source == TsSource.TAKEOUT_JSON
        assert ts.value == WHEN.replace(tzinfo=None)
    assert len(cache.all_faces()) == 3
    cache.close()

    after = {p.name for p in tmp_path.iterdir()} - {
        "cache.db",
        "cache.db-wal",
        "cache.db-shm",
        "cfg.json",
    }
    assert sorted(after) == before


def test_zip_members_feed_select_and_render(tmp_path):
    seed_img = tmp_path / "seed.png"
    seed_img.write_bytes(_png(9))
    with zipfile.ZipFile(tmp_path / "takeout.zip", "w") as z:
        for i in range(2):
            z.writestr(f"IMG_20200{1 + 5 * i}01_120000.png", _png(i))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"cache_db": str(tmp_path / "c.db"), "out_dir": str(tmp_path / "out")})
    )
    c = ["--config", str(cfg)]
    cli.main([*c, "seed", "add", str(seed_img), "--backend", "synthetic"])
    cli.main([*c, "run", str(tmp_path / "takeout.zip"), "--backend", "synthetic", "--no-au"])
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["n_frames"] == 2

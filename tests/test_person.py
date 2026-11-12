"""--person: one analyzed archive, one timelapse per person. Scoring and
selecting for one person must not disturb another's results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from facelapse import cli
from facelapse.cache import Cache
from facelapse.synthetic import FaceSpec, draw_face
from tests.fixtures.make_fixtures import make


def _frames(root: Path, person: str) -> list[str]:
    manifest = json.loads((root / "out" / person / "manifest.json").read_text())
    return [Path(r["path"]).name for r in manifest["frames"]]


def test_two_people_from_one_cache(tmp_path):
    make(tmp_path)
    ben_seed = tmp_path / "ben_seed.png"
    img = np.full((600, 800, 3), 110, np.uint8)
    draw_face(img, FaceSpec(400, 260, 90, identity="stranger"))
    Image.fromarray(img).save(ben_seed)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "cache_db": str(tmp_path / "c.db"),
                "out_dir": str(tmp_path / "out"),
                "bucket_grain": "month",
            }
        )
    )
    c = ["--config", str(cfg)]
    cli.main([*c, "analyze", str(tmp_path / "photos"), "--backend", "synthetic"])
    seeds = sorted(str(p) for p in (tmp_path / "seeds").glob("*.jpg"))
    cli.main([*c, "--person", "ana", "seed", "add", *seeds, "--backend", "synthetic"])
    cli.main([*c, "--person", "ben", "seed", "add", str(ben_seed), "--backend", "synthetic"])

    for person in ("ana", "ben"):
        for stage in (["score", "--no-au"], ["select"], ["render"]):
            cli.main([*c, "--person", person, *stage])
    ana_first = _frames(tmp_path, "ana")
    cli.main([*c, "--person", "ana", "render"])

    assert _frames(tmp_path, "ana") == ana_first, "ben's run changed ana's selection"
    assert len(ana_first) > 20 and not [n for n in ana_first if n.startswith("friend_only")]
    ben = _frames(tmp_path, "ben")
    assert ben and all(n.startswith(("friend_only", "with_friend")) for n in ben)

    cache = Cache(str(tmp_path / "c.db"), "ana")
    assert len(cache.seeds()) == 3
    assert cache.db.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 60
    cache.close()
    assert len(Cache(str(tmp_path / "c.db"), "ben").seeds()) == 1

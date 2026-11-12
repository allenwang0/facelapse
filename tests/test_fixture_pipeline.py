"""The full CLI on the synthetic archive: every filter must remove the case
it exists for, and nothing it should keep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from facelapse import cli
from tests.fixtures.make_fixtures import make

FILTERED = ("turned", "looking_down", "smile", "blink", "tiny", "friend_only", "screenshot")


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    make(root)
    cfg = root / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "cache_db": str(root / "cache.db"),
                "out_dir": str(root / "out"),
                "bucket_grain": "month",
            }
        )
    )
    seeds = sorted(str(p) for p in (root / "seeds").glob("*.jpg"))
    cli.main(["--config", str(cfg), "seed", "add", *seeds, "--backend", "synthetic"])
    cli.main(
        ["--config", str(cfg), "run", str(root / "photos"), "--backend", "synthetic", "--no-au"]
    )
    return root


def test_filtered_cases_never_reach_the_video(demo_run: Path):
    manifest = json.loads((demo_run / "out" / "manifest.json").read_text())
    names = [Path(r["path"]).name for r in manifest["frames"]]
    assert len(names) >= 20
    assert not [n for n in names if n.startswith(FILTERED)]
    assert "photo_001_edited.jpg" not in names or "photo_001.jpg" not in names


def test_one_frame_per_month_in_date_order(demo_run: Path):
    frames = json.loads((demo_run / "out" / "manifest.json").read_text())["frames"]
    buckets = [r["bucket"] for r in frames]
    assert len(buckets) == len(set(buckets))
    assert [r["date"] for r in frames] == sorted(r["date"] for r in frames)

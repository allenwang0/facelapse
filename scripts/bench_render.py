"""Render benchmark on synthetic frames (no faces, no ML).

    python scripts/bench_render.py make DIR N   # write N 1600x1200 JPEGs + a cache
    /usr/bin/time -l python scripts/bench_render.py run DIR   # macOS; -v on Linux
    python scripts/bench_render.py run DIR --json results/bench_render.json

--json records frames, frame size, wall time, the process's peak resident memory
(getrusage, the same figure /usr/bin/time reports), CPU and load average.

`make` is separate so image generation does not count toward the render's
peak memory. The cache marks every image as a selected winner with eyes at a
different position per frame, so render does a real warp each time.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from facelapse import pipeline
from facelapse.cache import Cache
from facelapse.config import Config
from facelapse.records import Face, ImageRecord, Timestamp, TsSource

W, H = 1600, 1200


def _config(root: Path) -> Config:
    cfg = Config()
    cfg.cache_db = str(root / "cache.db")
    cfg.out_dir = str(root / "out")
    return cfg


def make(root: Path, n: int) -> None:
    photos = root / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    cache = Cache(_config(root).cache_db)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    start = datetime(2015, 1, 1)
    for i in range(n):
        base = np.stack(
            [
                128 + 80 * np.sin(xx / (40 + i % 17) + i),
                120 + 60 * np.cos(yy / (35 + i % 13) + i),
                100 + 50 * np.sin((xx + yy) / 90 + i),
            ],
            axis=-1,
        ).astype(np.uint8)
        path = photos / f"frame_{i:05d}.jpg"
        Image.fromarray(base).save(path, quality=85)
        x, y = 500 + rng.uniform(-200, 200), 500 + rng.uniform(-150, 150)
        kps = np.array(
            [[x, y], [x + 180, y], [x + 90, y + 90], [x + 30, y + 160], [x + 150, y + 160]]
        )
        face = Face(
            content_hash=f"bench{i:05d}",
            face_id=0,
            bbox=np.array([x - 80, y - 120, x + 260, y + 260], np.float32),
            kps=kps.astype(np.float32),
            embedding=np.zeros(512, np.float32),
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            det_score=1.0,
            is_self=True,
            selected=True,
            bucket=str(i),
            composite=1.0,
        )
        ts = Timestamp(start + timedelta(days=7 * i), TsSource.EXIF_ORIGINAL)
        cache.upsert_image(ImageRecord(face.content_hash, str(path), W, H, ts, faces=[face]))
    cache.close()


def _cpu() -> str:
    out = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
    )
    return out.stdout.strip() or platform.processor() or "not recorded"


def run(root: Path, json_path: Path | None = None) -> None:
    load_before = os.getloadavg()
    t0 = time.perf_counter()
    result = pipeline.render_final(_config(root))
    wall = time.perf_counter() - t0
    load_after = os.getloadavg()
    # ru_maxrss is bytes on macOS and KiB on Linux
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    print(f"[bench] wall {wall:.1f} s, peak RSS {peak_bytes / 2**20:.0f} MiB -> {result['video']}")
    if json_path:
        frames = len(list((root / "photos").glob("*.jpg")))
        record = {
            "frames": frames,
            "width": W,
            "height": H,
            "wall_s": round(wall, 2),
            "peak_rss_bytes": peak_bytes,
            "cpu": _cpu(),
            "machine": f"{platform.machine()} {platform.system()}",
            "python": platform.python_version(),
            "load_before": [round(x, 2) for x in load_before],
            "load_after": [round(x, 2) for x in load_after],
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"[bench] wrote {json_path}")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        out = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--json" else None
        run(Path(sys.argv[2]), out)

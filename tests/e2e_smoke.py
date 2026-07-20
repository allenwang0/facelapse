"""End-to-end smoke test with the ML stages STUBBED. Proves the orchestration,
cache, identity margin logic, alignment, bucketing, photometric, smoothing,
timeline overlay, and encode all execute and yield a real mp4 + manifest +
review page. The ML models are the only unproven parts and they are isolated
behind these stubs.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelapse.config import Config
from facelapse.records import Face
from facelapse import pipeline
from facelapse.cache import Cache


def _dense_neutral(cx: float, cy: float, scale: float) -> np.ndarray:
    """478x2 landmarks with the specific indices neutral.py/landmarks.py read
    set to a neutral, eyes-open, mouth-closed geometry."""
    d = np.zeros((478, 2), np.float32)
    d[:] = [cx, cy]
    # eyes
    d[33] = [cx - 40 * scale, cy - 20 * scale]   # L outer
    d[133] = [cx - 12 * scale, cy - 20 * scale]  # L inner
    d[159] = [cx - 26 * scale, cy - 26 * scale]  # L top
    d[145] = [cx - 26 * scale, cy - 16 * scale]  # L bottom (open eye => gap)
    d[362] = [cx + 12 * scale, cy - 20 * scale]
    d[263] = [cx + 40 * scale, cy - 20 * scale]
    d[386] = [cx + 26 * scale, cy - 26 * scale]
    d[374] = [cx + 26 * scale, cy - 16 * scale]
    # mouth closed (small vertical gap), corners level (no smile)
    d[61] = [cx - 22 * scale, cy + 40 * scale]
    d[291] = [cx + 22 * scale, cy + 40 * scale]
    d[13] = [cx, cy + 38 * scale]
    d[14] = [cx, cy + 42 * scale]
    d[168] = [cx, cy - 24 * scale]               # nasion
    return d


class StubDetector:
    """Returns preset faces per image. 'me_*' images get an embedding near the
    seed; 'other_*' images get a distant embedding (a lookalike-ish sibling)."""
    def __init__(self):
        rng = np.random.default_rng(7)
        self.me = _norm(rng.random(512))
        self.sib = _norm(0.6 * self.me + 0.4 * _norm(rng.random(512)))  # confusable

    def analyze(self, rgb, content_hash):
        name = _name_by_hash.get(content_hash, "")
        h, w = rgb.shape[:2]
        cx, cy = w / 2, h / 2
        faces = []
        # subject face (you), roughly frontal, centered
        emb = self.me if name.startswith("me") else _norm(np.random.default_rng(1).random(512))
        yaw = 5.0 if "turn" not in name else 30.0
        f = Face(content_hash=content_hash, face_id=0,
                 bbox=np.array([cx - 90, cy - 110, cx + 90, cy + 110], np.float32),
                 kps=np.array([[cx - 35, cy - 20], [cx + 35, cy - 20], [cx, cy + 5],
                               [cx - 22, cy + 40], [cx + 22, cy + 40]], np.float32),
                 embedding=emb, yaw=yaw, pitch=3.0, roll=0.0, det_score=0.95)
        faces.append(f)
        # add a sibling face in "group" images to exercise the margin gate
        if "group" in name:
            g = Face(content_hash=content_hash, face_id=1,
                     bbox=np.array([cx + 60, cy - 80, cx + 200, cy + 120], np.float32),
                     kps=np.array([[cx + 100, cy - 20], [cx + 150, cy - 20], [cx + 125, cy + 5],
                                   [cx + 108, cy + 40], [cx + 142, cy + 40]], np.float32),
                     embedding=self.sib, yaw=8.0, pitch=2.0, roll=0.0, det_score=0.9)
            faces.append(g)
        return faces

    def embed(self, rgb):
        return self.me


class StubMesh:
    def landmarks(self, rgb_crop):
        h, w = rgb_crop.shape[:2]
        return _dense_neutral(w / 2, h / 2, max(w, h) / 200.0)


def _norm(v):
    # zero-center THEN normalize, so unrelated identities are near-orthogonal
    # like real ArcFace embeddings (all-positive random vectors are not).
    v = np.asarray(v, np.float32)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


_name_by_hash: dict[str, str] = {}


def _make_image(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    # a vaguely face-ish gradient so decode/sharpness/photometric have real pixels
    base = np.zeros((480, 480, 3), np.uint8)
    yy, xx = np.mgrid[0:480, 0:480]
    base[..., 0] = (128 + 80 * np.sin(xx / 60 + seed)).astype(np.uint8)
    base[..., 1] = (110 + 60 * np.cos(yy / 50 + seed)).astype(np.uint8)
    base[..., 2] = (90 + 40 * np.sin((xx + yy) / 70)).astype(np.uint8)
    base = (base.astype(np.int16) + rng.integers(-10, 10, base.shape)).clip(0, 255).astype(np.uint8)
    Image.fromarray(base).save(path, quality=90)


def main():
    work = Path(tempfile.mkdtemp())
    photos = work / "photos"; photos.mkdir()
    cfg = Config()
    cfg.cache_db = str(work / "cache.db")
    cfg.out_dir = str(work / "out")
    cfg.id_t_abs = 0.2      # relaxed for synthetic embeddings
    cfg.id_margin = 0.05
    cfg.min_interocular_px = 20.0
    cfg.bucket_grain = "month"
    cfg.fps = 4.0

    # build a spread of dated images: several "me", one "group" (sibling present),
    # one "turned" (should fail pose), one "other" (not me).
    plan = [
        ("me_2019", datetime(2019, 3, 10)),
        ("me_2019b", datetime(2019, 3, 12)),   # same month -> dedup/bucket competition
        ("me_group_2020", datetime(2020, 6, 1)),
        ("me_2021", datetime(2021, 9, 15)),
        ("me_turn_2022", datetime(2022, 1, 5)),
        ("other_2022", datetime(2022, 1, 6)),
        ("me_2023", datetime(2023, 11, 20)),
    ]
    import os
    from facelapse import ingest
    for i, (name, dt) in enumerate(plan):
        p = photos / f"{name}.jpg"
        _make_image(p, seed=i)
        os.utime(p, (dt.timestamp(), dt.timestamp()))  # mtime fallback date
        _name_by_hash[ingest.content_hash(p)] = name

    det, mesh = StubDetector(), StubMesh()

    # seed the reference chain with the "me" embedding at a couple of dates
    cache = Cache(cfg.cache_db)
    cache.add_seed("seed_2019.jpg", datetime(2019, 1, 1), det.me)
    cache.add_seed("seed_2023.jpg", datetime(2023, 1, 1), det.me)
    cache.close()

    pipeline.analyze(str(photos), cfg, detector=det, mesh=mesh, force=True)
    pipeline.score(cfg, au_scorer=None)   # geometry-only neutral
    winners = pipeline.select(cfg)
    result = pipeline.render_final(cfg)

    video = Path(result["video"]); manifest = json.loads(Path(result["manifest"]).read_text())
    assert video.exists() and video.stat().st_size > 0, "no video produced"
    print(f"\nOK: video {video.stat().st_size} bytes, {manifest['n_frames']} frames")
    print("selected dates:", [r["date"][:10] for r in manifest["frames"]])
    print("date sources:", {r["date_source"] for r in manifest["frames"]})
    # the 'other' and 'turned' images must not appear
    names = [_name_by_hash[r["content_hash"]] for r in manifest["frames"]]
    print("selected names:", names)
    assert not any(n.startswith("other") for n in names), "identity gate let a non-self through"
    assert not any("turn" in n for n in names), "pose gate let a turned face through"
    print("PASS: identity + pose gates excluded the right images")


if __name__ == "__main__":
    main()

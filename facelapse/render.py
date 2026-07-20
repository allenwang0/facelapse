"""Final assembly: aligned + photometrically normalized frames, in date order,
timeline overlay burned in, encoded to H.264. Also emits a manifest (which
source image became which frame, with scores and date provenance) and an
optional ASS subtitle so the timeline can be restyled without re-rendering.

Interpolation note: v1 renders honest cuts (crossfade_frames=0). If you add
FILM/RIFE later, the date LABEL must hold on the source frame's date across
interpolated frames while the marker GLIDES; that logic belongs here, keyed to
whether a frame is real or interpolated.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .timeline import TimelineRenderer, robust_range


def _even(frame: np.ndarray) -> np.ndarray:
    # libx264 needs even dimensions
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


def render_video(
    frames: list[np.ndarray],
    dates: list[datetime],
    confident: list[bool],
    manifest_rows: list[dict],
    cfg,
) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "timelapse.mp4"
    manifest_path = out_dir / "manifest.json"

    if cfg.draw_timeline:
        lo, hi = robust_range(dates, confident, cfg.date_range_percentiles)
        tl = TimelineRenderer(cfg, lo, hi)
        frames = [tl.draw(f, dt, c) for f, dt, c in zip(frames, dates, confident)]

    writer = imageio.get_writer(
        video_path, fps=cfg.fps, codec="libx264",
        quality=8, macro_block_size=1,
    )
    prev = None
    for f in frames:
        f = _even(f)
        if cfg.crossfade_frames > 0 and prev is not None:
            for a in np.linspace(0, 1, cfg.crossfade_frames + 2)[1:-1]:
                blend = (prev.astype(np.float32) * (1 - a) + f.astype(np.float32) * a)
                writer.append_data(blend.astype(np.uint8))
        writer.append_data(f)
        prev = f
    writer.close()

    manifest = {
        "n_frames": len(frames),
        "fps": cfg.fps,
        "bucket_grain": cfg.bucket_grain,
        "frames": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return {"video": str(video_path), "manifest": str(manifest_path)}

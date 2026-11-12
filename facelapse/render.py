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
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from .timeline import TimelineRenderer, percentile_range


def _even(frame: np.ndarray) -> np.ndarray:
    # libx264 needs even dimensions
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


STORY = (1080, 1920)
GIF_W = 360
HOLD_SECONDS = 1.0


@dataclass
class OutputOptions:
    """layout: "canvas" (cfg.canvas_w x canvas_h), "square" (canvas_w square
    around the eyes) or "story" (1080x1920, for phone stories). seconds:
    fixed total length; fps follows from the frame count and the last frame
    holds for one second. gif: also write a looping timelapse.gif. The story
    layout, GIF and fixed length are ported from face-lapse (commit 50de507)."""

    layout: str = "canvas"
    gif: bool = False
    seconds: float | None = None


def output_size(cfg, layout: str) -> tuple[int, int]:
    if layout == "square":
        return cfg.canvas_w, cfg.canvas_w
    if layout == "story":
        return STORY
    return cfg.canvas_w, cfg.canvas_h


def apply_layout(frame: np.ndarray, cfg, layout: str) -> np.ndarray:
    if layout == "square":
        h, w = frame.shape[:2]
        eye_y = 0.5 * (cfg.left_eye_xy[1] + cfg.right_eye_xy[1])
        top = int(min(max(0, eye_y - 0.4 * w), h - w))
        return frame[top : top + w]
    if layout == "story":
        W, H = STORY
        scaled = Image.fromarray(frame).resize(
            (W, round(frame.shape[0] * W / frame.shape[1])), Image.Resampling.LANCZOS
        )
        canvas = Image.new("RGB", STORY, (20, 20, 20))
        canvas.paste(scaled, (0, max(0, (H - scaled.height) // 2)))
        return np.asarray(canvas)
    return frame


def _fps(cfg, n: int, seconds: float | None) -> float:
    if seconds is None:
        return float(cfg.fps)
    if seconds <= HOLD_SECONDS:
        raise ValueError(f"--seconds must exceed the {HOLD_SECONDS:.0f} s end hold")
    return max(n, 1) / (seconds - HOLD_SECONDS)


def render_video(
    frames: Iterable[np.ndarray],
    dates: list[datetime],
    confident: list[bool],
    manifest_rows: list[dict],
    cfg,
    options: OutputOptions | None = None,
) -> dict:
    opts = options or OutputOptions()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "timelapse.mp4"
    manifest_path = out_dir / "manifest.json"
    fps = _fps(cfg, len(manifest_rows), opts.seconds)
    hold = round(fps * HOLD_SECONDS) if opts.seconds is not None else 0

    tl = None
    if cfg.draw_timeline and dates:
        lo, hi = percentile_range(dates, confident, cfg.date_range_percentiles)
        tl = TimelineRenderer(cfg, lo, hi, output_size(cfg, opts.layout))

    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    gif_frames: list[Image.Image] = []
    prev = None
    for f, dt, c in zip(frames, dates, confident):
        f = apply_layout(f, cfg, opts.layout)
        if tl is not None:
            f = tl.draw(f, dt, c)
        f = _even(f)
        if cfg.crossfade_frames > 0 and prev is not None:
            for a in np.linspace(0, 1, cfg.crossfade_frames + 2)[1:-1]:
                blend = prev.astype(np.float32) * (1 - a) + f.astype(np.float32) * a
                writer.append_data(blend.astype(np.uint8))
        writer.append_data(f)
        if opts.gif:
            img = Image.fromarray(f)
            gif_frames.append(img.resize((GIF_W, round(img.height * GIF_W / img.width))))
        prev = f
    for _ in range(hold if prev is not None else 0):
        writer.append_data(prev)
    writer.close()

    result = {"video": str(video_path), "manifest": str(manifest_path)}
    if opts.gif and gif_frames:
        gif_path = out_dir / "timelapse.gif"
        durations = [round(1000 / fps)] * len(gif_frames)
        durations[-1] += round(1000 * HOLD_SECONDS)
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=durations,
            loop=0,
            optimize=True,
        )
        result["gif"] = str(gif_path)

    manifest = {
        "n_frames": len(manifest_rows),
        "fps": fps,
        "layout": opts.layout,
        "hold_frames": hold,
        "bucket_grain": cfg.bucket_grain,
        "frames": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return result

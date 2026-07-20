"""Lower-third timeline overlay. The axis is proportional to REAL elapsed
time, not to frame index: the images tick forward at a steady rhythm while
the marker jumps unevenly, so the viewer feels the difference between the
periods you lived in front of a camera and the periods you vanished from the
record. That tension is the point of a dated aging video.

The bar scale is clamped to a robust percentile range of HIGH-CONFIDENCE
dates so a single corrupted 1970/2038 timestamp can't compress the whole bar
into a sliver.

Label: year large and persistent (the emotionally salient unit), full date
smaller beneath. January-1 tick marks turn the bar into a readable calendar.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _font(size: int, path: str | None):
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    for cand in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def robust_range(dates: list[datetime], confident: list[bool],
                 pct: tuple[float, float]) -> tuple[float, float]:
    """Bar min/max in epoch seconds, from confident dates, percentile-clamped."""
    epochs = np.array([d.timestamp() for d, c in zip(dates, confident) if c and d])
    if epochs.size == 0:
        epochs = np.array([d.timestamp() for d in dates if d])
    lo = float(np.percentile(epochs, pct[0]))
    hi = float(np.percentile(epochs, pct[1]))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _year_boundaries(lo: float, hi: float) -> list[tuple[float, int]]:
    y0 = datetime.fromtimestamp(lo).year
    y1 = datetime.fromtimestamp(hi).year
    out = []
    for y in range(y0, y1 + 2):
        e = datetime(y, 1, 1).timestamp()
        if lo <= e <= hi:
            out.append((e, y))
    return out


class TimelineRenderer:
    def __init__(self, cfg, lo: float, hi: float):
        self.cfg = cfg
        self.lo, self.hi = lo, hi
        self.W, self.H = cfg.canvas_w, cfg.canvas_h
        self.band_h = int(self.H * cfg.timeline_band_frac)
        self.pad = int(self.W * 0.06)
        self.bar_y = self.H - int(self.band_h * 0.35)
        self.year_font = _font(int(self.band_h * 0.42), cfg.font_path)
        self.date_font = _font(int(self.band_h * 0.24), cfg.font_path)

    def _x(self, epoch: float) -> float:
        t = (epoch - self.lo) / (self.hi - self.lo)
        t = min(1.0, max(0.0, t))
        return self.pad + t * (self.W - 2 * self.pad)

    def draw(self, frame_rgb: np.ndarray, dt: datetime, confident: bool) -> np.ndarray:
        img = Image.fromarray(frame_rgb).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)

        # semi-transparent lower-third band for constant text contrast
        d.rectangle([0, self.H - self.band_h, self.W, self.H], fill=(0, 0, 0, 150))

        bx0, bx1 = self.pad, self.W - self.pad
        d.line([(bx0, self.bar_y), (bx1, self.bar_y)], fill=(255, 255, 255, 180), width=2)

        # year tick marks + small year labels along the bar
        for epoch, year in _year_boundaries(self.lo, self.hi):
            x = self._x(epoch)
            d.line([(x, self.bar_y - 6), (x, self.bar_y + 6)], fill=(255, 255, 255, 140), width=1)

        # marker for the current frame's real date
        mx = self._x(dt.timestamp())
        d.ellipse([mx - 6, self.bar_y - 6, mx + 6, self.bar_y + 6],
                  fill=(255, 80, 80, 255))

        # label: year primary, date secondary. tilde-prefix if low confidence.
        year_txt = dt.strftime("%Y")
        date_txt = dt.strftime("%b %d") if confident else "~" + dt.strftime("%b %Y")
        ty = self.H - self.band_h + int(self.band_h * 0.10)
        d.text((self.pad, ty), year_txt, font=self.year_font, fill=(255, 255, 255, 255))
        yw = d.textlength(year_txt, font=self.year_font)
        d.text((self.pad + yw + 14, ty + int(self.band_h * 0.16)), date_txt,
               font=self.date_font, fill=(220, 220, 220, 255))

        out = Image.alpha_composite(img, overlay).convert("RGB")
        return np.asarray(out)

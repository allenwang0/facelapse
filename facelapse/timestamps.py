"""Resolve a capture timestamp with an explicit priority ladder and record
WHICH source won, because the temporal axis of the whole video is only as
good as this. A missing trustworthy date is itself a reject signal: it
correlates with screenshots, memes, and edited re-saves, which is exactly
the junk you want gone.

Priority: Takeout sidecar JSON > EXIF DateTimeOriginal > EXIF DateTimeDigitized
> filename pattern > file mtime.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from PIL import Image, ExifTags

from .records import Timestamp, TsSource

_EXIF_TAG = {v: k for k, v in ExifTags.TAGS.items()}
_ORIGINAL = _EXIF_TAG.get("DateTimeOriginal")
_DIGITIZED = _EXIF_TAG.get("DateTimeDigitized")

# Common phone filename stamps. Order matters: most specific first.
_FILENAME_PATTERNS = [
    re.compile(r"(?:IMG|PXL|VID)[_-](\d{8})[_-]?(\d{6})?"),   # IMG_20210517_143000
    re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})[ _T-](\d{2})[-_:]?(\d{2})"),  # 2021-05-17_14-30
    re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)"),        # bare 20210517
]


def _parse_exif_dt(s: str) -> datetime | None:
    # EXIF datetime is "YYYY:MM:DD HH:MM:SS"
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def _from_takeout_sidecar(path: Path) -> datetime | None:
    """Google Takeout writes a JSON sidecar next to each media file. The
    naming has drifted over the years; try the common variants."""
    candidates = [
        path.with_suffix(path.suffix + ".json"),
        path.with_suffix(path.suffix + ".supplemental-metadata.json"),
        path.with_name(path.name + ".json"),
    ]
    for c in candidates:
        if not c.exists():
            continue
        try:
            data = json.loads(c.read_text())
        except Exception:
            continue
        for key in ("photoTakenTime", "creationTime"):
            node = data.get(key)
            if isinstance(node, dict) and "timestamp" in node:
                try:
                    return datetime.fromtimestamp(int(node["timestamp"]))
                except (ValueError, OverflowError, OSError):
                    pass
    return None


def _from_exif(path: Path) -> tuple[datetime | None, TsSource]:
    try:
        exif = Image.open(path).getexif()
    except Exception:
        return None, TsSource.NONE
    if _ORIGINAL and _ORIGINAL in exif:
        dt = _parse_exif_dt(str(exif[_ORIGINAL]))
        if dt:
            return dt, TsSource.EXIF_ORIGINAL
    if _DIGITIZED and _DIGITIZED in exif:
        dt = _parse_exif_dt(str(exif[_DIGITIZED]))
        if dt:
            return dt, TsSource.EXIF_DIGITIZED
    return None, TsSource.NONE


def _from_filename(path: Path) -> datetime | None:
    name = path.name
    for pat in _FILENAME_PATTERNS:
        m = pat.search(name)
        if not m:
            continue
        g = [x for x in m.groups() if x]
        try:
            if len(g) == 1 and len(g[0]) == 8:          # YYYYMMDD
                return datetime.strptime(g[0], "%Y%m%d")
            if len(g) == 2 and len(g[0]) == 8:          # YYYYMMDD + HHMMSS
                return datetime.strptime(g[0] + g[1], "%Y%m%d%H%M%S")
            if len(g) >= 5:                              # Y M D H M
                return datetime(int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]))
        except ValueError:
            continue
    return None


def resolve(path: Path) -> Timestamp:
    dt = _from_takeout_sidecar(path)
    if dt:
        return Timestamp(dt, TsSource.TAKEOUT_JSON)

    dt, src = _from_exif(path)
    if dt:
        return Timestamp(dt, src)

    dt = _from_filename(path)
    if dt:
        return Timestamp(dt, TsSource.FILENAME)

    try:
        return Timestamp(datetime.fromtimestamp(path.stat().st_mtime), TsSource.MTIME)
    except OSError:
        return Timestamp(None, TsSource.NONE)

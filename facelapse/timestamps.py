"""Resolve a capture timestamp with an explicit priority ladder and record
WHICH source won, because the temporal axis of the whole video is only as
good as this. A missing trustworthy date is itself a reject signal: it
correlates with screenshots, memes, and edited re-saves, which is exactly
the junk you want gone.

Priority: EXIF DateTimeOriginal > EXIF DateTimeDigitized > Takeout sidecar
JSON > filename pattern > file mtime. EXIF is the camera's wall clock, which
is what the date label should show. The Takeout epoch is UTC; it is shifted
by EXIF OffsetTimeOriginal when present and otherwise kept in UTC, never
converted to the zone of the machine running the tool.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Any

from PIL import ExifTags, Image

from .records import Timestamp, TsSource

_EXIF_TAG = {v: k for k, v in ExifTags.TAGS.items()}
_ORIGINAL = _EXIF_TAG.get("DateTimeOriginal")
_DIGITIZED = _EXIF_TAG.get("DateTimeDigitized")
_OFFSET_ORIGINAL = _EXIF_TAG.get("OffsetTimeOriginal", 36881)

# Common phone filename stamps. Order matters: most specific first.
_FILENAME_PATTERNS = [
    re.compile(r"(?:IMG|PXL|VID)[_-](\d{8})[_-]?(\d{6})?"),  # IMG_20210517_143000
    re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})[ _T-](\d{2})[-_:]?(\d{2})"),  # 2021-05-17_14-30
    re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)"),  # bare 20210517
]


def _parse_exif_dt(s: str) -> datetime | None:
    # EXIF datetime is "YYYY:MM:DD HH:MM:SS"
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def load_takeout_sidecar(path: Path) -> dict[str, Any] | None:
    """Google Takeout writes a JSON sidecar next to each media file. The
    naming has drifted over the years; try the common variants."""
    for c in takeout_sidecar_names(path.name):
        sidecar = path.with_name(c)
        if not sidecar.exists():
            continue
        try:
            data = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def takeout_sidecar_names(name: str) -> list[str]:
    return [name + ".json", name + ".supplemental-metadata.json"]


def _from_takeout(data: dict[str, Any], offset: timedelta | None) -> datetime | None:
    """Takeout stores a UTC epoch. Shift it by the camera's EXIF offset when
    known, else keep UTC; never by the zone of the machine running this."""
    for key in ("photoTakenTime", "creationTime"):
        node = data.get(key)
        if isinstance(node, dict) and "timestamp" in node:
            try:
                utc = datetime.fromtimestamp(int(node["timestamp"]), timezone.utc)
            except (ValueError, OverflowError, OSError):
                continue
            return (utc + (offset or timedelta(0))).replace(tzinfo=None)
    return None


def _parse_offset(s: object) -> timedelta | None:
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", str(s).strip())
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    return sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))


def _from_exif(src: Path | IO[bytes]) -> tuple[datetime | None, TsSource, timedelta | None]:
    try:
        exif = Image.open(src).getexif()
    except OSError:
        return None, TsSource.NONE, None
    # the capture dates live in the Exif sub-IFD, not in IFD0
    tags = dict(exif)
    tags.update(exif.get_ifd(ExifTags.IFD.Exif))
    offset = _parse_offset(tags[_OFFSET_ORIGINAL]) if _OFFSET_ORIGINAL in tags else None
    for tag, source in ((_ORIGINAL, TsSource.EXIF_ORIGINAL), (_DIGITIZED, TsSource.EXIF_DIGITIZED)):
        if tag in tags:
            dt = _parse_exif_dt(str(tags[tag]))
            if dt:
                return dt, source, offset
    return None, TsSource.NONE, offset


def _from_filename(path: Path) -> datetime | None:
    name = path.name
    for pat in _FILENAME_PATTERNS:
        m = pat.search(name)
        if not m:
            continue
        g = [x for x in m.groups() if x]
        try:
            if len(g) == 1 and len(g[0]) == 8:  # YYYYMMDD
                return datetime.strptime(g[0], "%Y%m%d")
            if len(g) == 3:  # YYYY MM DD
                return datetime(int(g[0]), int(g[1]), int(g[2]))
            if len(g) == 2 and len(g[0]) == 8:  # YYYYMMDD + HHMMSS
                return datetime.strptime(g[0] + g[1], "%Y%m%d%H%M%S")
            if len(g) >= 5:  # Y M D H M
                return datetime(int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]))
        except ValueError:
            continue
    return None


def resolve(path: Path) -> Timestamp:
    try:
        mtime: float | None = path.stat().st_mtime
    except OSError:
        mtime = None
    return resolve_from(path.name, path, load_takeout_sidecar(path), mtime)


def resolve_from(
    name: str,
    image: Path | IO[bytes],
    sidecar: dict[str, Any] | None,
    mtime: float | None,
) -> Timestamp:
    """The ladder itself, independent of where the bytes live (a folder or a
    Takeout zip member)."""
    dt, src, offset = _from_exif(image)
    if dt:
        return Timestamp(dt, src)

    if sidecar is not None:
        dt = _from_takeout(sidecar, offset)
        if dt:
            return Timestamp(dt, TsSource.TAKEOUT_JSON)

    dt = _from_filename(Path(name))
    if dt:
        return Timestamp(dt, TsSource.FILENAME)

    if mtime is None:
        return Timestamp(None, TsSource.NONE)
    return Timestamp(datetime.fromtimestamp(mtime), TsSource.MTIME)

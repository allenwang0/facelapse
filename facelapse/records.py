"""Shared data types. Deliberately flat dataclasses, not an ORM: this is
single-use local infrastructure, so the schema is the SQLite tables in
cache.py and these are just typed views over rows.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


class TsSource(enum.IntEnum):
    """Ordered by trustworthiness; lower value = more trusted."""
    TAKEOUT_JSON = 0
    EXIF_ORIGINAL = 1
    EXIF_DIGITIZED = 2
    FILENAME = 3
    MTIME = 4
    NONE = 5


# Confidence buckets derived from source. FILENAME is date-accurate but
# occasionally wrong; MTIME is close to fiction.
CONFIDENCE = {
    TsSource.TAKEOUT_JSON: "high",
    TsSource.EXIF_ORIGINAL: "high",
    TsSource.EXIF_DIGITIZED: "high",
    TsSource.FILENAME: "medium",
    TsSource.MTIME: "low",
    TsSource.NONE: "none",
}


@dataclass
class Timestamp:
    value: datetime | None
    source: TsSource

    @property
    def confidence(self) -> str:
        return CONFIDENCE[self.source]

    @property
    def is_trustworthy(self) -> bool:
        return self.source <= TsSource.FILENAME and self.value is not None


@dataclass
class Face:
    """One detected face inside one image, plus every score we attach to it.

    kps: 5x2 float array [left_eye, right_eye, nose, mouth_l, mouth_r]
         (InsightFace order -- see detection.py # VERIFY)
    dense: 468x2 (or 478x2) MediaPipe landmarks in image pixels, or None
    embedding: L2-normalized ArcFace vector (512-d for buffalo_l)
    pose: (yaw, pitch, roll) degrees -- see detection.py # VERIFY on axis order
    """
    content_hash: str
    face_id: int
    bbox: np.ndarray                 # (4,) x1,y1,x2,y2
    kps: np.ndarray                  # (5,2)
    embedding: np.ndarray            # (512,)
    yaw: float
    pitch: float
    roll: float
    det_score: float
    dense: np.ndarray | None = None  # (468,2) or None

    # scores populated by later stages (None until computed)
    interocular_px: float | None = None
    sharpness: float | None = None
    ear: float | None = None         # eye aspect ratio (eyes-open signal)
    mar: float | None = None         # mouth aspect ratio (mouth-open signal)
    neutral_penalty: float | None = None
    id_sim: float | None = None      # cosine sim to age-matched reference
    id_margin: float | None = None   # gap to 2nd-best face in same image
    is_self: bool = False            # passed the identity gate as THE you-face
    passes_pose: bool | None = None
    passes_neutral: bool | None = None
    composite: float | None = None   # per-bucket selection score
    selected: bool = False
    bucket: str | None = None


@dataclass
class ImageRecord:
    content_hash: str
    path: str
    width: int
    height: int
    ts: Timestamp
    faces: list[Face] = field(default_factory=list)
    decoded_ok: bool = True
    error: str | None = None

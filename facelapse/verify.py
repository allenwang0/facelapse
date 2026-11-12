"""`facelapse verify PHOTO`: check the library assumptions marked # VERIFY
on one of your own photos before trusting a full run.

It writes verify.png with the five detector keypoints numbered, the
MediaPipe indices that landmarks.py reads highlighted, and yaw, pitch and
roll printed. It also runs checks that need no labeling, by transforming
the face crop and re-detecting:

  keypoint order  eyes are kps 0 and 1 left to right, nose below the eyes,
                  mouth corners below the nose (detection.py, quality.py)
  mirror          flipping the crop horizontally flips the sign of yaw and
                  roll and keeps pitch; this pins which pose axis is yaw
                  (detection.py _pose_triple)
  rotate          rotating the crop in-plane by 15 degrees moves roll by
                  about 15 and leaves yaw about where it was
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .records import Face

MIN_ANGLE = 3.0  # degrees; below this a sign comparison is noise
ROLL_TOL = 4.0


@dataclass
class Check:
    name: str
    passed: bool | None  # None: inconclusive on this photo
    detail: str


def _largest(faces: list[Face]) -> Face | None:
    if not faces:
        return None
    return max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))


def _crop(rgb: np.ndarray, f: Face, margin: float = 0.4) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in f.bbox)
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    h, w = rgb.shape[:2]
    a, b = max(0, int(x1 - mx)), max(0, int(y1 - my))
    return np.ascontiguousarray(rgb[b : min(h, int(y2 + my)), a : min(w, int(x2 + mx))])


def _same_sign(a: float, b: float) -> bool | None:
    if abs(a) < MIN_ANGLE or abs(b) < MIN_ANGLE:
        return None
    return (a > 0) == (b > 0)


def keypoint_order(f: Face) -> Check:
    le, re, nose, ml, mr = f.kps
    ok = bool(le[0] < re[0] and nose[1] > max(le[1], re[1]) and min(ml[1], mr[1]) > nose[1])
    ok = ok and bool(ml[0] < mr[0])
    detail = "eyes left-to-right, then nose, then mouth corners" if ok else f"kps={f.kps.tolist()}"
    return Check("keypoint order", ok, detail)


def mirror_check(detector: Any, crop: np.ndarray, base: Face) -> Check:
    m = _largest(detector.analyze(np.ascontiguousarray(crop[:, ::-1]), "verify"))
    if m is None:
        return Check("mirror", False, "no face found in the mirrored crop")
    yaw_same = _same_sign(base.yaw, m.yaw)
    roll_same = _same_sign(base.roll, m.roll)
    pitch_same = _same_sign(base.pitch, m.pitch)
    detail = (
        f"yaw {base.yaw:+.1f} -> {m.yaw:+.1f}, pitch {base.pitch:+.1f} -> {m.pitch:+.1f}, "
        f"roll {base.roll:+.1f} -> {m.roll:+.1f}"
    )
    if yaw_same is None:
        return Check("mirror", None, detail + " (|yaw| too small to compare; use a turned face)")
    passed = not yaw_same and roll_same is not True and pitch_same is not False
    return Check("mirror", passed, detail)


def rotate_check(detector: Any, crop: np.ndarray, base: Face, degrees: float = 15.0) -> Check:
    h, w = crop.shape[:2]
    # OpenCV's positive angle is counterclockwise on screen; negate it so the
    # eye line turns by +degrees in image coordinates (y down), the
    # convention roll uses.
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -degrees, 1.0)
    rotated = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    r = _largest(detector.analyze(rotated, "verify"))
    if r is None:
        return Check("rotate", False, "no face found in the rotated crop")
    d_roll = r.roll - base.roll
    detail = (
        f"roll {base.roll:+.1f} -> {r.roll:+.1f} (moved {d_roll:+.1f}, expected {degrees:+.1f})"
    )
    return Check("rotate", abs(d_roll - degrees) <= ROLL_TOL, detail)


def run_checks(detector: Any, rgb: np.ndarray) -> tuple[Face | None, list[Check]]:
    f = _largest(detector.analyze(rgb, "verify"))
    if f is None:
        return None, [Check("detect", False, "no face found")]
    crop = _crop(rgb, f)
    base = _largest(detector.analyze(crop, "verify"))
    if base is None:
        return f, [Check("detect", False, "face lost after cropping")]
    return f, [
        keypoint_order(f),
        mirror_check(detector, crop, base),
        rotate_check(detector, crop, base),
    ]


def write_png(rgb: np.ndarray, f: Face, dense: np.ndarray | None, path: Path) -> None:
    from . import landmarks as L

    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    r = max(3, int((f.bbox[2] - f.bbox[0]) / 60))
    d.rectangle([float(v) for v in f.bbox], outline=(255, 255, 0), width=2)
    for i, (x, y) in enumerate(f.kps):
        d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=2)
        d.text((x + r + 2, y - r), str(i), fill=(255, 0, 0))
    if dense is not None:
        idx = [*L.LEFT_EYE.values(), *L.RIGHT_EYE.values(), *L.MOUTH.values(), L.NASION]
        for i in idx:
            x, y = dense[i]
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 255, 255))
            d.text((x + 3, y + 3), str(i), fill=(0, 255, 255))
    d.text(
        (10, 10),
        f"yaw {f.yaw:+.1f}  pitch {f.pitch:+.1f}  roll {f.roll:+.1f}",
        fill=(255, 255, 255),
    )
    img.save(path)

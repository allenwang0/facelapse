"""Synthetic faces and a detector that reads them, for tests, the demo and
CI. No ML and no real photos: a face is a skin-colored ellipse with
saturated color markers for the eyes (blue), nose (green), mouth corners
(red), mouth center (yellow) and an identity badge on the forehead. The
detector finds those markers with color thresholds and returns the same
Face records the InsightFace backend does, so every downstream stage runs
unchanged. Select it with `--backend synthetic`.

Geometry, in units of the interocular distance (iod), in a frame whose x
axis runs along the eye line: eyes at (-0.5, 0) and (0.5, 0); nose at
(0.5 sin(yaw), 0.55 + 0.5 sin(pitch)); mouth corners at (+-0.4, 1.0) raised
by `smile`; mouth center at (0.5 sin(yaw), 1.0). The detector inverts those
formulas, so mirroring an image flips yaw and roll and keeps pitch, as it
does for a real head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .records import Face

SKIN = (224, 172, 140)
EYE = (0, 0, 255)
NOSE = (0, 200, 0)
CORNER = (255, 0, 0)
MOUTH = (255, 255, 0)

# identity badge colors; the embedding is derived from which one is found
IDENTITIES = {
    "subject": (255, 0, 255),
    "twin": (0, 255, 255),
    "stranger": (255, 128, 0),
}
# cosine similarity of each identity's base embedding to the subject's
_SIMILARITY = {"subject": 1.0, "twin": 0.93, "stranger": 0.1}

# (low, high) RGB bounds per marker, wide enough to survive JPEG
_BOUNDS = {
    "skin": ((190, 140, 105), (255, 205, 175)),
    "eye": ((0, 0, 170), (90, 90, 255)),
    "nose": ((0, 140, 0), (90, 255, 90)),
    "corner": ((170, 0, 0), (255, 90, 90)),
    "mouth": ((170, 170, 0), (255, 255, 90)),
    "subject": ((170, 0, 170), (255, 90, 255)),
    "twin": ((0, 170, 170), (90, 255, 255)),
    "stranger": ((200, 90, 0), (255, 170, 70)),
}


@dataclass
class FaceSpec:
    cx: float  # eye midpoint, image pixels
    cy: float
    iod: float  # interocular distance, pixels
    yaw: float = 0.0  # degrees
    pitch: float = 0.0
    roll: float = 0.0
    identity: str = "subject"
    smile: float = 0.0  # mouth corner lift, in iod
    eye_open: float = 1.0  # 1 open, 0 closed


def draw_face(img: np.ndarray, s: FaceSpec) -> None:
    """Draw one face onto an RGB uint8 image in place."""
    r = math.radians(s.roll)
    ux, uy = math.cos(r), math.sin(r)

    def at(x: float, y: float) -> tuple[int, int]:
        px = s.cx + s.iod * (x * ux - y * uy)
        py = s.cy + s.iod * (x * uy + y * ux)
        return round(px), round(py)

    sy, sp = math.sin(math.radians(s.yaw)), math.sin(math.radians(s.pitch))
    axes = (round(1.1 * s.iod), round(1.5 * s.iod))
    cv2.ellipse(img, at(0.0, 0.35), axes, s.roll, 0, 360, SKIN, -1)
    eye_axes = (round(0.16 * s.iod), max(1, round(0.12 * s.iod * s.eye_open)))
    for ex in (-0.5, 0.5):
        cv2.ellipse(img, at(ex, 0.0), eye_axes, s.roll, 0, 360, EYE, -1)
    cv2.circle(img, at(0.5 * sy, 0.55 + 0.5 * sp), round(0.1 * s.iod), NOSE, -1)
    for mx in (-0.4, 0.4):
        cv2.circle(img, at(mx, 1.0 - s.smile), round(0.07 * s.iod), CORNER, -1)
    cv2.circle(img, at(0.5 * sy, 1.0), round(0.07 * s.iod), MOUTH, -1)
    badge = np.array([at(-0.3, -0.75), at(0.3, -0.75), at(0.3, -0.5), at(-0.3, -0.5)])
    cv2.fillConvexPoly(img, badge.astype(np.int32), IDENTITIES[s.identity])


def _mask(rgb: np.ndarray, key: str) -> np.ndarray:
    lo, hi = _BOUNDS[key]
    return cv2.inRange(rgb, np.array(lo, np.uint8), np.array(hi, np.uint8))


@dataclass
class _Blob:
    x: float  # centroid
    y: float
    left: int
    top: int
    w: int
    h: int
    area: int


def _blobs(mask: np.ndarray, min_area: int) -> list[_Blob]:
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= min_area:
            out.append(_Blob(float(cents[i][0]), float(cents[i][1]), x, y, w, h, area))
    return sorted(out, key=lambda b: -b.area)


@dataclass
class _Markers:
    bbox: np.ndarray
    le: np.ndarray
    re: np.ndarray
    nose: np.ndarray
    corners: tuple[np.ndarray, np.ndarray]
    mouth: np.ndarray
    eye_h: float
    eye_w: float
    identity: str | None


def _find_faces(rgb: np.ndarray) -> list[_Markers]:
    found = []
    for skin in _blobs(_mask(rgb, "skin"), min_area=400):
        x1, y1 = max(0, skin.left - 2), max(0, skin.top - 2)
        x2, y2 = skin.left + skin.w + 2, skin.top + skin.h + 2
        crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
        m = _markers_in(crop)
        if m is None:
            continue
        off = np.array([x1, y1], np.float32)
        m.le, m.re, m.nose, m.mouth = m.le + off, m.re + off, m.nose + off, m.mouth + off
        m.corners = (m.corners[0] + off, m.corners[1] + off)
        m.bbox = np.array([x1, y1, x2, y2], np.float32)
        found.append(m)
    return found


def _markers_in(crop: np.ndarray) -> _Markers | None:
    eyes = _blobs(_mask(crop, "eye"), 4)[:2]
    nose = _blobs(_mask(crop, "nose"), 3)[:1]
    corners = _blobs(_mask(crop, "corner"), 2)[:2]
    mouth = _blobs(_mask(crop, "mouth"), 2)[:1]
    if len(eyes) < 2 or not nose or len(corners) < 2 or not mouth:
        return None
    eyes.sort(key=lambda b: b.x)
    corners.sort(key=lambda b: b.x)
    identity = None
    for name in IDENTITIES:
        if _blobs(_mask(crop, name), 4):
            identity = name
            break

    def pt(b: _Blob) -> np.ndarray:
        return np.array([b.x, b.y], np.float32)

    return _Markers(
        bbox=np.zeros(4, np.float32),
        le=pt(eyes[0]),
        re=pt(eyes[1]),
        nose=pt(nose[0]),
        corners=(pt(corners[0]), pt(corners[1])),
        mouth=pt(mouth[0]),
        eye_h=float(np.mean([e.h for e in eyes])),
        eye_w=float(np.mean([e.w for e in eyes])),
        identity=identity,
    )


def _frame(m: _Markers) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Eye midpoint, unit eye-line vector, unit downward normal, iod."""
    d = m.re - m.le
    iod = float(np.linalg.norm(d))
    u = d / iod
    return (m.le + m.re) / 2, u, np.array([-u[1], u[0]], np.float32), iod


def pose(m: _Markers) -> tuple[float, float, float]:
    mid, u, v, iod = _frame(m)
    rel = m.nose - mid
    yaw = math.degrees(math.asin(float(np.clip(rel @ u / iod / 0.5, -1, 1))))
    pitch = math.degrees(math.asin(float(np.clip((rel @ v / iod - 0.55) / 0.5, -1, 1))))
    roll = math.degrees(math.atan2(float(u[1]), float(u[0])))
    return yaw, pitch, roll


def identity_embedding(identity: str | None, jitter_seed: int) -> np.ndarray:
    """Unit 512-d vector. Each identity has a fixed base direction with a
    set cosine similarity to the subject; jitter_seed adds per-photo noise
    so similarities spread the way real embeddings do."""
    base_rng = np.random.default_rng(1234)
    subject = base_rng.normal(size=512)
    subject /= np.linalg.norm(subject)
    ortho = base_rng.normal(size=512)
    ortho -= (ortho @ subject) * subject
    ortho /= np.linalg.norm(ortho)
    sim = _SIMILARITY.get(identity or "", 0.0)
    if identity is None:
        vec = np.random.default_rng(jitter_seed).normal(size=512)
    else:
        vec = sim * subject + math.sqrt(1 - sim**2) * ortho
    noise = np.random.default_rng(jitter_seed).normal(size=512)
    vec = vec / np.linalg.norm(vec) + 0.25 * noise / np.linalg.norm(noise)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


class SyntheticDetector:
    """Same interface as detection.FaceDetector."""

    def analyze(self, rgb: np.ndarray, content_hash: str) -> list[Face]:
        faces = []
        seed = int(content_hash[:8], 16) if _is_hex(content_hash[:8]) else 0
        for i, m in enumerate(_find_faces(rgb)):
            yaw, pitch, roll = pose(m)
            kps = np.stack([m.le, m.re, m.nose, m.corners[0], m.corners[1]]).astype(np.float32)
            faces.append(
                Face(
                    content_hash=content_hash,
                    face_id=i,
                    bbox=m.bbox,
                    kps=kps,
                    embedding=identity_embedding(m.identity, seed + i),
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    det_score=1.0,
                )
            )
        return faces

    def embed(self, rgb: np.ndarray) -> np.ndarray | None:
        faces = self.analyze(rgb, "00000000")
        if not faces:
            return None
        faces.sort(key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        return faces[-1].embedding


class SyntheticMesh:
    """Same interface as landmarks.FaceMesh: dense landmarks in crop pixels,
    with only the indices landmarks.py reads filled in."""

    def landmarks(self, rgb_crop: np.ndarray) -> np.ndarray | None:
        from . import landmarks as L

        m = _markers_in(np.ascontiguousarray(rgb_crop))
        if m is None:
            return None
        mid, u, v, iod = _frame(m)
        d = np.tile(mid, (478, 1)).astype(np.float32)
        half_w, half_h = m.eye_w / 2, m.eye_h / 2
        for eye, idx in ((m.le, L.LEFT_EYE), (m.re, L.RIGHT_EYE)):
            d[idx["outer"]] = eye - half_w * u
            d[idx["inner"]] = eye + half_w * u
            d[idx["top"]] = eye - half_h * v
            d[idx["bottom"]] = eye + half_h * v
        d[L.MOUTH["left"]] = m.corners[0]
        d[L.MOUTH["right"]] = m.corners[1]
        d[L.MOUTH["top"]] = m.mouth - 0.02 * iod * v
        d[L.MOUTH["bottom"]] = m.mouth + 0.02 * iod * v
        d[L.NASION] = mid - 0.15 * iod * v
        return d


def _is_hex(s: str) -> bool:
    try:
        int(s, 16)
    except ValueError:
        return False
    return True

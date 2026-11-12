"""Apple Vision detector backend for macOS, for running without the
InsightFace/MediaPipe stack. Ported from face-lapse tools/detect.m (commit
50de507). The Objective-C helper (vision_detect.m) is compiled with clang
on first use into the cache directory.

Vision returns landmarks and head pose but no identity embedding, so faces
from this backend carry an empty embedding and `facelapse score` must run
with --single-person: the identity gate is skipped and the largest face in
each image is taken as the subject.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .records import Face

SOURCE = Path(__file__).with_name("vision_detect.m")
FRAMEWORKS = ["Foundation", "Vision", "ImageIO", "CoreGraphics"]


def compile_helper(out: Path) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("the vision backend needs macOS (Apple Vision)")
    clang = shutil.which("clang")
    if clang is None:
        raise RuntimeError("clang not found; install the Xcode Command Line Tools")
    if out.exists() and out.stat().st_mtime >= SOURCE.stat().st_mtime:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [clang, "-fobjc-arc", "-O2", str(SOURCE), "-o", str(out)]
    for fw in FRAMEWORKS:
        cmd += ["-framework", fw]
    subprocess.run(cmd, check=True)
    return out


def _centroid(points: list[list[float]]) -> np.ndarray | None:
    return np.mean(np.asarray(points, np.float64), axis=0) if points else None


def faces_from_vision(record: dict[str, Any], content_hash: str) -> list[Face]:
    """Convert one helper output line into Face records: radians to degrees,
    eyes ordered by image x, and the five keypoints built from the pupils
    (eye centroids as fallback), the nose centroid and the outer-lip
    extremes."""
    faces: list[Face] = []
    for d in record.get("faces", []):
        eyes = [
            _centroid(d.get("left_pupil") or d.get("left_eye") or []),
            _centroid(d.get("right_pupil") or d.get("right_eye") or []),
        ]
        nose = _centroid(d.get("nose") or [])
        lips = np.asarray(d.get("outer_lips") or [], np.float64)
        if eyes[0] is None or eyes[1] is None or nose is None or len(lips) < 2:
            continue
        le, re = sorted(eyes, key=lambda p: float(p[0]))
        ml, mr = lips[np.argmin(lips[:, 0])], lips[np.argmax(lips[:, 0])]
        faces.append(
            Face(
                content_hash=content_hash,
                face_id=len(faces),
                bbox=np.asarray(d["bbox"], np.float32),
                kps=np.stack([le, re, nose, ml, mr]).astype(np.float32),
                embedding=np.zeros(0, np.float32),
                yaw=_degrees(d.get("yaw")),
                pitch=_degrees(d.get("pitch")),
                roll=_degrees(d.get("roll")),
                det_score=float(d.get("confidence", 0.0)),
            )
        )
    return faces


def _degrees(v: float | None) -> float:
    return float("nan") if v is None else math.degrees(float(v))


class VisionDetector:
    """Same interface as detection.FaceDetector, minus identity."""

    def __init__(self, helper: Path):
        self.proc = subprocess.Popen(
            [str(compile_helper(helper))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.tmp = tempfile.TemporaryDirectory()

    def analyze(self, rgb: np.ndarray, content_hash: str) -> list[Face]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        path = Path(self.tmp.name) / "frame.png"
        Image.fromarray(rgb).save(path)
        self.proc.stdin.write(f"{path}\n")
        self.proc.stdin.flush()
        record = json.loads(self.proc.stdout.readline())
        if "error" in record:
            raise RuntimeError(f"vision helper: {record['error']}")
        return faces_from_vision(record, content_hash)

    def embed(self, rgb: np.ndarray) -> np.ndarray | None:
        raise RuntimeError("the vision backend has no identity embedding; use --single-person")

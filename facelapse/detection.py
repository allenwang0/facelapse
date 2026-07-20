"""Face detection + landmarks + ArcFace embedding + pose in one pass, via
InsightFace buffalo_l. This is the single expensive inference; its output is
cached and everything downstream reads the cache.

CANNOT be exercised in every environment (weights download on first use).
Two library-specific facts are marked # VERIFY because getting them wrong
corrupts alignment and pose filtering silently:

  1. kps order. InsightFace SCRFD returns 5 keypoints. The conventional
     order is [left_eye, right_eye, nose, mouth_left, mouth_right] in the
     IMAGE frame (subject's right eye appears on image-left). Alignment and
     interocular distance depend on indices 0 and 1 being the two eyes.

  2. face.pose axis order. InsightFace exposes an estimated head pose. The
     ordering of the triple (commonly (pitch, yaw, roll)) is not guaranteed
     stable across versions. Confirm on a known left-turned test face that
     the value you treat as yaw actually moves with left/right rotation.
"""
from __future__ import annotations

import numpy as np

from .records import Face


class FaceDetector:
    def __init__(self, det_size: int = 640, ctx_id: int = 0):
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "insightface not installed. `pip install -r requirements-ml.txt`. "
                "The geometry/data pipeline runs without it; detection does not."
            ) from e
        self.app = FaceAnalysis(name="buffalo_l")
        # ctx_id >= 0 selects GPU; -1 forces CPU.
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def analyze(self, rgb: np.ndarray, content_hash: str) -> list[Face]:
        # InsightFace expects BGR.
        bgr = rgb[:, :, ::-1]
        dets = self.app.get(bgr)
        faces: list[Face] = []
        for i, d in enumerate(dets):
            kps = np.asarray(d.kps, dtype=np.float32)            # (5,2) # VERIFY order
            emb = np.asarray(d.normed_embedding, dtype=np.float32)  # L2-normalized
            yaw, pitch, roll = _pose_triple(d)                    # degrees # VERIFY
            faces.append(Face(
                content_hash=content_hash, face_id=i,
                bbox=np.asarray(d.bbox, dtype=np.float32),
                kps=kps, embedding=emb,
                yaw=float(yaw), pitch=float(pitch), roll=float(roll),
                det_score=float(d.det_score),
            ))
        return faces

    def embed(self, rgb: np.ndarray) -> np.ndarray | None:
        """Single best-face embedding, for seeding references."""
        faces = self.analyze(rgb, content_hash="__seed__")
        if not faces:
            return None
        # pick the largest face as the seed subject
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                   reverse=True)
        return faces[0].embedding


def _pose_triple(det) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) in degrees.

    # VERIFY: insightface commonly exposes det.pose as (pitch, yaw, roll).
    We remap to (yaw, pitch, roll). If pose is absent, fall back to NaN so
    the pose filter can decide policy rather than silently passing.
    """
    pose = getattr(det, "pose", None)
    if pose is None:
        return float("nan"), float("nan"), float("nan")
    pose = np.asarray(pose, dtype=np.float32).ravel()
    if pose.shape[0] < 3:
        return float("nan"), float("nan"), float("nan")
    pitch, yaw, roll = pose[0], pose[1], pose[2]   # VERIFY this ordering
    return float(yaw), float(pitch), float(roll)

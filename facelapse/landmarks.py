"""Dense 468-point landmarks via MediaPipe Face Mesh, run on an accepted
face crop (not the whole image, that's wasteful). Feeds the geometric
neutral backstops (EAR/MAR) and the optional nasion alignment point.

# VERIFY: MediaPipe landmark indices below are the widely-used canonical
# ones, but confirm against your installed mediapipe version by plotting
# them on one of your own faces before trusting EAR/MAR thresholds.
"""
from __future__ import annotations

import numpy as np

# canonical MediaPipe FaceMesh indices (VERIFY on your version)
LEFT_EYE = dict(outer=33, inner=133, top=159, bottom=145)
RIGHT_EYE = dict(outer=362, inner=263, top=386, bottom=374)
MOUTH = dict(left=61, right=291, top=13, bottom=14)
NASION = 168   # nose-bridge point between the eyes; stable across expression


class FaceMesh:
    def __init__(self):
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mediapipe not installed. `pip install -r requirements-ml.txt`."
            ) from e
        self._mp = mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        )

    def landmarks(self, rgb_crop: np.ndarray) -> np.ndarray | None:
        """Return (468,2) or (478,2) landmarks in PIXEL coords of the crop,
        or None if no face found in the crop."""
        h, w = rgb_crop.shape[:2]
        res = self.mesh.process(rgb_crop)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        return pts


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(dense: np.ndarray) -> float:
    """Mean EAR across both eyes. Low => closed/blink."""
    def ear(e):
        vertical = _dist(dense[e["top"]], dense[e["bottom"]])
        horizontal = _dist(dense[e["outer"]], dense[e["inner"]])
        return vertical / horizontal if horizontal > 1e-6 else 0.0
    return 0.5 * (ear(LEFT_EYE) + ear(RIGHT_EYE))


def mouth_aspect_ratio(dense: np.ndarray) -> float:
    """Vertical lip gap over mouth width. High => open mouth / mid-speech."""
    vertical = _dist(dense[MOUTH["top"]], dense[MOUTH["bottom"]])
    horizontal = _dist(dense[MOUTH["left"]], dense[MOUTH["right"]])
    return vertical / horizontal if horizontal > 1e-6 else 0.0


def nasion_point(dense: np.ndarray) -> np.ndarray:
    return dense[NASION].copy()

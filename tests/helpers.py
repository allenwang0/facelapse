"""Shared builders for synthetic test records."""

from __future__ import annotations

from typing import Any

import numpy as np

from facelapse.records import Face


def make_face(**kw: Any) -> Face:
    base: dict[str, Any] = dict(
        content_hash="h",
        face_id=0,
        bbox=np.array([0, 0, 100, 100], np.float32),
        kps=np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], np.float32),
        embedding=np.ones(512, np.float32) / np.sqrt(512),
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        det_score=0.9,
    )
    base.update(kw)
    return Face(**base)

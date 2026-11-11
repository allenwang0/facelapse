"""Temporal smoothing must remove landmark jitter without moving any frame's
eyes off the canonical target. The transforms map each photo's own pixel
coordinates to the canvas, so averaging them across photos is meaningless;
smoothing has to happen in canvas space."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from facelapse import align as align_mod
from facelapse import smoothing
from facelapse.config import Config
from tests.helpers import make_face

CFG = Config()
TARGET = (CFG.left_eye_xy, CFG.right_eye_xy)


def _face_at(x: float, y: float, iod: float, angle_deg: float, dense=None):
    a = math.radians(angle_deg)
    le = np.array([x, y])
    re = le + iod * np.array([math.cos(a), math.sin(a)])
    kps = np.array([le, re, (le + re) / 2 + [0, iod / 2], le + [0, iod], re + [0, iod]])
    return make_face(kps=kps.astype(np.float32), dense=dense)


def _canvas_eyes(f, p) -> np.ndarray:
    M = p.matrix()
    return f.kps[:2].astype(np.float64) @ M[:, :2].T + M[:, 2]


def _smooth(faces, window=3):
    params = [align_mod.estimate(f, CFG) for f in faces]
    eyes = [f.kps[:2] for f in faces]
    return smoothing.smooth_params(params, eyes, TARGET, window)


def test_frames_from_far_apart_source_positions_stay_on_target():
    # The scenario from the bug report: left eyes at x=100, 2000, 100 in
    # 4000x3000 sources. Averaging tx moved frame 2's eye ~1,267 px.
    faces = [_face_at(x, 1500, 200, 0) for x in (100, 2000, 100)]
    for f, p in zip(faces, _smooth(faces)):
        assert np.allclose(_canvas_eyes(f, p), np.array(TARGET), atol=1.0)


frame = st.tuples(
    st.floats(0, 4000),
    st.floats(0, 3000),
    st.floats(30, 600),
    st.floats(-60, 60),
)


@settings(max_examples=150, deadline=None)
@given(st.lists(frame, min_size=3, max_size=12), st.sampled_from([3, 5, 7]))
def test_eyes_land_on_target_for_any_source_geometry(frames, window):
    faces = [_face_at(*fr) for fr in frames]
    for f, p in zip(faces, _smooth(faces, window)):
        assert np.allclose(_canvas_eyes(f, p), np.array(TARGET), atol=1.0)


def _dense_with_nasion(f, nasion_xy) -> np.ndarray:
    d = np.zeros((478, 2), np.float32)
    d[168] = nasion_xy
    return d


def _jitter(faces, params) -> float:
    mids = np.array([_canvas_eyes(f, p).mean(axis=0) for f, p in zip(faces, params)])
    return float(np.mean(np.linalg.norm(np.diff(mids, axis=0), axis=1)))


def test_smoothing_reduces_nasion_noise_jitter():
    # WHY: the nasion term exists to stabilize the fit, but a noisy nasion
    # makes the eyes wander frame to frame. Smoothing must damp that.
    rng = np.random.default_rng(0)
    faces = []
    for i in range(60):
        f = _face_at(500 + 7 * i, 800 - 3 * i, 180, 0)
        le, re = f.kps[0], f.kps[1]
        nasion = (le + re) / 2 + [0, -0.15 * 180] + rng.normal(0, 1.5, 2)
        f.dense = _dense_with_nasion(f, nasion)
        faces.append(f)
    raw = [align_mod.estimate(f, CFG) for f in faces]
    smoothed = smoothing.smooth_params(raw, [f.kps[:2] for f in faces], TARGET, 5)
    assert _jitter(faces, raw) > 0.1
    assert _jitter(faces, smoothed) < 0.6 * _jitter(faces, raw)

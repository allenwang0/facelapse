"""The verify checks must catch a detector whose pose axes are swapped or
whose keypoints are out of order. They run here against the synthetic
detector, whose geometry is known, and against deliberately broken copies
of it."""

from __future__ import annotations

import numpy as np

from facelapse import cli, verify
from facelapse.synthetic import FaceSpec, SyntheticDetector, draw_face


def _photo(yaw=12.0, pitch=6.0, roll=5.0) -> np.ndarray:
    img = np.full((600, 800, 3), 100, np.uint8)
    draw_face(img, FaceSpec(400, 260, 90, yaw=yaw, pitch=pitch, roll=roll))
    return img


class SwappedAxes(SyntheticDetector):
    """Reports (pitch, yaw, roll) as (yaw, pitch, roll): the bug VERIFY guards."""

    def analyze(self, rgb, content_hash):
        faces = super().analyze(rgb, content_hash)
        for f in faces:
            f.yaw, f.pitch = f.pitch, f.yaw
        return faces


class SwappedEyes(SyntheticDetector):
    def analyze(self, rgb, content_hash):
        faces = super().analyze(rgb, content_hash)
        for f in faces:
            f.kps = f.kps[[1, 0, 2, 3, 4]]
        return faces


def _by_name(checks):
    return {c.name: c.passed for c in checks}


def test_correct_detector_passes_every_check():
    _, checks = verify.run_checks(SyntheticDetector(), _photo())
    assert _by_name(checks) == {"keypoint order": True, "mirror": True, "rotate": True}


def test_swapped_pose_axes_fail_the_mirror_check():
    _, checks = verify.run_checks(SwappedAxes(), _photo())
    assert _by_name(checks)["mirror"] is False


def test_swapped_eye_keypoints_fail_the_order_check():
    _, checks = verify.run_checks(SwappedEyes(), _photo())
    assert _by_name(checks)["keypoint order"] is False


def test_frontal_face_is_inconclusive_not_passed():
    _, checks = verify.run_checks(SyntheticDetector(), _photo(yaw=0.0))
    assert _by_name(checks)["mirror"] is None


def test_command_writes_verify_png(tmp_path):
    from PIL import Image

    photo = tmp_path / "me.png"
    Image.fromarray(_photo()).save(photo)
    out = tmp_path / "verify.png"
    assert cli.main(["verify", str(photo), "--backend", "synthetic", "--out", str(out)]) == 0
    assert out.exists()

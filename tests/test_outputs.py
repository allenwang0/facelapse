"""Output formats ported from face-lapse: story layout, looping GIF and a
fixed-length video with a one-second end hold."""

from __future__ import annotations

import imageio.v2 as imageio
import pytest
from PIL import Image

from facelapse import cli
from tests.report_fixture import build_demo


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    return root, build_demo(root, synthetic_banner=False)


def _render(demo, *extra):
    root, cfg = demo
    cli.main(["--config", str(cfg), "render", *extra])
    return root / "out"


def test_story_layout_is_1080_by_1920(demo):
    out = _render(demo, "--layout", "story")
    frame = imageio.get_reader(out / "timelapse.mp4").get_data(0)
    assert frame.shape[:2] == (1920, 1080)


def test_square_layout_is_square(demo):
    out = _render(demo, "--layout", "square")
    frame = imageio.get_reader(out / "timelapse.mp4").get_data(0)
    assert frame.shape[0] == frame.shape[1]


def test_seconds_sets_length_with_one_second_hold(demo):
    out = _render(demo, "--seconds", "7")
    reader = imageio.get_reader(out / "timelapse.mp4")
    meta = reader.get_meta_data()
    n = reader.count_frames()
    assert abs(n / meta["fps"] - 7.0) < 0.25
    last, before_hold = reader.get_data(n - 1), reader.get_data(n - 2)
    assert abs(last.astype(int) - before_hold.astype(int)).mean() < 2


def test_gif_loops_and_holds_the_last_frame(demo):
    out = _render(demo, "--gif")
    gif = Image.open(out / "timelapse.gif")
    assert gif.info.get("loop") == 0
    assert gif.n_frames > 10
    gif.seek(gif.n_frames - 1)
    assert gif.info["duration"] > 1000

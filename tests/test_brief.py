"""The project brief and README say what the committed results say.

The portfolio links readers to this brief, so its numbers are recomputed here
from site/card.json and results/bench_render.json, independently of
scripts/build_brief.py: a wrong series, a rounding slip or a stale page fails
here instead of reaching Pages. Unmeasured claims must stay off both pages.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CARD = json.loads((ROOT / "site" / "card.json").read_text())
BENCH = json.loads((ROOT / "results" / "bench_render.json").read_text())


def _build(tmp_path: Path) -> str:
    out = tmp_path / "index.html"
    subprocess.run(
        [sys.executable, "scripts/build_brief.py", "--out", str(out)], cwd=ROOT, check=True
    )
    return out.read_text()


def _text(page: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", page))


def _series(name: str) -> list[float]:
    return next(s for s in CARD["chart"]["series"] if s["name"] == name)["points"]


def _jitter(points: list[list[float]]) -> float:
    vals = [p[1] for p in points]
    return sum(abs(b - a) for a, b in zip(vals, vals[1:])) / (len(vals) - 1)


def test_jitter_figure_matches_the_committed_trace(tmp_path: Path) -> None:
    # The one figure's headline must be the card's own trace, raw vs smoothed.
    raw, smooth = _jitter(_series("Raw fit")), _jitter(_series("Smoothed"))
    assert smooth < raw, "the figure claims smoothing reduces jitter"
    text = _text(_build(tmp_path))
    assert f"from {raw:.2f} to {smooth:.2f}\u00a0px" in text
    assert smooth == pytest.approx(CARD["metric"]["value"], abs=1e-3)


def test_memory_claim_is_the_committed_measurement(tmp_path: Path) -> None:
    mib = BENCH["peak_rss_bytes"] / 2**20
    text = _text(_build(tmp_path))
    assert f"{mib:,.0f}\u00a0MiB" in text
    assert f"{BENCH['frames']:,} synthetic" in text
    readme = (ROOT / "README.md").read_text()
    assert f"{mib:,.0f}\u00a0MiB peak memory for {BENCH['frames']:,} synthetic" in readme


def test_synthetic_data_is_labelled_in_the_hero_and_image(tmp_path: Path) -> None:
    page = _build(tmp_path)
    hero = page.split('<section class="section')[0]
    assert "synthetic" in _text(hero).lower()
    alt = re.search(r'<img src="timelapse.gif"[^>]*alt="([^"]+)"', page).group(1)
    assert "Synthetic" in alt or "synthetic" in alt


def test_unmeasured_claims_stay_out(tmp_path: Path) -> None:
    # The before/after memory pair and "takes seconds" have no committed measurement.
    readme = (ROOT / "README.md").read_text()
    for name, text in (("brief", _text(_build(tmp_path))), ("README", readme)):
        for phrase in ("4,466", "382 MB", "takes seconds"):
            assert phrase not in text, f"{name} contains unmeasured claim {phrase!r}"


def test_committed_brief_and_readme_are_current(tmp_path: Path) -> None:
    assert _build(tmp_path) == (ROOT / "docs" / "index.html").read_text(), (
        "docs/index.html is stale: run python3 scripts/build_brief.py --readme README.md"
    )
    assert "allenwang0.github.io/facelapse" in (ROOT / "README.md").read_text()

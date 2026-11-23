"""The report must show the numbers the pipeline logged, and its JS
threshold rule must classify images exactly as identity.resolve_image does,
or the calibration explorer would recommend thresholds that behave
differently in the real run."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from facelapse import cli
from facelapse.config import Config
from facelapse.identity import ReferenceChain, resolve_image
from facelapse.report import build_data
from tests.helpers import embedding_with_similarity, make_face, unit
from tests.report_fixture import build_demo

TEMPLATE = Path(__file__).resolve().parents[1] / "facelapse" / "report_template.html"


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    return root, build_demo(root)


def _embedded(root: Path) -> dict:
    page = (root / "out" / "report.html").read_text()
    return json.loads(re.search(r"const D = (.*?);\n", page).group(1))


def test_funnel_counts_equal_the_logged_counts(tmp_path, capsys):
    capsys.readouterr()
    build_demo(tmp_path)
    logged = {
        m.group(1).strip(): (int(m.group(2)), int(m.group(3)))
        for m in re.finditer(r"\[funnel\] (.{18}) kept\s+(\d+) / (\d+)", capsys.readouterr().out)
    }
    embedded = {r["stage"]: (r["kept"], r["total"]) for r in _embedded(tmp_path)["funnel"]}
    assert embedded == logged
    assert len(embedded) == 11


def test_every_view_has_data_and_a_takeaway(demo):
    data = _embedded(demo[0])
    assert data["points"] and data["coverage"] and data["alignment"] and data["provenance"]
    for key in ("funnel_note", "coverage_note", "alignment_note", "provenance_note"):
        assert data[key], key
    assert data["banner"].startswith("Synthetic data")


def test_page_text_is_filled_in_without_javascript(demo):
    """Headline numbers render in the HTML itself (first paint, no JS) and no
    template slot is left behind."""
    page = (demo[0] / "out" / "report.html").read_text()
    data = _embedded(demo[0])
    assert "{{" not in page and "/*KIT_" not in page
    frames = len(data["alignment"])
    h1 = re.search(r'<h1 id="h1">(.*?)</h1>', page).group(1)
    assert h1.startswith(f"{frames} frames from {data['funnel'][0]['total']} photos")
    j = data["jitter"]
    assert f"from {j['raw']:.2f}\u202fpx to {j['smooth']:.2f}\u202fpx" in page


def test_inlined_kit_cannot_close_its_own_script_tag(demo):
    """The kit's comments quote <script> tags; inlined verbatim they would end
    the page's script early and every chart would fail to draw."""
    page = (demo[0] / "out" / "report.html").read_text()
    bodies = re.findall(r"<script>(.*?)</script>", page, re.S)
    # one element holds all of report.js and charts.js, start to end
    assert any("RK.initBar" in b and "C.table = function" in b for b in bodies)
    assert any(b.lstrip().startswith("const D = ") for b in bodies)


def test_page_and_card_quote_the_same_jitter(demo):
    """site/card.json (the portfolio card) and the page must not disagree:
    both come from the same manifest and the same jitter formula."""
    from scripts.make_card import build_card

    manifest = json.loads((demo[0] / "out" / "manifest.json").read_text())
    card = build_card(manifest, "a" * 40, 60)
    j = _embedded(demo[0])["jitter"]
    assert card["metric"]["value"] == pytest.approx(j["smooth"], abs=1e-3)
    assert f"from {j['raw']:.2f} px to {j['smooth']:.2f} px" in card["takeaway"]


def test_drift_example_names_the_neighbour_the_smoother_borrowed_from():
    """A frame a 2-point fit puts on target (raw 0) ends up off target after
    smoothing because it takes a 1/window share of a neighbour's offset. The
    page explains the smoothed line with exactly that arithmetic, so it must
    only claim it when the numbers match."""
    from facelapse.report import _drift_example

    rows = [{"raw": r, "smooth": s} for r, s in [(0, 0), (0, 2.55), (7.65, 2.55), (0, 0)]]
    ex = _drift_example(rows, 3)
    assert ex["frame"] == 2 and ex["neighbour"] == 3 and ex["is_share"]
    rows[1]["smooth"] = 1.0
    assert _drift_example(rows, 3)["is_share"] is False
    steady = [{"raw": 1.0, "smooth": 0.5}, {"raw": 1.0, "smooth": 0.5}]
    assert _drift_example(steady, 3) is None


def test_no_cache_is_an_error_and_writes_nothing(tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"cache_db": str(tmp_path / "none.db"), "out_dir": str(tmp_path / "out")})
    )
    assert cli.main(["--config", str(cfg), "report"]) == 1
    assert not (tmp_path / "out").exists()
    assert "no cache" in capsys.readouterr().err


def test_empty_cache_reports_missing_seeds_and_selection(tmp_path):
    cfg = Config()
    cfg.cache_db = str(tmp_path / "c.db")
    cfg.out_dir = str(tmp_path / "out")
    from facelapse.cache import Cache

    Cache(cfg.cache_db).close()
    data = build_data(cfg)
    assert data["has_seeds"] is False and data["has_selected"] is False
    assert data["alignment"] is None


def _random_cases(n: int):
    rng = np.random.default_rng(11)
    seed = unit(rng.normal(size=512))
    chain = ReferenceChain([("s", None, seed)])
    cases = []
    for i in range(n):
        k = int(rng.integers(1, 4))
        faces = [
            make_face(
                face_id=j,
                embedding=embedding_with_similarity(seed, float(rng.uniform(-0.2, 1)), i * 10 + j),
            )
            for j in range(k)
        ]
        t_abs, margin = float(rng.uniform(0, 1)), float(rng.uniform(0, 0.4))
        outcome, _ = resolve_image(faces, None, chain, t_abs, margin)
        sims = sorted((f.id_sim for f in faces), reverse=True)
        gap = sims[0] - (sims[1] if k > 1 else -1.0)
        cases.append({"top": sims[0], "gap": gap, "t": t_abs, "m": margin, "want": outcome.value})
    return cases


def test_js_outcome_rule_matches_resolve_image():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; the JS rule check needs it")
    src = TEMPLATE.read_text()
    rule = src[src.index("/*RULE*/") : src.index("/*END RULE*/")]
    cases = _random_cases(1000)
    script = (
        rule
        + f"\nconst cases = {json.dumps(cases)};\n"
        + ("console.log(JSON.stringify(cases.map(c => outcomeOf(c.top, c.gap, c.t, c.m))));")
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    got = json.loads(out.stdout)
    assert got == [c["want"] for c in cases]
    assert {"none", "ambiguous", "confident"} <= set(got)

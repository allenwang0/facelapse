"""Load the demo report in a real browser at phone and desktop widths, in
light and dark, and check it renders, does not scroll sideways and reacts to
the calibration sliders. Screenshots land in artifacts/screenshots for CI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

from tests.report_fixture import build_demo  # noqa: E402

SHOTS = Path(__file__).resolve().parents[2] / "artifacts" / "screenshots"


@pytest.fixture(scope="module")
def report_url(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    build_demo(root)
    return (root / "out" / "report.html").as_uri()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            # PLAYWRIGHT_CHANNEL=chrome uses an installed Chrome instead of
            # the bundled headless shell
            b = p.chromium.launch(channel=os.environ.get("PLAYWRIGHT_CHANNEL") or None)
        except sync_api.Error as e:
            pytest.skip(f"no chromium for playwright: {e}")
        yield b
        b.close()


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.parametrize("size", [(390, 844), (1280, 800)], ids=["phone", "desktop"])
def test_report_renders(browser, report_url, scheme, size):
    page = browser.new_page(viewport={"width": size[0], "height": size[1]}, color_scheme=scheme)
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(report_url)
    assert errors == []
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    for view in ("funnel", "scatter", "coverage", "align", "prov"):
        assert page.locator(f"#{view} svg").count() == 1, view
    assert "Synthetic data" in page.inner_text("#banner")

    before = page.inner_text("#calib-counts")
    page.locator("#m").fill("0.9")
    page.locator("#m").dispatch_event("input")
    assert page.inner_text("#calib-counts") != before

    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / f"report-{size[0]}-{scheme}.png"), full_page=True)
    page.close()

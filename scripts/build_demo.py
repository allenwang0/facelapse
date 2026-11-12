"""Build the synthetic demo site: run the pipeline on the generated archive,
render the GIF, write the report, and copy what the README and GitHub Pages
show into SITE_DIR (index.html, timelapse.gif, report.png).

    python scripts/build_demo.py SITE_DIR [--screenshot]

--screenshot needs Playwright with Chromium (pip install -e .[e2e]).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelapse import cli  # noqa: E402
from tests.report_fixture import build_demo  # noqa: E402


def screenshot(page_path: Path, out: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get("PLAYWRIGHT_CHANNEL") or None)
        page = browser.new_page(viewport={"width": 1100, "height": 900}, color_scheme="light")
        page.goto(page_path.as_uri())
        page.screenshot(path=str(out), full_page=True)
        browser.close()


def main(site: Path, shot: bool) -> None:
    work = Path(tempfile.mkdtemp())
    cfg = build_demo(work)
    cli.main(["--config", str(cfg), "render", "--gif", "--seconds", "8"])
    cli.main(["--config", str(cfg), "report", "--synthetic"])
    site.mkdir(parents=True, exist_ok=True)
    shutil.copy(work / "out" / "report.html", site / "index.html")
    shutil.copy(work / "out" / "timelapse.gif", site / "timelapse.gif")
    if shot:
        screenshot(site / "index.html", site / "report.png")
    print(f"[demo] {site}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), "--screenshot" in sys.argv[2:])

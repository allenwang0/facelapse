"""Write site/card.json (portfolio card, schema_version 1) from a real run of
the pipeline on the synthetic demo archive.

    python scripts/make_card.py

The chart is the alignment stability trace: canvas eye-midpoint offset per
frame, raw fit vs temporally smoothed. The metric is the mean frame-to-frame
jitter of the smoothed trace. Run it from a clean checkout: the card records
the commit that produced it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.report_fixture import build_demo  # noqa: E402

COMMAND = "python scripts/make_card.py"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def jitter(values: list[float]) -> float:
    return float(np.mean(np.abs(np.diff(values))))


def build_card(manifest: dict, sha: str, n_photos: int) -> dict:
    frames = manifest["frames"]
    raw = [r["eye_dev_raw_px"] for r in frames]
    smooth = [r["eye_dev_smoothed_px"] for r in frames]
    j_raw, j_smooth = jitter(raw), jitter(smooth)
    years = sorted({r["date"][:4] for r in frames})
    return {
        "schema_version": 1,
        "repo": "facelapse",
        "sha": sha,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": COMMAND,
        "data_note": (
            f"Synthetic: {n_photos} generated photos of drawn faces with stubbed detection, "
            f"{years[0]}-{years[-1]}, fixture seed 2026; {len(frames)} frames selected."
        ),
        "metric": {
            "label": "Frame-to-frame eye jitter after smoothing",
            "value": round(j_smooth, 3),
            "unit": "px",
            "higher_is_better": False,
        },
        "takeaway": (
            f"Canvas-space smoothing cuts frame-to-frame eye jitter from {j_raw:.2f} px "
            f"to {j_smooth:.2f} px across {len(frames)} frames."
        ),
        "chart": {
            "type": "line",
            "x_label": "Frame",
            "x_unit": "index",
            "x_scale": "linear",
            "y_label": "Eye-midpoint offset from target",
            "y_unit": "px",
            "y_scale": "linear",
            "series": [
                {
                    "name": "Smoothed",
                    "style": "solid",
                    "points": [[i + 1, round(v, 3)] for i, v in enumerate(smooth)],
                },
                {
                    "name": "Raw fit",
                    "style": "dashed",
                    "points": [[i + 1, round(v, 3)] for i, v in enumerate(raw)],
                },
            ],
        },
    }


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=no").strip():
        sys.exit("commit your changes first: card.json records the commit that produced it")
    work = Path(tempfile.mkdtemp())
    build_demo(work, synthetic_banner=False)
    manifest = json.loads((work / "out" / "manifest.json").read_text())
    n_photos = len(list((work / "photos").glob("*.*g")))
    card = build_card(manifest, _git("rev-parse", "HEAD").strip(), n_photos)
    out = ROOT / "site" / "card.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(card, indent=2) + "\n")
    print(f"[card] {out}: {card['takeaway']}")


if __name__ == "__main__":
    main()

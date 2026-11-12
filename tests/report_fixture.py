"""Build the synthetic demo run and its report once, for the report tests,
the Pages workflow and scripts/make_card.py."""

from __future__ import annotations

import json
from pathlib import Path

from facelapse import cli
from tests.fixtures.make_fixtures import make


def build_demo(root: Path, synthetic_banner: bool = True) -> Path:
    """Generate the archive under root, run the pipeline with the synthetic
    backend, write the report. Returns the config path."""
    make(root)
    cfg = root / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "cache_db": str(root / "cache.db"),
                "out_dir": str(root / "out"),
                "bucket_grain": "month",
            }
        )
    )
    seeds = sorted(str(p) for p in (root / "seeds").glob("*.jpg"))
    c = ["--config", str(cfg)]
    cli.main([*c, "seed", "add", *seeds, "--backend", "synthetic"])
    cli.main([*c, "run", str(root / "photos"), "--backend", "synthetic", "--no-au"])
    cli.main([*c, "report", *(["--synthetic"] if synthetic_banner else [])])
    return cfg

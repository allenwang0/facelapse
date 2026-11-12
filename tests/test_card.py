"""site/card.json feeds the portfolio site, which renders it without
checking the numbers again. It must match the schema_version 1 contract and
come from a real run of the code at the recorded commit."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import jsonschema
import pytest

from scripts.make_card import build_card, jitter

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "tests" / "card.schema.json").read_text())
CARD = ROOT / "site" / "card.json"


def _numbers(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numbers(v)


def _check(card: dict) -> None:
    jsonschema.validate(card, SCHEMA)
    assert all(math.isfinite(v) for v in _numbers(card))
    assert len(json.dumps(card)) <= 200_000
    assert card["takeaway"] and len(card["takeaway"]) <= 140


def test_committed_card_is_valid_and_traceable():
    if not CARD.exists():
        pytest.skip("site/card.json not generated yet; run python scripts/make_card.py")
    card = json.loads(CARD.read_text())
    _check(card)
    assert "Synthetic" in card["data_note"]
    known = subprocess.run(
        ["git", "cat-file", "-t", card["sha"]], cwd=ROOT, capture_output=True, text=True
    )
    assert known.stdout.strip() == "commit", "card sha must be a commit in this repo"
    smooth = [p[1] for p in card["chart"]["series"][0]["points"]]
    assert card["metric"]["value"] == pytest.approx(jitter(smooth), abs=1e-3)


def test_build_card_matches_schema_and_rejects_nan():
    frames = [
        {
            "date": f"20{16 + i // 4}-01-01",
            "eye_dev_raw_px": float(i % 3),
            "eye_dev_smoothed_px": 0.5,
        }
        for i in range(12)
    ]
    card = build_card({"frames": frames}, "a" * 40, 60)
    _check(card)
    card["metric"]["value"] = float("nan")
    with pytest.raises(AssertionError):
        _check(card)
    bad = dict(card, sha="abc")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, SCHEMA)

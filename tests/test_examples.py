"""The README points users at the example presets; each one must load."""

from __future__ import annotations

from pathlib import Path

import pytest

from facelapse.config import Config

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.config.json"))


def test_the_three_documented_presets_exist():
    names = {p.name for p in EXAMPLES}
    assert names == {
        "balanced.config.json",
        "conservative.config.json",
        "sparse-archive.config.json",
    }


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_preset_loads_and_validates(path):
    cfg = Config.load(str(path))
    cfg._validate()

"""CLI contract: the README's commands must parse as written."""

from __future__ import annotations

import pytest

from facelapse import cli


def _parsed(monkeypatch, argv):
    seen = {}

    def capture(args):
        seen["args"] = args

    for name in ("cmd_run", "cmd_select", "cmd_score", "cmd_render"):
        monkeypatch.setattr(cli, name, capture)
    cli.main(argv)
    return seen["args"]


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "photos", "--grain", "week"],
        ["--grain", "week", "run", "photos"],
    ],
)
def test_grain_works_before_or_after_the_subcommand(monkeypatch, argv):
    assert _parsed(monkeypatch, argv).grain == "week"


def test_config_works_after_the_subcommand(monkeypatch):
    assert _parsed(monkeypatch, ["select", "--config", "x.json"]).config == "x.json"

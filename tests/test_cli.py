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


def test_run_accepts_rejects(monkeypatch):
    assert _parsed(monkeypatch, ["run", "photos", "--rejects", "r.json"]).rejects == "r.json"


def test_run_warns_when_au_scorer_is_missing(monkeypatch, capsys, tmp_path):
    # WHY: run and score must report the same degraded neutral scoring;
    # run used to swallow the error silently.
    from facelapse import neutral, pipeline

    def no_feat():
        raise RuntimeError("py-feat not installed")

    monkeypatch.setattr(neutral, "AUScorer", no_feat)
    for stage in ("analyze", "score", "select"):
        monkeypatch.setattr(pipeline, stage, lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "render_final", lambda cfg, rejects=None: {"video": "v"})
    cli.main(["run", str(tmp_path)])
    assert "[warn] py-feat not installed" in capsys.readouterr().out

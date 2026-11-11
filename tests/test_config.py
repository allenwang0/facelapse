"""Config validation and cache integrity, the two guarantees the old
string-grepping test_fixes.py claimed to check."""

from __future__ import annotations

import pytest

from facelapse.cache import Cache
from facelapse.config import Config


@pytest.mark.parametrize(
    "field, value",
    [
        ("id_t_abs", 1.5),
        ("id_margin", -0.1),
        ("max_abs_yaw_deg", 120.0),
        ("bucket_grain", "year"),
        ("photometric_mode", "sepia"),
        ("fps", 0.0),
        ("score_weight_frontality", 5.0),
    ],
)
def test_validate_rejects_out_of_range_values(field: str, value: object) -> None:
    # WHY: a bad threshold silently starves the funnel; it must fail at load time.
    cfg = Config()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=field.split("_")[0]):
        cfg._validate()


def test_defaults_validate() -> None:
    Config()._validate()


def test_cache_enforces_foreign_keys(tmp_path) -> None:
    # WHY: a face row without its image row is unrenderable; SQLite only
    # enforces the constraint when the pragma is on for the connection.
    cache = Cache(str(tmp_path / "c.db"))
    assert cache.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    cache.close()

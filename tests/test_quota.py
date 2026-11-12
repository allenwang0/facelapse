"""select --max-frames spreads a frame budget across years (ported from
face-lapse). The quota must hit the budget exactly; the original overshot."""

from __future__ import annotations

from datetime import datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st

from facelapse.bucketing import thin_by_year, year_quotas
from tests.helpers import make_face


def test_face_lapse_overshoot_case():
    q = year_quotas({"2021": 47, "2022": 67, "2023": 67}, 180)
    assert sum(q.values()) == 180
    assert q["2021"] == 47


@given(
    st.dictionaries(
        st.sampled_from([str(y) for y in range(2000, 2030)]), st.integers(1, 400), min_size=1
    ),
    st.integers(0, 2000),
)
def test_quota_properties(sizes, n):
    q = year_quotas(sizes, n)
    assert sum(q.values()) == min(n, sum(sizes.values()))
    assert all(0 <= q[y] <= sizes[y] for y in sizes)
    fair = n / len(sizes)
    assert all(q[y] == sizes[y] for y in sizes if sizes[y] < fair)


def test_thin_by_year_keeps_sparse_years_and_spreads_dense_ones():
    ts, faces = {}, []
    start = datetime(2020, 1, 1)
    for i in range(50):  # dense 2020
        h = f"a{i}"
        ts[h] = start + timedelta(days=7 * i)
        faces.append(make_face(content_hash=h))
    for i in range(3):  # sparse 2021
        h = f"b{i}"
        ts[h] = datetime(2021, 3 + 3 * i, 1)
        faces.append(make_face(content_hash=h))
    kept = thin_by_year(faces, ts, 13)
    assert len(kept) == 13
    assert sum(1 for f in kept if f.content_hash.startswith("b")) == 3
    dates = [ts[f.content_hash] for f in kept if f.content_hash.startswith("a")]
    assert dates[0] == start and dates[-1] == start + timedelta(days=7 * 49)


def test_select_max_frames_caps_the_selection(tmp_path):
    from facelapse import pipeline
    from facelapse.cache import Cache
    from tests.report_fixture import build_demo

    cfg_path = build_demo(tmp_path, synthetic_banner=False)
    from facelapse.config import Config

    cfg = Config.load(str(cfg_path))
    kept = pipeline.select(cfg, max_frames=10)
    cache = Cache(cfg.cache_db)
    selected = cache.all_faces(only_selected=True)
    cache.close()
    assert len(kept) == len(selected) == 10

"""`facelapse report`: one self-contained HTML page (inline SVG and vanilla
JS, no CDN) that explains a run. It reads cache.db and out/manifest.json,
embeds the numbers as JSON, and works offline on private data.

Views: the funnel, an identity calibration explorer, timeline coverage,
alignment stability and date provenance. Each has a one-line takeaway
computed here from the data.
"""

from __future__ import annotations

import base64
import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import ingest
from .bucketing import bucket_key
from .cache import Cache
from .config import Config
from .pipeline import STAGES
from .records import TsSource

THUMB_W = 96
SYNTHETIC_BANNER = "Synthetic data: faces are drawn shapes and detection is stubbed."


def _bucket_start(key: str, grain: str) -> date:
    if grain == "day":
        return date.fromisoformat(key)
    if grain == "month":
        return date.fromisoformat(key + "-01")
    year, week = key.split("-W")
    return date.fromisocalendar(int(year), int(week), 1)


def _bucket_days(grain: str) -> int:
    return {"day": 1, "week": 7, "month": 30}[grain]


def _thumb(path: str) -> str | None:
    try:
        rgb, _ = ingest.thumbnail(Path(path), THUMB_W)
    except ingest.DECODE_ERRORS:
        return None
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _funnel(cache: Cache) -> tuple[list[dict[str, Any]], str]:
    stored = cache.stages()
    rows = []
    for stage in STAGES:
        if stage not in stored:
            continue
        kept, total, reasons = stored[stage]
        top = sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
        rows.append({"stage": stage, "kept": kept, "total": total, "reasons": top})
    cuts = [(r["total"] - r["kept"]) / r["total"] for r in rows if r["total"]]
    if not cuts or max(cuts) == 0:
        return rows, "No stage has removed anything yet."
    worst = rows[
        int(np.argmax([(r["total"] - r["kept"]) / r["total"] if r["total"] else 0 for r in rows]))
    ]
    pct = 100 * (worst["total"] - worst["kept"]) / worst["total"]
    return (
        rows,
        f"The {worst['stage']} stage removed {pct:.0f}% of what reached it, the largest cut.",
    )


def _points(cache: Cache, thumbs: bool) -> list[dict[str, Any]]:
    by_img: dict[str, list[Any]] = {}
    for f in cache.all_faces():
        if f.id_sim is not None:
            by_img.setdefault(f.content_hash, []).append(f)
    outcomes = dict(cache.db.execute("SELECT content_hash, id_outcome FROM images").fetchall())
    points = []
    for ch, faces in by_img.items():
        sims = sorted((f.id_sim for f in faces), reverse=True)
        runner = sims[1] if len(sims) > 1 else -1.0
        path = cache.image_path(ch)
        points.append(
            {
                "top": round(sims[0], 4),
                "gap": round(sims[0] - runner, 4),
                "faces": len(faces),
                "stored": outcomes.get(ch),
                "file": Path(path).name,
                "thumb": _thumb(path) if thumbs else None,
            }
        )
    return points


def _coverage(cache: Cache, grain: str) -> tuple[list[dict[str, Any]], str]:
    counts: dict[str, int] = {}
    for ts_iso, src in cache.db.execute(
        "SELECT i.ts_iso, i.ts_source FROM images i WHERE i.decoded_ok=1 AND EXISTS "
        "(SELECT 1 FROM faces f WHERE f.content_hash=i.content_hash)"
    ):
        if ts_iso and TsSource(src) <= TsSource.FILENAME:
            key = bucket_key(datetime.fromisoformat(ts_iso), grain)
            counts[key] = counts.get(key, 0) + 1
    winners = {f.bucket for f in cache.all_faces(only_selected=True)}
    rows = [
        {"key": k, "start": _bucket_start(k, grain).isoformat(), "n": n, "winner": k in winners}
        for k, n in sorted(counts.items(), key=lambda kv: _bucket_start(kv[0], grain))
    ]
    if len(rows) < 2:
        return rows, "Too few dated buckets to show gaps."
    starts = [date.fromisoformat(r["start"]) for r in rows]
    gaps = [(b - a, a, b) for a, b in zip(starts, starts[1:])]
    longest, a, b = max(gaps)
    step = timedelta(days=_bucket_days(grain))
    if longest <= step * 1.5:
        return rows, f"Every {grain} from {starts[0]} to {starts[-1]} has at least one photo."
    return rows, (
        f"Longest gap: {longest.days} days with no usable photo, "
        f"from {a.isoformat()} to {b.isoformat()}."
    )


def _jitter(values: list[float]) -> float:
    return float(np.mean(np.abs(np.diff(values)))) if len(values) > 1 else 0.0


def _alignment(manifest: dict[str, Any] | None) -> tuple[list[dict[str, Any]] | None, str]:
    if manifest is None:
        return None, ""
    rows = [
        {
            "i": i,
            "date": (r["date"] or "")[:10],
            "file": Path(r["path"]).name,
            "raw": round(r["eye_dev_raw_px"], 3),
            "smooth": round(r["eye_dev_smoothed_px"], 3),
        }
        for i, r in enumerate(manifest["frames"])
        if "eye_dev_raw_px" in r
    ]
    if not rows:
        return rows, "This manifest predates the alignment columns; rerun facelapse render."
    raw = _jitter([r["raw"] for r in rows])
    smooth = _jitter([r["smooth"] for r in rows])
    return rows, (
        f"Frame-to-frame eye-midpoint jitter: {smooth:.2f} px smoothed vs {raw:.2f} px raw."
    )


def _provenance(cache: Cache) -> tuple[dict[str, int], str]:
    counts: dict[str, int] = {}
    for f in cache.all_faces(only_selected=True):
        name = cache.image_ts(f.content_hash).source.name
        counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    if not total:
        return counts, ""
    strong = sum(
        counts.get(s.name, 0)
        for s in (TsSource.TAKEOUT_JSON, TsSource.EXIF_ORIGINAL, TsSource.EXIF_DIGITIZED)
    )
    return (
        counts,
        f"{100 * strong / total:.0f}% of the {total} winners are dated by EXIF or Takeout.",
    )


def build_data(cfg: Config, thumbs: bool = True, synthetic: bool = False) -> dict[str, Any]:
    cache = Cache(cfg.cache_db, cfg.person)
    manifest_path = Path(cfg.out_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    funnel, funnel_note = _funnel(cache)
    coverage, coverage_note = _coverage(cache, cfg.bucket_grain)
    alignment, alignment_note = _alignment(manifest)
    provenance, provenance_note = _provenance(cache)
    data = {
        "banner": SYNTHETIC_BANNER if synthetic else None,
        "funnel": funnel,
        "funnel_note": funnel_note,
        "has_seeds": bool(cache.seeds()),
        "thresholds": {"t_abs": cfg.id_t_abs, "margin": cfg.id_margin},
        "points": _points(cache, thumbs),
        "has_selected": bool(cache.all_faces(only_selected=True)),
        "grain": cfg.bucket_grain,
        "coverage": coverage,
        "coverage_note": coverage_note,
        "alignment": alignment,
        "alignment_note": alignment_note,
        "provenance": provenance,
        "provenance_note": provenance_note,
    }
    cache.close()
    return data


def write_report(cfg: Config, thumbs: bool = True, synthetic: bool = False) -> Path:
    if not Path(cfg.cache_db).exists():
        raise FileNotFoundError(f"no cache at {cfg.cache_db}; run facelapse analyze first")
    data = build_data(cfg, thumbs=thumbs, synthetic=synthetic)
    blob = json.dumps(data).replace("</", "<\\/")
    page = (Path(__file__).with_name("report_template.html")).read_text()
    out = Path(cfg.out_dir) / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.replace("/*DATA*/null", blob))
    return out

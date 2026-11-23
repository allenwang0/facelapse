"""`facelapse report`: one self-contained HTML page (inline SVG and vanilla
JS, no CDN) that explains a run. It reads cache.db and out/manifest.json,
embeds the numbers as JSON, and works offline on private data.

Views: a summary hero (with the rendered timelapse when it exists), the
funnel, an identity calibration explorer, timeline coverage, alignment
stability and date provenance. Every sentence that carries a number is
computed here from the data; the page styles come from the vendored
report-kit (facelapse/kit/, inlined at build time, never edited here).
"""

from __future__ import annotations

import base64
import html
import io
import json
import platform
import subprocess
from datetime import date, datetime, timedelta, timezone
from importlib import metadata
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


def _drift_example(rows: list[dict[str, Any]], window: int) -> dict[str, Any] | None:
    """The frame the smoother moved farthest from its target, and the
    neighbour whose raw offset it absorbed. smoothing.smooth_params sets each
    frame's canvas offset to the moving average of the raw offsets in its
    window (edge-padded), so a frame that a 2-point fit puts exactly on
    target inherits a share of a nearby frame's offset."""
    if not rows or window <= 1:
        return None
    i = max(range(len(rows)), key=lambda k: rows[k]["smooth"] - rows[k]["raw"])
    if rows[i]["smooth"] <= rows[i]["raw"]:
        return None
    half = window // 2
    near = [k for k in range(i - half, i + half + 1) if 0 <= k < len(rows) and k != i]
    j = max(near, key=lambda k: rows[k]["raw"])
    share = rows[j]["raw"] / window
    return {
        "frame": i + 1,
        "raw": rows[i]["raw"],
        "smooth": rows[i]["smooth"],
        "neighbour": j + 1,
        "neighbour_raw": rows[j]["raw"],
        # true when the smoothed offset is exactly the neighbour's share
        "is_share": abs(share - rows[i]["smooth"]) < 0.01,
        "farther": sum(r["smooth"] > r["raw"] for r in rows),
    }


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


def _media(out_dir: Path) -> dict[str, Any] | None:
    """The run's own rendered timelapse, referenced next to report.html. The
    GIF starts paused on a poster (its first frame, inlined) so nothing
    loops until the reader presses play."""
    gif = out_dir / "timelapse.gif"
    if gif.exists():
        try:
            with Image.open(gif) as im:
                first = im.convert("RGB")
                size = first.size
                buf = io.BytesIO()
                first.save(buf, format="JPEG", quality=82)
        except (OSError, ValueError):
            return None
        poster = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return {"kind": "gif", "src": gif.name, "poster": poster, "w": size[0], "h": size[1]}
    if (out_dir / "timelapse.mp4").exists():
        return {"kind": "mp4", "src": "timelapse.mp4"}
    return None


def _git_sha() -> str | None:
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _env(cfg: Config) -> dict[str, Any]:
    """What produced this page, for the Reproduce table and the footer."""
    return {
        "facelapse": _version("facelapse"),
        "sha": _git_sha(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "numpy": _version("numpy"),
        "opencv": _version("opencv-python-headless") or _version("opencv-python"),
        "pillow": _version("pillow"),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "smooth_window": cfg.smooth_window,
    }


def build_data(cfg: Config, thumbs: bool = True, synthetic: bool = False) -> dict[str, Any]:
    cache = Cache(cfg.cache_db, cfg.person)
    manifest_path = Path(cfg.out_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    funnel, funnel_note = _funnel(cache)
    coverage, coverage_note = _coverage(cache, cfg.bucket_grain)
    alignment, alignment_note = _alignment(manifest)
    provenance, provenance_note = _provenance(cache)
    jitter = None
    if alignment:
        # smoothing.smooth_params rounds an even window up to the next odd one
        window = cfg.smooth_window if cfg.smooth_window % 2 else cfg.smooth_window + 1
        jitter = {
            "raw": round(_jitter([r["raw"] for r in alignment]), 4),
            "smooth": round(_jitter([r["smooth"] for r in alignment]), 4),
            "window": window,
            "example": _drift_example(alignment, window),
        }
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
        "jitter": jitter,
        "media": _media(Path(cfg.out_dir)),
        "env": _env(cfg),
    }
    cache.close()
    return data


# ---------------------------------------------------------------- page copy
# Each sentence below is built from the run's numbers so the page, the
# README and site/card.json can't drift apart by hand-editing.

STAGE_PHRASE = {
    "decoded": "decoding",
    "face found": "face detection",
    "identity": "identity matching",
    "pose": "the head-pose filter",
    "neutral": "the expression filter",
    "size": "the face-size filter",
    "dedup": "de-duplication",
    "trusted date": "the date check",
}
PICK_STAGE = "bucket winners"
SOURCE_LABEL = {
    "TAKEOUT_JSON": "Takeout",
    "EXIF_ORIGINAL": "EXIF",
    "EXIF_DIGITIZED": "EXIF digitized",
    "FILENAME": "filename",
    "MTIME": "file mtime",
}


def esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def _n(v: int) -> str:
    return f"{v:,}"


def _pct(a: float, b: float) -> str:
    p = 100 * a / b if b else 0.0
    return f"{p:.0f}\u202f%" if p >= 10 or p == 0 else f"{p:.1f}\u202f%"


def _card(span: int, href: str, label: str, metric: str, cmp: str, foot: str) -> str:
    return (
        f'<a class="card span-{span}" href="#{href}"><span class="label">{label}</span>'
        f'<span class="metric num">{metric}</span><span class="cmp">{cmp}</span>'
        f'<span class="foot">{foot}</span></a>'
    )


def _unit(u: str) -> str:
    return f'<span class="unit">{esc(u)}</span>'


def page_copy(d: dict[str, Any]) -> dict[str, str]:
    """HTML snippets for the template's {{slots}}. Text renders in the first
    paint and without JavaScript; the charts draw on top of it."""
    synthetic = bool(d.get("banner"))
    grain = d.get("grain") or "bucket"
    stage = {r["stage"]: r for r in d.get("funnel") or []}
    first = (d.get("funnel") or [{}])[0]
    ingested = first.get("total") or first.get("kept") or 0
    frames = len(d["alignment"]) if d.get("alignment") else None
    if frames is None and PICK_STAGE in stage:
        frames = stage[PICK_STAGE]["kept"]
    dates = sorted(r["date"] for r in d.get("alignment") or [] if r.get("date"))
    years = f"{dates[0][:4]}\u2013{dates[-1][:4]}" if dates else ""
    c: dict[str, str] = {}

    # hero
    if frames is not None and ingested:
        c["h1"] = f"{_n(frames)} frames from {_n(ingested)} photos" + (
            f", {years}" if years else ""
        )
    else:
        c["h1"] = "Facelapse run report"
    c["lede"] = (
        f"facelapse looked for one person across {_n(ingested)} photos, dropped turned heads, "
        f"expressions, duplicates and undated shots, kept the best photo per {esc(grain)} "
        "and lined up the eyes. This page shows where every photo went."
        if ingested
        else "Run facelapse analyze to fill this report."
    )
    conds = (
        ['<span class="key">Synthetic data</span>', "<span>Drawn faces, stubbed detection</span>"]
        if synthetic
        else ['<span class="key">Your archive</span>']
    )
    conds += ["<span>One run</span>", f"<span>One photo per {esc(grain)}</span>"]
    c["conditions"] = "".join(conds)
    c["banner"] = (
        f'<p class="caveat" id="banner" role="note"><b>{esc(d["banner"])}</b> '
        "The numbers show that each stage works, not how accurate it is on real faces.</p>"
        if synthetic
        else ""
    )
    m = d.get("media")
    if m and m["kind"] == "gif":
        c["media"] = (
            '<figure class="hero-media" aria-labelledby="media-cap">'
            f'<img id="tl" src="{m["poster"]}" data-gif="{esc(m["src"])}" '
            f'width="{m["w"]}" height="{m["h"]}" '
            f'alt="First frame of the rendered timelapse: an aligned face on a fixed canvas.">'
            '<div class="media-bar">'
            '<button class="btn" type="button" id="tl-play" aria-pressed="false">'
            "Play timelapse</button>"
            f'<span id="media-cap">The rendered output: {_n(frames or 0)} aligned frames, '
            f"one per {esc(grain)}.</span></div></figure>"
        )
    elif m and m["kind"] == "mp4":
        c["media"] = (
            '<figure class="hero-media" aria-labelledby="media-cap">'
            f'<video controls preload="metadata" src="{esc(m["src"])}"></video>'
            f'<div class="media-bar"><span id="media-cap">The rendered output: '
            f"{_n(frames or 0)} aligned frames.</span></div></figure>"
        )
    else:
        c["media"] = ""

    # funnel
    filters = [
        r
        for r in d.get("funnel") or []
        if r["stage"] in STAGE_PHRASE and r["total"] and r["kept"] < r["total"]
    ]
    pick = stage.get(PICK_STAGE)
    if filters:
        big = max(filters, key=lambda r: r["total"] - r["kept"])
        phrase = STAGE_PHRASE[big["stage"]]
        removed = big["total"] - big["kept"]
        c["funnel_h2"] = (
            f"{phrase[0].upper() + phrase[1:]} removed the most photos of any filter: "
            f"{_n(removed)} of {_n(big['total'])}"
        )
    else:
        c["funnel_h2"] = "No filter has removed a photo yet"
    if pick and pick["total"]:
        c["funnel_lede"] = (
            f"{_n(pick['total'])} dated photos passed every filter. Keeping the best one per "
            f"{esc(grain)} then set aside {_n(pick['total'] - pick['kept'])} more by design, "
            f"leaving {_n(pick['kept'])} frames."
        )
    else:
        c["funnel_lede"] = "Run facelapse select to pick one photo per bucket."
    if pick and filters and pick["total"] - pick["kept"] >= removed:
        c["funnel_fig"] = f"The one-per-{esc(grain)} pick sets aside more photos than any filter"
    elif filters:
        c["funnel_fig"] = (
            f"{phrase[0].upper() + phrase[1:]} removes more photos than any other stage"
        )
    else:
        c["funnel_fig"] = "Photos left after each stage"
    c["funnel_sub"] = (
        f"Photos left after each stage \u00b7 count \u00b7 n = {_n(ingested)} ingested"
    )

    # identity
    pts = d.get("points") or []
    out = {k: sum(p["stored"] == k for p in pts) for k in ("confident", "ambiguous", "none")}
    if pts:
        c["calib_h2"] = (
            f"{_n(out['confident'])} of {_n(len(pts))} photos match the seed photos confidently; "
            f"{_n(out['ambiguous'])} close calls go to review"
        )
        nones = [p["top"] for p in pts if p["stored"] == "none"]
        rest = [p["top"] for p in pts if p["stored"] != "none"]
        split = ""
        if nones and rest and max(nones) < min(rest):
            split = (
                f" No photo scores between {max(nones):.2f} and {min(rest):.2f} similarity, so any "
                "id_t_abs in that range gives this run the same result."
            )
        c["calib_lede"] = (
            "id_t_abs is the lowest similarity to your seed photos that counts as you. id_margin "
            "is how far your face must beat the next-best face in the same photo; closer calls "
            f"go to the review page.{split}"
        )
    else:
        c["calib_h2"] = "No photo has been scored against the seeds yet"
        c["calib_lede"] = "Run facelapse seed add, then facelapse score."
    t = d.get("thresholds") or {}
    c["calib_sub"] = (
        f"One mark per photo \u00b7 cosine similarity \u00b7 n = {_n(len(pts))} \u00b7 run used "
        f"id_t_abs {t.get('t_abs', 0):.2f}, id_margin {t.get('margin', 0):.2f}"
    )
    c["calib_caveat"] = (
        '<p class="caveat" role="note"><b>Drawn faces separate cleanly.</b> This run says '
        "nothing about how far apart real faces score, so check the gap on your own archive.</p>"
        if synthetic
        else ""
    )

    # coverage
    cov = d.get("coverage") or []
    with_frame = sum(r["winner"] for r in cov)
    gap = _gap(cov, grain)
    if gap:
        c["coverage_h2"] = (
            f"The longest gap is {_n(gap[0])} days with no usable photo, from {gap[1]} to {gap[2]}"
        )
    elif cov:
        c["coverage_h2"] = f"Every {esc(grain)} in the timeline has at least one photo"
    else:
        c["coverage_h2"] = "No dated photos yet"
    c["coverage_lede"] = (
        f"{_n(with_frame)} of the {_n(len(cov))} {esc(grain)}s with a dated face produced a "
        "frame. In the rest, every photo was removed by a filter."
        if cov
        else "Run facelapse select to see coverage."
    )
    c["coverage_fig"] = (
        f"Frames span {years or 'the timeline'}, broken by one {_n(gap[0])}-day gap"
        if gap
        else f"Photos per {esc(grain)}"
    )
    c["coverage_sub"] = (
        f"Photos with a dated face per {esc(grain)} \u00b7 count \u00b7 "
        f"{_n(len(cov))} {esc(grain)}s"
    )

    # alignment
    j = d.get("jitter")
    if j:
        n_tr = len(d["alignment"]) - 1
        if j["smooth"] < j["raw"]:
            c["align_h2"] = (
                f"Smoothing cuts frame-to-frame eye jitter from {j['raw']:.2f}\u202fpx "
                f"to {j['smooth']:.2f}\u202fpx"
            )
            c["align_fig"] = (
                f"Smoothing lowers mean jitter by {_pct(j['raw'] - j['smooth'], j['raw'])}"
            )
        else:
            c["align_h2"] = (
                f"Smoothing does not lower eye jitter on this run: {j['smooth']:.2f}\u202fpx "
                f"smoothed vs {j['raw']:.2f}\u202fpx raw"
            )
            c["align_fig"] = "Smoothed and raw jitter per frame"
        c["align_lede"] = (
            "Jitter is how much the eye midpoint's distance from its target changes from one "
            f"frame to the next, averaged over {_n(n_tr)} transitions. Smoothing replaces each "
            f"frame's offset with the average over a {j['window']}-frame window."
        )
        ex = j.get("example")
        if ex:
            if ex["is_share"]:
                why = (
                    f", a 1/{j['window']} share of frame {ex['neighbour']}'s "
                    f"{ex['neighbour_raw']:.2f}\u202fpx ({ex['neighbour_raw']:.2f} \u00f7 "
                    f"{j['window']} = {ex['smooth']:.2f})"
                )
            else:
                why = (
                    f"; its window includes frame {ex['neighbour']}, whose raw fit is "
                    f"{ex['neighbour_raw']:.2f}\u202fpx off"
                )
            c["align_caveat"] = (
                '<p class="caveat" role="note"><b>Steadier is not closer.</b> '
                f"In {_n(ex['farther'])} of {_n(len(d['alignment']))} frames the smoothed eyes sit "
                "farther from the target than the raw fit, because a frame inherits part of its "
                f"neighbours' offsets. Frame {ex['frame']} moves from {ex['raw']:.2f} to "
                f"{ex['smooth']:.2f}\u202fpx{why}.</p>"
            )
        else:
            c["align_caveat"] = ""
        c["align_sub"] = (
            f"Change in eye-midpoint offset between consecutive frames \u00b7 px \u00b7 "
            f"{_n(n_tr)} transitions"
        )
    else:
        c["align_h2"] = "Alignment has not been measured yet"
        c["align_lede"] = "Run facelapse render to measure alignment."
        c["align_caveat"] = ""
        c["align_sub"] = ""
        c["align_fig"] = "Eye alignment per frame"

    # bento
    prov = d.get("provenance") or {}
    total = sum(prov.values())
    strong = sum(prov.get(k, 0) for k in ("TAKEOUT_JSON", "EXIF_ORIGINAL", "EXIF_DIGITIZED"))
    cards = []
    if frames is not None and ingested:
        cut = (
            f"{_n(pick['total'] - pick['kept'])} set aside by the one-per-{esc(grain)} pick"
            if pick
            else ""
        )
        cards.append(
            _card(
                4,
                "funnel-s",
                "Frames rendered",
                f"{_n(frames)} {_unit('of ' + _n(ingested))}",
                f"photos became frames; {_n(ingested - (pick['total'] if pick else frames))} "
                f"removed by filters, {cut}.",
                "Where photos were dropped \u2192",
            )
        )
    if gap:
        cards.append(
            _card(
                4,
                "coverage-s",
                "Longest gap",
                f"{_n(gap[0])} {_unit('days')}",
                f"from {gap[1]} to {gap[2]} with no usable photo.",
                f"{_n(len(cov))} {esc(grain)}s with photos",
            )
        )
    if j:
        cards.append(
            _card(
                4,
                "align-s",
                "Eye jitter, raw \u2192 smoothed",
                f"{j['raw']:.2f} \u2192 {j['smooth']:.2f} {_unit('px')}",
                "mean change in eye position between consecutive frames.",
                f"{j['window']}-frame smoothing window \u00b7 one run",
            )
        )
    if pts:
        cards.append(
            _card(
                6,
                "calib-s",
                "Identity matches",
                f"{_n(out['confident'])} {_unit('of ' + _n(len(pts)))}",
                f"photos matched confidently; {_n(out['ambiguous'])} close calls went to "
                "review and "
                f"{_n(out['none'])} matched no seed.",
                f"id_t_abs {t.get('t_abs', 0):.2f} \u00b7 id_margin {t.get('margin', 0):.2f}",
            )
        )
    if total:
        parts = ", ".join(
            f"{_n(n)} {SOURCE_LABEL.get(k, k)}"
            for k, n in sorted(prov.items(), key=lambda kv: -kv[1])
        )
        cards.append(
            _card(
                6,
                "method",
                "Frames dated by EXIF or Takeout",
                _pct(strong, total),
                f"of {_n(total)} frames ({parts}).",
                "Filename dates are weaker; mtime-only photos are dropped",
            )
        )
    c["bento"] = "".join(cards)

    # interpretation
    shows = ["Every stage runs end to end and logs what it removed and why."]
    shows.append("The thresholds on this page classify photos exactly as the pipeline does.")
    if j and j["smooth"] < j["raw"]:
        shows.append(
            f"Smoothing lowers frame-to-frame jitter on this run, "
            f"{j['raw']:.2f} to {j['smooth']:.2f}\u202fpx."
        )
    nots = []
    if synthetic:
        nots.append("Accuracy on real faces: detection is stubbed and the faces are drawn.")
    nots.append("Run-to-run spread: this is one run.")
    if j and j.get("example"):
        nots.append("That smoothed frames sit closer to the target. They are steadier, not closer.")
    c["shows"] = "".join(f"<li>{x}</li>" for x in shows)
    c["not_shows"] = "".join(f"<li>{x}</li>" for x in nots)

    # reproduce + footer
    e = d.get("env") or {}
    sha = e.get("sha")
    rows = [
        ("facelapse", e.get("facelapse")),
        ("Commit", sha[:7] if sha else None),
        ("Python", e.get("python")),
        ("Platform", e.get("platform")),
        ("NumPy", e.get("numpy")),
        ("OpenCV", e.get("opencv")),
        ("Pillow", e.get("pillow")),
        ("Bucket", grain),
        ("id_t_abs / id_margin", f"{t.get('t_abs', 0):.2f} / {t.get('margin', 0):.2f}"),
        ("Smoothing window", f"{j['window']} frames" if j else None),
        ("Generated", e.get("generated")),
    ]
    c["env_rows"] = "".join(
        f'<tr><th scope="row">{esc(k)}</th><td>{esc(v)}</td></tr>' for k, v in rows if v
    )
    c["env_count"] = str(sum(1 for _, v in rows if v))
    made = (
        f'commit <span class="mono">{esc(sha[:7])}</span>'
        if sha
        else f"facelapse {esc(e.get('facelapse') or '')}"
    )
    c["footer"] = f"Generated from {made} \u00b7 {esc(e.get('generated') or '')}"
    c["description"] = esc(
        "Run report for facelapse, a tool that turns a photo archive into a face-aging timelapse: "
        "where photos were dropped, identity thresholds, timeline gaps and alignment stability."
    )
    return c


def _gap(cov: list[dict[str, Any]], grain: str) -> tuple[int, str, str] | None:
    """Longest stretch between consecutive dated buckets, as _coverage
    reports it (None when no gap exceeds 1.5 buckets)."""
    if len(cov) < 2:
        return None
    starts = [date.fromisoformat(r["start"]) for r in cov]
    longest, a, b = max((b - a, a, b) for a, b in zip(starts, starts[1:]))
    if longest <= timedelta(days=_bucket_days(grain)) * 1.5:
        return None
    return longest.days, a.isoformat(), b.isoformat()


def _kit(name: str) -> str:
    # the kit's comments quote "<script ...></script>"; escape closing tags so
    # the inlined copy can't end its own <script>/<style> element early
    text = (Path(__file__).with_name("kit") / name).read_text()
    return text.replace("</script", "<\\/script").replace("</style", "<\\/style")


def write_report(cfg: Config, thumbs: bool = True, synthetic: bool = False) -> Path:
    if not Path(cfg.cache_db).exists():
        raise FileNotFoundError(f"no cache at {cfg.cache_db}; run facelapse analyze first")
    data = build_data(cfg, thumbs=thumbs, synthetic=synthetic)
    # the poster is already in the <img>; don't inline it twice
    media = data["media"] and {k: v for k, v in data["media"].items() if k != "poster"}
    blob = json.dumps({**data, "media": media}).replace("</", "<\\/")
    page = (Path(__file__).with_name("report_template.html")).read_text()
    for key, value in page_copy(data).items():
        page = page.replace("{{" + key + "}}", value)
    page = page.replace("/*KIT_CSS*/", _kit("report.css"))
    page = page.replace("/*KIT_JS*/", _kit("report.js") + "\n" + _kit("charts.js"))
    out = Path(cfg.out_dir) / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.replace("/*DATA*/null", blob))
    return out

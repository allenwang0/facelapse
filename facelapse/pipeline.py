"""Orchestration. Four commands' worth of logic:

  analyze : ingest -> detect/embed/dense (EXPENSIVE, cached). idempotent.
  score   : identity gate -> pose -> neutral -> quality. cheap, re-runnable.
  select  : dedup -> bucket -> best-per-bucket. cheap.
  render  : align -> photometric -> smooth -> timeline -> encode + manifest.

Every filtering stage logs pass/fail counts. Attrition multiplies across a
funnel; if you keep 4% of photos you must be able to see WHICH stage ate them.
Silent starvation is the main failure mode of this kind of pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import imagehash
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed: return iterable unchanged
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


from . import align as align_mod
from . import bucketing, dedup, ingest, photometric, quality, render, smoothing, timestamps
from . import neutral as neutral_mod
from . import pose as pose_mod
from .cache import Cache
from .config import Config
from .identity import IdOutcome, ReferenceChain, resolve_image
from .records import ImageRecord

# Funnel stages in order. Each is recorded in the cache with its rejection
# reasons so `facelapse report` shows the same numbers the log printed.
STAGES = [
    "ingested",
    "decoded",
    "face found",
    "identity",
    "pose",
    "neutral",
    "size",
    "dedup",
    "trusted date",
    "bucket winners",
    "rendered",
]


def _log(
    stage: str,
    kept: int,
    total: int,
    cache: Cache | None = None,
    reasons: dict[str, int] | None = None,
) -> None:
    pct = (100.0 * kept / total) if total else 0.0
    print(f"[funnel] {stage:<18} kept {kept:>6} / {total:<6} ({pct:5.1f}%)")
    if cache is not None:
        cache.record_stage(stage, kept, total, {k: v for k, v in (reasons or {}).items() if v})


# ---------------------------------------------------------------- analyze
def analyze(folder: str, cfg: Config, detector=None, mesh=None, force: bool = False) -> None:
    """Expensive inference pass. Requires the ML deps. Detector/mesh injected
    so tests can stub them."""
    if detector is None:
        from .detection import FaceDetector

        detector = FaceDetector()
    if mesh is None:
        try:
            from .landmarks import FaceMesh

            mesh = FaceMesh()
        except RuntimeError:
            mesh = None
            print(
                "[warn] mediapipe absent: dense landmarks off; "
                "neutral scoring and nasion alignment degraded."
            )

    cache = Cache(cfg.cache_db)
    # Count total images first for progress bar
    image_paths = list(ingest.iter_images(folder, cfg.image_exts))
    n_img = n_face = 0
    for path in tqdm(image_paths, desc="Analyzing images", unit="img"):
        try:
            ch = ingest.content_hash(path)
        except ValueError as e:
            ch = ingest.path_key(path)
            cache.upsert_image(
                ImageRecord(
                    ch, str(path), 0, 0, timestamps.resolve(path), decoded_ok=False, error=str(e)
                )
            )
            continue
        if cache.has_image(ch) and not force:
            continue
        try:
            rgb = ingest.decode(path)
        except ingest.DECODE_ERRORS as e:
            cache.upsert_image(
                ImageRecord(
                    ch, str(path), 0, 0, timestamps.resolve(path), decoded_ok=False, error=str(e)
                )
            )
            continue
        ts = timestamps.resolve(path)
        faces = detector.analyze(rgb, ch)
        for f in faces:
            f.interocular_px = quality.interocular_px(f)
            f.sharpness = quality.sharpness(rgb, f.bbox)
            if mesh is not None:
                crop = _crop(rgb, f.bbox)
                dense = mesh.landmarks(crop) if crop.size else None
                if dense is not None:
                    dense[:, 0] += max(0, int(f.bbox[0]))  # back to image coords
                    dense[:, 1] += max(0, int(f.bbox[1]))
                    f.dense = dense
        cache.upsert_image(ImageRecord(ch, str(path), rgb.shape[1], rgb.shape[0], ts, faces=faces))
        cache.set_phash(ch, str(dedup.phash(rgb)))
        n_img += 1
        n_face += len(faces)
    print(f"[analyze] {n_face} faces detected across {n_img} new images")
    rows = cache.db.execute(
        "SELECT i.decoded_ok, i.error, COUNT(f.face_id) FROM images i "
        "LEFT JOIN faces f ON f.content_hash = i.content_hash GROUP BY i.content_hash"
    ).fetchall()
    decoded = [r for r in rows if r[0]]
    errors: dict[str, int] = {}
    for r in rows:
        if not r[0]:
            key = (r[1] or "unreadable").split(":")[0][:60]
            errors[key] = errors.get(key, 0) + 1
    with_face = sum(1 for r in decoded if r[2])
    _log("ingested", len(rows), len(rows), cache)
    _log("decoded", len(decoded), len(rows), cache, errors)
    _log(
        "face found", with_face, len(decoded), cache, {"no face detected": len(decoded) - with_face}
    )
    cache.close()


# ---------------------------------------------------------------- score
def score(cfg: Config, au_scorer=None) -> None:
    cache = Cache(cfg.cache_db)
    seeds = cache.seeds()
    chain = ReferenceChain(seeds, k=cfg.id_reference_k)

    # cleared in the same transaction as the writes below, so a stricter
    # rerun cannot leave last run's is_self flags behind
    cache.reset_scores()
    all_faces = cache.all_faces()
    by_img: dict[str, list] = {}
    for f in tqdm(all_faces, desc="Grouping faces", unit="face", disable=len(all_faces) < 100):
        by_img.setdefault(f.content_hash, []).append(f)

    promoted = cache.promotions()
    counts = {o: 0 for o in IdOutcome}
    self_faces = []
    for ch, faces in by_img.items():
        when = cache.image_ts(ch).value
        outcome, chosen = resolve_image(faces, when, chain, cfg.id_t_abs, cfg.id_margin)
        picked = [f for f in faces if (ch, f.face_id) in promoted]
        if picked:
            for f in faces:
                f.is_self = f is picked[0]
            outcome, chosen = IdOutcome.PROMOTED, picked[0]
        counts[outcome] += 1
        cache.set_outcome(ch, outcome.value)
        if outcome in (IdOutcome.CONFIDENT, IdOutcome.PROMOTED):
            self_faces.append(chosen)
    identity_reasons = {
        "ambiguous: top two faces too close": counts[IdOutcome.AMBIGUOUS],
        "no face matched the seeds": counts[IdOutcome.NONE],
    }

    # pose + neutral gates on the self-faces only
    pose_reasons = {"yaw over the limit": 0, "pitch over the limit": 0, "pose unknown": 0}
    neutral_reasons = {"eyes closed": 0, "mouth open": 0, "smile or other expression": 0}
    for f in self_faces:
        f.passes_pose = pose_mod.passes_pose(f, cfg.max_abs_yaw_deg, cfg.max_abs_pitch_deg)
        if f.dense is not None:
            neutral_mod.score_face(f, cfg, au_scorer=au_scorer)
            f.passes_neutral = (
                f.neutral_penalty is not None and f.neutral_penalty <= cfg.neutral_max_penalty
            )
        else:
            f.passes_neutral = None  # unknown; select stage will treat conservatively
        if not f.passes_pose:
            if np.isnan(f.yaw) or np.isnan(f.pitch):
                pose_reasons["pose unknown"] += 1
            elif abs(f.yaw) > cfg.max_abs_yaw_deg:
                pose_reasons["yaw over the limit"] += 1
            else:
                pose_reasons["pitch over the limit"] += 1
        elif f.passes_neutral is False:
            if f.ear is not None and f.ear < cfg.ear_closed_below:
                neutral_reasons["eyes closed"] += 1
            elif f.mar is not None and f.mar > cfg.mar_open_above:
                neutral_reasons["mouth open"] += 1
            else:
                neutral_reasons["smile or other expression"] += 1
    cache.update_faces(all_faces)
    posed = [f for f in self_faces if f.passes_pose]
    neutral = [f for f in posed if f.passes_neutral in (True, None)]
    _log("identity", len(self_faces), len(by_img), cache, identity_reasons)
    _log("pose", len(posed), len(self_faces), cache, pose_reasons)
    _log("neutral", len(neutral), len(posed), cache, neutral_reasons)
    cache.close()


def apply_review(cfg: Config, review_path: str) -> int:
    """Store the "this one is me" promotions from a review.json download.
    They persist in the cache, so every later score run honors them."""
    data = json.loads(Path(review_path).read_text())
    keys = [(p["content_hash"], int(p["face_id"])) for p in data.get("promotions", [])]
    cache = Cache(cfg.cache_db)
    cache.add_promotions(keys)
    cache.close()
    return len(keys)


# ---------------------------------------------------------------- select
def select(cfg: Config, allow_mtime_dates: bool = False) -> list:
    """allow_mtime_dates admits photos dated only by file mtime; by default
    they are dropped because mtime dates screenshots, copies and re-saves."""
    cache = Cache(cfg.cache_db)
    cache.reset_selection()
    self_faces = cache.all_faces(only_self=True)

    # gates: pose required; neutral required if known, allowed-through if unknown
    passed = [f for f in self_faces if f.passes_pose and (f.passes_neutral in (True, None))]
    gated = [f for f in passed if (f.interocular_px or 0) >= cfg.min_interocular_px]
    small = f"eyes under {cfg.min_interocular_px:.0f} px apart"
    _log("size", len(gated), len(passed), cache, {small: len(passed) - len(gated)})

    # dedup across the gated set (re-saved copies)
    # pHash is stored by analyze; caches from before that column are filled here once
    hashes = {}
    for f in gated:
        stored = cache.phash(f.content_hash)
        if stored is None:
            rgb = ingest.decode(Path(cache.image_path(f.content_hash)))
            stored = str(dedup.phash(rgb))
            cache.set_phash(f.content_hash, stored)
        hashes[f.content_hash] = imagehash.hex_to_hash(stored)
    canon = dedup.group_duplicates(hashes, cfg.phash_hamming_dup_below)
    seen, deduped = set(), []
    for f in gated:
        rep = canon[f.content_hash]
        if rep in seen:
            continue
        seen.add(rep)
        deduped.append(f)
    dup = {"near-duplicate of another photo": len(gated) - len(deduped)}
    _log("dedup", len(deduped), len(gated), cache, dup)

    ts_of = {}
    date_reasons: dict[str, int] = {}
    for f in deduped:
        ts = cache.image_ts(f.content_hash)
        if ts.value is not None and (ts.is_trustworthy or allow_mtime_dates):
            ts_of[f.content_hash] = ts.value
        else:
            key = "dated only by file mtime" if ts.value is not None else "no date"
            date_reasons[key] = date_reasons.get(key, 0) + 1
    dated = [f for f in deduped if f.content_hash in ts_of]
    _log("trusted date", len(dated), len(deduped), cache, date_reasons)
    winners = bucketing.select_best_per_bucket(dated, ts_of, cfg)
    lost = {"lost to a better photo in its bucket": len(dated) - len(winners)}
    _log("bucket winners", len(winners), len(dated), cache, lost)

    cache.update_faces(winners)
    cache.close()
    return winners


# ---------------------------------------------------------------- render
def render_final(cfg: Config, rejects: set[str] | None = None) -> dict:
    """Two streaming passes, so peak memory does not grow with frame count.
    Pass 1 (photometric modes only) warps each frame to measure its mean
    luminance and keeps nothing else; pass 2 decodes, warps, normalizes,
    overlays and encodes one frame at a time."""
    cache = Cache(cfg.cache_db)
    winners = cache.all_faces(only_selected=True)
    n_selected = len(winners)
    rejects = rejects or set()
    winners = [f for f in winners if f.content_hash not in rejects]

    ts_of = {f.content_hash: cache.image_ts(f.content_hash) for f in winners}
    paths = {f.content_hash: cache.image_path(f.content_hash) for f in winners}
    cache.close()
    winners.sort(key=lambda f: ts_of[f.content_hash].value)

    params_seq = [align_mod.estimate(f, cfg) for f in winners]
    params_seq = smoothing.smooth_params(
        params_seq,
        [f.kps[:2] for f in winners],
        (cfg.left_eye_xy, cfg.right_eye_xy),
        cfg.smooth_window,
    )

    def warped(i: int) -> np.ndarray:
        rgb = ingest.decode(Path(paths[winners[i].content_hash]))
        return align_mod.warp(rgb, params_seq[i], cfg)

    ref = None
    if cfg.photometric_mode != "none" and winners:
        lums = [
            photometric.luminance(warped(i))
            for i in tqdm(
                range(len(winners)),
                desc="Measuring exposure",
                unit="frame",
                disable=len(winners) < 20,
            )
        ]
        ref = warped(photometric.median_index(lums))

    def frames():
        for i in tqdm(
            range(len(winners)), desc="Rendering", unit="frame", disable=len(winners) < 20
        ):
            frame = warped(i)
            if ref is not None:
                frame = photometric.normalize(frame, ref, cfg.photometric_mode)
            yield frame

    dates = [ts_of[f.content_hash].value for f in winners]
    confident = [ts_of[f.content_hash].confidence == "high" for f in winners]
    rows = [
        {
            "content_hash": f.content_hash,
            "path": paths[f.content_hash],
            "date": ts_of[f.content_hash].value.isoformat(),
            "date_source": ts_of[f.content_hash].source.name,
            "date_confidence": ts_of[f.content_hash].confidence,
            "bucket": f.bucket,
            "composite": f.composite,
            "yaw": f.yaw,
            "pitch": f.pitch,
            "neutral_penalty": f.neutral_penalty,
            "id_sim": f.id_sim,
            "id_margin": f.id_margin,
        }
        for f in winners
    ]
    result = render.render_video(frames(), dates, confident, rows, cfg)
    cache = Cache(cfg.cache_db)
    reviewed = {"rejected on the review page": n_selected - len(rows)}
    _log("rendered", len(rows), n_selected, cache, reviewed)
    cache.close()
    return result


def _crop(rgb: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    return rgb[y1:y2, x1:x2]

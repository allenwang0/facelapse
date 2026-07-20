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

from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed: return iterable unchanged
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

from . import ingest, timestamps, quality, pose as pose_mod, neutral as neutral_mod
from . import dedup, bucketing, align as align_mod, photometric, smoothing, render
from .cache import Cache
from .config import Config
from .identity import ReferenceChain, resolve_image, IdOutcome
from .records import ImageRecord


def _log(stage: str, kept: int, total: int) -> None:
    pct = (100.0 * kept / total) if total else 0.0
    print(f"[funnel] {stage:<18} kept {kept:>6} / {total:<6} ({pct:5.1f}%)")


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
            print("[warn] mediapipe absent: dense landmarks off; "
                  "neutral scoring and nasion alignment degraded.")

    cache = Cache(cfg.cache_db)
    # Count total images first for progress bar
    image_paths = list(ingest.iter_images(folder, cfg.image_exts))
    n_img = n_face = 0
    for path in tqdm(image_paths, desc="Analyzing images", unit="img"):
        ch = ingest.content_hash(path)
        if cache.has_image(ch) and not force:
            continue
        try:
            rgb = ingest.decode(path)
        except Exception as e:
            cache.upsert_image(ImageRecord(ch, str(path), 0, 0,
                               timestamps.resolve(path), decoded_ok=False, error=str(e)))
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
                    dense[:, 0] += max(0, int(f.bbox[0]))   # back to image coords
                    dense[:, 1] += max(0, int(f.bbox[1]))
                    f.dense = dense
        cache.upsert_image(ImageRecord(ch, str(path), rgb.shape[1], rgb.shape[0], ts, faces=faces))
        n_img += 1
        n_face += len(faces)
    _log("analyze(images)", n_img, n_img)
    print(f"[funnel] analyze(faces)      {n_face} faces detected across {n_img} new images")
    cache.close()


# ---------------------------------------------------------------- score
def score(cfg: Config, au_scorer=None) -> None:
    cache = Cache(cfg.cache_db)
    seeds = cache.seeds()
    chain = ReferenceChain(seeds, k=cfg.id_reference_k)

    # group faces by image for the margin-based identity decision
    all_faces = cache.all_faces()
    by_img: dict[str, list] = {}
    for f in tqdm(all_faces, desc="Grouping faces", unit="face", disable=len(all_faces) < 100):
        by_img.setdefault(f.content_hash, []).append(f)

    n_conf = n_amb = n_none = 0
    self_faces = []
    for ch, faces in by_img.items():
        when = cache.image_ts(ch).value
        outcome, chosen = resolve_image(faces, when, chain, cfg.id_t_abs, cfg.id_margin)
        if outcome == IdOutcome.CONFIDENT:
            n_conf += 1
            self_faces.append(chosen)
        elif outcome == IdOutcome.AMBIGUOUS:
            n_amb += 1
        else:
            n_none += 1
        cache.update_faces(faces)
    print(f"[funnel] identity           confident {n_conf}  ambiguous {n_amb}  none {n_none}")

    # pose + neutral gates on the self-faces only
    kept_pose = kept_neutral = 0
    for f in self_faces:
        f.passes_pose = pose_mod.passes_pose(f, cfg.max_abs_yaw_deg, cfg.max_abs_pitch_deg)
        if f.dense is not None:
            neutral_mod.score_face(f, cfg, au_scorer=au_scorer)
            f.passes_neutral = (
                f.neutral_penalty is not None and f.neutral_penalty <= cfg.neutral_max_penalty
            )
        else:
            f.passes_neutral = None   # unknown; select stage will treat conservatively
        if f.passes_pose:
            kept_pose += 1
        if f.passes_neutral:
            kept_neutral += 1
        cache.update_face(f)
    _log("pose", kept_pose, len(self_faces))
    _log("neutral", kept_neutral, len(self_faces))
    cache.close()


# ---------------------------------------------------------------- select
def select(cfg: Config) -> list:
    cache = Cache(cfg.cache_db)
    self_faces = cache.all_faces(only_self=True)

    # gates: pose required; neutral required if known, allowed-through if unknown
    gated = [f for f in self_faces
             if f.passes_pose and (f.passes_neutral in (True, None))
             and (f.interocular_px or 0) >= cfg.min_interocular_px]
    _log("size+gates", len(gated), len(self_faces))

    # dedup across the gated set (re-saved copies)
    hashes = {}
    for f in gated:
        rgb = ingest.decode(Path(cache.image_path(f.content_hash)))
        hashes[f.content_hash] = dedup.phash(rgb)
    canon = dedup.group_duplicates(hashes, cfg.phash_hamming_dup_below)
    seen, deduped = set(), []
    for f in gated:
        rep = canon[f.content_hash]
        if rep in seen:
            continue
        seen.add(rep)
        deduped.append(f)
    _log("dedup", len(deduped), len(gated))

    ts_of = {f.content_hash: cache.image_ts(f.content_hash).value for f in deduped}
    ts_of = {k: v for k, v in ts_of.items() if v is not None}
    winners = bucketing.select_best_per_bucket(
        [f for f in deduped if f.content_hash in ts_of], ts_of, cfg)
    _log("bucket-winners", len(winners), len(deduped))

    cache.update_faces(winners)
    cache.close()
    return winners


# ---------------------------------------------------------------- render
def render_final(cfg: Config, rejects: set[str] | None = None) -> dict:
    cache = Cache(cfg.cache_db)
    winners = cache.all_faces(only_selected=True)
    rejects = rejects or set()
    winners = [f for f in winners if f.content_hash not in rejects]

    ts_of = {f.content_hash: cache.image_ts(f.content_hash) for f in winners}
    winners.sort(key=lambda f: ts_of[f.content_hash].value)

    # First pass: estimate alignment parameters (lightweight)
    params_seq = []
    for f in tqdm(winners, desc="Estimating alignment", unit="frame", disable=len(winners) < 20):
        rgb = ingest.decode(Path(cache.image_path(f.content_hash)))
        params_seq.append(align_mod.estimate(f, cfg))

    # Smooth parameters across the sequence
    params_seq = smoothing.smooth_params(params_seq, cfg.smooth_window)

    # Second pass: decode, warp, build frames (streaming - don't accumulate raw images)
    frames, dates, confident, rows = [], [], [], []
    for f, p in tqdm(list(zip(winners, params_seq)), desc="Warping frames", unit="frame", disable=len(winners) < 20):
        rgb = ingest.decode(Path(cache.image_path(f.content_hash)))
        frames.append(align_mod.warp(rgb, p, cfg))
        ts = ts_of[f.content_hash]
        dates.append(ts.value)
        confident.append(ts.confidence == "high")
        rows.append({
            "content_hash": f.content_hash, "path": cache.image_path(f.content_hash),
            "date": ts.value.isoformat() if ts.value else None,
            "date_source": ts.source.name, "date_confidence": ts.confidence,
            "bucket": f.bucket, "composite": f.composite,
            "yaw": f.yaw, "pitch": f.pitch, "neutral_penalty": f.neutral_penalty,
            "id_sim": f.id_sim, "id_margin": f.id_margin,
        })

    if cfg.photometric_mode != "none" and frames:
        ref = photometric.choose_reference(frames)
        frames = [photometric.normalize(fr, ref, cfg.photometric_mode) for fr in frames]

    result = render.render_video(frames, dates, confident, rows, cfg)
    _log("render", len(frames), len(winners))
    cache.close()
    return result


def _crop(rgb: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    return rgb[y1:y2, x1:x2]

"""Every tunable in one place.

Design intent: the pipeline has ~15 thresholds and each one silently
starves the funnel if set wrong. Centralizing them means you tune in one
file and re-run the cheap downstream stages against the cached inference,
instead of hunting magic numbers across modules.

None of these defaults are gospel. The identity thresholds in particular
MUST be recalibrated against your own labeled pairs (see identity.py and
the calibrate command); the values here are placeholders chosen to be
conservative, not correct-for-you.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Config:
    # --- paths ---
    cache_db: str = "cache/facelapse.db"          # SQLite: image + face records, embeddings, scores
    seeds_path: str = "cache/seeds.json"           # hand-confirmed reference photos of you
    out_dir: str = "out"                           # renders, manifest, review page

    # --- ingest ---
    image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")

    # --- identity gate (RECALIBRATE per identity.py.calibrate) ---
    # t_abs: minimum cosine similarity for "this could be me at all".
    # margin: required gap between best and second-best face in an image.
    # Both are model- and normalization-dependent. Do not trust these numbers.
    id_t_abs: float = 0.34          # VERIFY: derive from your you-vs-you distribution
    id_margin: float = 0.10         # VERIFY: derive from your you-vs-relatives distribution
    id_reference_k: int = 3         # compare against k temporally-nearest references, take max

    # --- pose filter (roll is corrected in alignment, NOT filtered here) ---
    max_abs_yaw_deg: float = 20.0
    max_abs_pitch_deg: float = 20.0

    # --- neutral-expression gate ---
    # Lower penalty = more neutral. Gate is generous on purpose; the review
    # step catches leakers. AU weights apply only if py-feat is installed.
    neutral_max_penalty: float = 1.0
    au_weights: dict = field(default_factory=lambda: {
        "AU12": 1.0,   # lip corner puller (smile)  -- dominant term
        "AU06": 0.8,   # cheek raiser (Duchenne)
        "AU25": 0.6,   # lips part
        "AU26": 0.5,   # jaw drop
        "AU01": 0.3, "AU02": 0.3, "AU04": 0.3,  # brow movement
    })
    # geometric backstops (fire regardless of py-feat availability)
    ear_closed_below: float = 0.15   # eye aspect ratio below this = blink/closed, reject
    mar_open_above: float = 0.55     # mouth aspect ratio above this = open mouth, reject

    # --- quality / face size ---
    min_interocular_px: float = 40.0   # faces smaller than this lack real detail, reject

    # --- temporal bucketing ---
    bucket_grain: str = "week"         # "day" | "week" | "month"

    # --- dedup ---
    phash_hamming_dup_below: int = 6   # <= this many bits differ => near-duplicate

    # --- alignment (canonical frame) ---
    canvas_w: int = 720
    canvas_h: int = 900
    # eye centers land at these canonical pixel positions -> fixes roll/scale/pos
    left_eye_xy: tuple[float, float] = (260.0, 360.0)
    right_eye_xy: tuple[float, float] = (460.0, 360.0)
    use_nasion: bool = True            # add nose-bridge point for least-squares stability

    # --- photometric ---
    photometric_mode: str = "hist_match"   # "none" | "exposure" | "hist_match"

    # --- temporal parameter smoothing ---
    smooth_window: int = 3             # odd moving-average window over transform params; 1 disables

    # --- render ---
    fps: float = 6.0
    crossfade_frames: int = 0          # 0 = honest hard cuts
    timeline_band_frac: float = 0.14   # lower-third band height as fraction of canvas_h
    draw_timeline: bool = True
    date_range_percentiles: tuple[float, float] = (1.0, 99.0)  # clamp bar scale vs outliers
    font_path: str | None = None       # None => auto-locate DejaVuSans, else PIL default

    @staticmethod
    def load(path: str | None) -> "Config":
        if not path:
            return Config()
        data = json.loads(Path(path).read_text())
        return Config(**data)

    def dump(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=list))

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

    # --- composite score weights (bucketing selection) ---
    score_weight_frontality: float = 0.30   # pose proximity minimizes swivel
    score_weight_neutral: float = 0.25      # neutrality
    score_weight_sharpness: float = 0.20    # sharpness
    score_weight_resolution: float = 0.15   # face resolution
    score_weight_identity: float = 0.05     # identity confidence (light tiebreak)
    score_weight_eyes_open: float = 0.05    # eyes open
    resolution_scale_factor: float = 3.0    # min_interocular_px * this for normalization

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
        cfg = Config(**data)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        """Validate config parameters are in valid ranges."""
        errors = []

        # Identity thresholds
        if not 0.0 <= self.id_t_abs <= 1.0:
            errors.append(f"id_t_abs must be in [0,1], got {self.id_t_abs}")
        if not 0.0 <= self.id_margin <= 1.0:
            errors.append(f"id_margin must be in [0,1], got {self.id_margin}")
        if self.id_reference_k < 1:
            errors.append(f"id_reference_k must be >= 1, got {self.id_reference_k}")

        # Pose
        if not 0.0 <= self.max_abs_yaw_deg <= 90.0:
            errors.append(f"max_abs_yaw_deg must be in [0,90], got {self.max_abs_yaw_deg}")
        if not 0.0 <= self.max_abs_pitch_deg <= 90.0:
            errors.append(f"max_abs_pitch_deg must be in [0,90], got {self.max_abs_pitch_deg}")

        # Neutral
        if self.neutral_max_penalty < 0.0:
            errors.append(f"neutral_max_penalty must be >= 0, got {self.neutral_max_penalty}")
        if not 0.0 <= self.ear_closed_below <= 1.0:
            errors.append(f"ear_closed_below must be in [0,1], got {self.ear_closed_below}")
        if not 0.0 <= self.mar_open_above <= 2.0:
            errors.append(f"mar_open_above must be in [0,2], got {self.mar_open_above}")

        # Quality
        if self.min_interocular_px < 0.0:
            errors.append(f"min_interocular_px must be >= 0, got {self.min_interocular_px}")

        # Bucketing
        if self.bucket_grain not in ("day", "week", "month"):
            errors.append(f"bucket_grain must be 'day'/'week'/'month', got {self.bucket_grain}")

        # Dedup
        if not 0 <= self.phash_hamming_dup_below <= 64:
            errors.append(f"phash_hamming_dup_below must be in [0,64], got {self.phash_hamming_dup_below}")

        # Canvas
        if self.canvas_w <= 0 or self.canvas_h <= 0:
            errors.append(f"canvas dimensions must be > 0, got {self.canvas_w}x{self.canvas_h}")

        # Photometric
        if self.photometric_mode not in ("none", "exposure", "hist_match"):
            errors.append(f"photometric_mode must be 'none'/'exposure'/'hist_match', got {self.photometric_mode}")

        # Render
        if self.fps <= 0.0:
            errors.append(f"fps must be > 0, got {self.fps}")
        if self.crossfade_frames < 0:
            errors.append(f"crossfade_frames must be >= 0, got {self.crossfade_frames}")
        if not 0.0 <= self.timeline_band_frac <= 1.0:
            errors.append(f"timeline_band_frac must be in [0,1], got {self.timeline_band_frac}")

        # Composite score weights should sum to ~1.0 for interpretability
        weight_sum = (self.score_weight_frontality + self.score_weight_neutral +
                      self.score_weight_sharpness + self.score_weight_resolution +
                      self.score_weight_identity + self.score_weight_eyes_open)
        if not 0.9 <= weight_sum <= 1.1:
            errors.append(f"Composite score weights should sum to ~1.0, got {weight_sum:.3f}")

        if errors:
            raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))

    def dump(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=list))

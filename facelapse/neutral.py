"""Neutral-expression scoring. Do NOT trust an emotion classifier's 'neutral'
label. Measure the specific muscle movements you want absent, via Action
Units, and back them with independent geometric checks that fail differently.

neutral_penalty = weighted AU activation (if py-feat available)
                + geometric penalties (mouth open, smile, eyes closed)

Lower is more neutral. The AU term is skipped gracefully if py-feat is not
installed; the geometric term always runs from MediaPipe dense landmarks.

Even combined, precision is not high enough to render blind: this produces a
ranked shortlist, and the review step makes the final cut. That is by design.
"""
from __future__ import annotations

import numpy as np

from . import landmarks as L
from .records import Face


class AUScorer:
    """py-feat Action Unit detector. Optional; import guarded."""

    def __init__(self):
        try:
            from feat import Detector  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "py-feat not installed. Neutral scoring will fall back to "
                "geometry-only. `pip install -r requirements-ml.txt` to enable AUs."
            ) from e
        self.detector = Detector()

    def au_activations(self, rgb_crop: np.ndarray) -> dict[str, float]:
        """Return {AU code: activation in [0,1]} for the crop. Wrapping only;
        exact py-feat return schema is version-dependent -- VERIFY columns."""
        import tempfile
        import imageio.v2 as imageio
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
            imageio.imwrite(tf.name, rgb_crop)
            res = self.detector.detect_image(tf.name)
        aus = {}
        for col in res.columns:
            if isinstance(col, str) and col.upper().startswith("AU"):
                try:
                    aus[col.upper().replace("AU", "AU").split("_")[0]] = float(res[col].iloc[0])
                except Exception:
                    continue
        return aus


def au_penalty(aus: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(weights.get(k, 0.0) * v for k, v in aus.items()))


def geometric_penalty(dense: np.ndarray, cfg) -> tuple[float, float, float]:
    """Return (penalty, ear, mar) from dense landmarks. Penalizes open mouth,
    smile geometry, and closed/blinking eyes."""
    ear = L.eye_aspect_ratio(dense)
    mar = L.mouth_aspect_ratio(dense)

    pen = 0.0
    # eyes closed / mid-blink: as ruinous as a smile in a single frame
    if ear < cfg.ear_closed_below:
        pen += 1.0
    # mouth open / mid-speech
    if mar > cfg.mar_open_above:
        pen += 0.6
    # smile heuristic: mouth corners raised relative to the eye line.
    # corners above the mouth-center vertical => upturned => smiling.
    corners_y = 0.5 * (dense[L.MOUTH["left"]][1] + dense[L.MOUTH["right"]][1])
    center_y = 0.5 * (dense[L.MOUTH["top"]][1] + dense[L.MOUTH["bottom"]][1])
    # image y grows downward; corners higher (smaller y) than center => smile
    smile = max(0.0, center_y - corners_y)
    mouth_w = np.linalg.norm(dense[L.MOUTH["left"]] - dense[L.MOUTH["right"]]) + 1e-6
    pen += 1.2 * (smile / mouth_w)
    return float(pen), float(ear), float(mar)


def score_face(
    f: Face, cfg, au_scorer: "AUScorer | None" = None,
    rgb_crop: np.ndarray | None = None,
) -> None:
    """Populate f.neutral_penalty, f.ear, f.mar. Requires dense landmarks for
    the geometric term; AU term added only if scorer+crop provided."""
    if f.dense is None:
        # no dense landmarks -> cannot judge neutrality; fail loud via None
        f.neutral_penalty = None
        return
    geo, ear, mar = geometric_penalty(f.dense, cfg)
    f.ear, f.mar = ear, mar
    total = geo
    if au_scorer is not None and rgb_crop is not None:
        try:
            aus = au_scorer.au_activations(rgb_crop)
            total += au_penalty(aus, cfg.au_weights)
        except Exception:
            pass   # AU stage optional; geometry already gives a usable signal
    f.neutral_penalty = total

"""Identity is a rank-with-margin problem, not a per-face threshold problem.

For each image with multiple faces:
  1. score every face's cosine similarity against the TEMPORALLY NEAREST
     reference embeddings (age-matched: a 2016 face is judged against
     2016-era you, not current you -- ArcFace is only partially age-invariant).
  2. take the top face. Require it to clear an absolute floor T_abs
     ("looks like me at all") AND to beat the runner-up by a margin M
     ("looks meaningfully MORE like me than anyone else here").

Face count never enters the decision. The margin is what rejects lookalike
relatives; a crowded photo where you dominate is fine, a two-person photo
where you and a sibling are a coin-flip apart is not.

Outcomes per image:
  - no face clears T_abs                         -> not you / unusable  (drop)
  - one face clears T_abs, margin >= M           -> confident hit       (take it)
  - top clears T_abs but margin < M              -> ambiguous           (review/drop)
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import Enum

import numpy as np

from .records import Face


class IdOutcome(Enum):
    NONE = "none"
    CONFIDENT = "confident"
    AMBIGUOUS = "ambiguous"
    PROMOTED = "promoted"  # ambiguous, then confirmed by hand on the review page


class ReferenceChain:
    """Holds seed embeddings with their dates and answers 'how much does this
    embedding look like age-matched me'."""

    def __init__(self, seeds: list[tuple[str, datetime | None, np.ndarray]], k: int = 3):
        self.k = k
        self.embeds = np.stack([e for _, _, e in seeds]) if seeds else np.zeros((0, 512))
        # seconds since epoch; None dates get NaN and are only used as fallback
        self.times = np.array([(_epoch(t) if t else np.nan) for _, t, _ in seeds], dtype=np.float64)

    def similarity(self, emb: np.ndarray, when: datetime | None) -> float:
        """Max cosine similarity against the k temporally-nearest references.
        Embeddings are L2-normalized, so dot product == cosine."""
        if self.embeds.shape[0] == 0:
            raise RuntimeError("No reference seeds. Run `facelapse seed` first.")
        sims = self.embeds @ emb  # (n_refs,)
        if when is not None and np.isfinite(self.times).any():
            dt = np.abs(self.times - _epoch(when))
            dt = np.where(np.isnan(dt), np.inf, dt)
            nearest = np.argsort(dt)[: self.k]
            return float(np.max(sims[nearest]))
        # no usable date -> compare against all references
        return float(np.max(sims))


def resolve_image(
    faces: list[Face],
    when: datetime | None,
    chain: ReferenceChain,
    t_abs: float,
    margin: float,
) -> tuple[IdOutcome, Face | None]:
    """Score all faces in one image; apply floor + margin. Mutates id_sim /
    id_margin on the faces for later inspection."""
    if not faces:
        return IdOutcome.NONE, None

    sims = [chain.similarity(f.embedding, when) for f in faces]
    order = np.argsort(sims)[::-1]
    top_i = int(order[0])
    top_sim = sims[top_i]
    runner = sims[int(order[1])] if len(sims) > 1 else -1.0
    gap = top_sim - runner

    for f, s in zip(faces, sims):
        f.id_sim = float(s)
    faces[top_i].id_margin = float(gap)

    if top_sim < t_abs:
        return IdOutcome.NONE, None
    if gap < margin:
        return IdOutcome.AMBIGUOUS, faces[top_i]
    faces[top_i].is_self = True
    return IdOutcome.CONFIDENT, faces[top_i]


def calibrate(
    triples: Sequence[tuple[float, float, bool]], target_far: float = 0.01
) -> dict[str, float]:
    """Pick id_margin from YOUR labeled multi-face images instead of a
    borrowed benchmark number.

    Each triple is (top_sim, runner_up_sim, top_is_me) for one image: the
    margin is applied within an image, so it is calibrated on the same
    quantity. Returns the smallest margin whose false-accept rate (images
    where the top face is someone else, yet the gap clears the margin) is at
    most target_far, with the resulting true-accept rate on images where the
    top face is you.
    """
    gaps = np.array([top - runner for top, runner, _ in triples], dtype=np.float64)
    is_me = np.array([me for _, _, me in triples], dtype=bool)
    neg = np.sort(gaps[~is_me])[::-1]
    pos = gaps[is_me]
    allowed = int(np.floor(target_far * neg.size + 1e-9))
    margin = 0.0 if allowed >= neg.size else float(np.nextafter(neg[allowed], np.inf))
    far = float(np.mean(neg >= margin)) if neg.size else 0.0
    tar = float(np.mean(pos >= margin)) if pos.size else 0.0
    return {
        "margin": margin,
        "false_accept_rate": far,
        "true_accept_rate": tar,
        "n_me_on_top": float(pos.size),
        "n_other_on_top": float(neg.size),
    }


def _epoch(dt: datetime) -> float:
    return dt.timestamp()

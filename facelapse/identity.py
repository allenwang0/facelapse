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

from datetime import datetime
from enum import Enum

import numpy as np

from .records import Face


class IdOutcome(Enum):
    NONE = "none"
    CONFIDENT = "confident"
    AMBIGUOUS = "ambiguous"


class ReferenceChain:
    """Holds seed embeddings with their dates and answers 'how much does this
    embedding look like age-matched me'."""

    def __init__(self, seeds: list[tuple[str, datetime | None, np.ndarray]], k: int = 3):
        self.k = k
        self.embeds = np.stack([e for _, _, e in seeds]) if seeds else np.zeros((0, 512))
        # seconds since epoch; None dates get NaN and are only used as fallback
        self.times = np.array(
            [(_epoch(t) if t else np.nan) for _, t, _ in seeds], dtype=np.float64
        )

    def similarity(self, emb: np.ndarray, when: datetime | None) -> float:
        """Max cosine similarity against the k temporally-nearest references.
        Embeddings are L2-normalized, so dot product == cosine."""
        if self.embeds.shape[0] == 0:
            raise RuntimeError("No reference seeds. Run `facelapse seed` first.")
        sims = self.embeds @ emb                       # (n_refs,)
        if when is not None and np.isfinite(self.times).any():
            dt = np.abs(self.times - _epoch(when))
            dt = np.where(np.isnan(dt), np.inf, dt)
            nearest = np.argsort(dt)[: self.k]
            return float(np.max(sims[nearest]))
        # no usable date -> compare against all references
        return float(np.max(sims))


def resolve_image(
    faces: list[Face], when: datetime | None, chain: ReferenceChain,
    t_abs: float, margin: float,
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
    same_pairs: list[tuple[np.ndarray, np.ndarray]],
    diff_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Derive defensible thresholds from YOUR labeled pairs instead of a
    borrowed benchmark number.

    same_pairs: (you, you) across different years -> lower tail tells you how
                low a TRUE match legitimately scores once aged.
    diff_pairs: (you, confusable-other) -> upper tail tells you how high a
                FALSE match climbs.

    Returns suggested t_abs and margin, and flags the overlap region that
    belongs in human review.
    """
    same = np.array([float(a @ b) for a, b in same_pairs]) if same_pairs else np.array([])
    diff = np.array([float(a @ b) for a, b in diff_pairs]) if diff_pairs else np.array([])
    out: dict = {}
    if same.size:
        out["same_min"] = float(same.min())
        out["same_p05"] = float(np.percentile(same, 5))
        out["same_median"] = float(np.median(same))
        # floor just below the true-match lower tail so aged frames survive
        out["suggested_t_abs"] = round(float(np.percentile(same, 5)) - 0.02, 3)
    if diff.size:
        out["diff_max"] = float(diff.max())
        out["diff_p95"] = float(np.percentile(diff, 95))
        # margin wide enough to clear the false-match upper tail
        out["suggested_margin"] = round(max(0.05, float(np.percentile(diff, 95))
                                            - out.get("same_p05", 0.0)), 3)
    if same.size and diff.size:
        overlap = max(0.0, float(diff.max()) - float(same.min()))
        out["overlap"] = overlap
        out["overlap_warning"] = (
            "Non-trivial overlap: pairs in this band are irreducibly ambiguous "
            "and must go to human review." if overlap > 0.03 else "Clean separation."
        )
    return out


def _epoch(dt: datetime) -> float:
    return dt.timestamp()

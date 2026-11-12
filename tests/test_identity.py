"""resolve_image edge cases: the identity gate decides which face is you,
so its boundaries must hold exactly."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from facelapse.identity import IdOutcome, ReferenceChain, resolve_image
from tests.helpers import embedding_with_similarity, make_face, unit

ME = unit(np.random.default_rng(1).normal(size=512))
CHAIN = ReferenceChain([("s", datetime(2020, 1, 1), ME)])


def _faces(*sims):
    return [
        make_face(face_id=i, embedding=embedding_with_similarity(ME, s, i))
        for i, s in enumerate(sims)
    ]


def test_single_face_above_floor_is_confident():
    outcome, chosen = resolve_image(_faces(0.6), None, CHAIN, 0.34, 0.10)
    assert outcome == IdOutcome.CONFIDENT and chosen.is_self


def test_single_face_below_floor_is_none():
    assert resolve_image(_faces(0.2), None, CHAIN, 0.34, 0.10)[0] == IdOutcome.NONE


def test_exact_tie_is_ambiguous_and_marks_nobody():
    faces = _faces(0.7, 0.7)
    faces[1].embedding = faces[0].embedding.copy()
    outcome, _ = resolve_image(faces, None, CHAIN, 0.34, 0.10)
    assert outcome == IdOutcome.AMBIGUOUS
    assert not any(f.is_self for f in faces)


def test_margin_is_measured_against_the_runner_up():
    faces = _faces(0.5, 0.8, 0.6)
    outcome, chosen = resolve_image(faces, None, CHAIN, 0.34, 0.10)
    assert outcome == IdOutcome.CONFIDENT and chosen.face_id == 1
    assert chosen.id_margin == pytest.approx(faces[1].id_sim - faces[2].id_sim)


def test_no_seeds_fails_loud():
    with pytest.raises(RuntimeError, match="seed"):
        resolve_image(_faces(0.9), None, ReferenceChain([]), 0.34, 0.10)


def test_no_faces_is_none():
    assert resolve_image([], None, CHAIN, 0.34, 0.10) == (IdOutcome.NONE, None)

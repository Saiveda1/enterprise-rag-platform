"""Reciprocal Rank Fusion correctness."""
from __future__ import annotations

import pytest

from ragplatform.retrieval import Hit, reciprocal_rank_fusion


def test_rrf_exact_scores():
    # doc 1 ranks #1 and #2; doc 2 ranks #2 and #1; doc 3 only in list A at #3.
    a = [Hit(1, 0.9), Hit(2, 0.8), Hit(3, 0.7)]
    b = [Hit(2, 0.95), Hit(1, 0.6)]
    fused = reciprocal_rank_fusion([a, b], k=3, rrf_k=60)
    scores = {h.idx: h.score for h in fused}
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[3] == pytest.approx(1 / 63)


def test_rrf_rewards_agreement():
    # A doc ranked #2 in BOTH lists should beat docs ranked #1 in only one list.
    # RRF(rrf_k=1): shared doc = 1/3 + 1/3 = 0.667 > single-list top = 1/2 = 0.5.
    a = [Hit(1, 1.0), Hit(99, 0.5)]  # doc 1 top of A, doc 99 second
    b = [Hit(2, 1.0), Hit(99, 0.5)]  # doc 2 top of B, doc 99 second
    fused = reciprocal_rank_fusion([a, b], k=4, rrf_k=1)
    assert fused[0].idx == 99
    scores = {h.idx: h.score for h in fused}
    assert scores[99] == pytest.approx(1 / 3 + 1 / 3)
    assert scores[1] == pytest.approx(1 / 2)


def test_rrf_sorted_descending():
    a = [Hit(i, 0.0) for i in range(5)]
    b = [Hit(i, 0.0) for i in [4, 3, 2, 1, 0]]
    fused = reciprocal_rank_fusion([a, b], k=5)
    scores = [h.score for h in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_respects_k():
    a = [Hit(i, 0.0) for i in range(10)]
    fused = reciprocal_rank_fusion([a], k=3)
    assert len(fused) == 3

"""Ranking + groundedness metric correctness."""
from __future__ import annotations

import math

import pytest

from ragplatform.metrics import (
    groundedness,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    ranked = [5, 2, 9, 1, 7]
    assert recall_at_k(ranked, {9}, 3) == 1.0
    assert recall_at_k(ranked, {9}, 2) == 0.0
    assert recall_at_k(ranked, {2, 7}, 5) == 1.0
    assert recall_at_k(ranked, {2, 7}, 2) == 0.5
    assert recall_at_k(ranked, set(), 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([5, 2, 9], {9}) == pytest.approx(1 / 3)
    assert reciprocal_rank([9, 2, 5], {9}) == 1.0
    assert reciprocal_rank([1, 2, 3], {99}) == 0.0


def test_ndcg_single_relevant():
    # Gold at rank 3 -> DCG = 1/log2(4); IDCG = 1/log2(2) = 1.
    val = ndcg_at_k([1, 2, 9, 4], {9}, 10)
    assert val == pytest.approx(1 / math.log2(4))
    # Gold at rank 1 -> perfect.
    assert ndcg_at_k([9, 1, 2], {9}, 10) == pytest.approx(1.0)


def test_ndcg_multiple_relevant_ideal_is_one():
    # All relevant docs at the top -> nDCG == 1.
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 10) == pytest.approx(1.0)
    # Same set but reversed relevance placement is still ideal here.
    val = ndcg_at_k([1, 9, 2], {1, 2}, 10)
    ideal = 1 + 1 / math.log2(3)
    got = 1 + 1 / math.log2(4)
    assert val == pytest.approx(got / ideal)


def test_groundedness():
    cited = ["the token rotation policy requires ninety days"]
    # Fully supported answer.
    assert groundedness("token rotation policy", cited) == 1.0
    # Half the answer tokens are unsupported.
    val = groundedness("token hallucinated", cited)
    assert val == pytest.approx(0.5)
    assert groundedness("", cited) == 0.0

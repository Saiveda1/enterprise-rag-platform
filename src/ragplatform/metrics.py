"""Retrieval + generation quality metrics with honest, testable math.

All ranking metrics take a ranked list of chunk indices and a set of gold
(relevant) indices. Binary relevance is used throughout, but the nDCG
implementation is graded-relevance ready.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(ranked: Sequence[int], gold: Iterable[int], k: int) -> float:
    """Fraction of gold items appearing in the top-k (1.0 / 0.0 for single gold)."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    top = set(ranked[:k])
    return len(top & gold_set) / len(gold_set)


def reciprocal_rank(ranked: Sequence[int], gold: Iterable[int]) -> float:
    """1 / rank of the first relevant item (rank is 1-based); 0 if none."""
    gold_set = set(gold)
    for rank, idx in enumerate(ranked, start=1):
        if idx in gold_set:
            return 1.0 / rank
    return 0.0


def _dcg(relevances: Sequence[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked: Sequence[int], gold: Iterable[int], k: int) -> float:
    """Normalized discounted cumulative gain at k (binary relevance)."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    rels = [1.0 if idx in gold_set else 0.0 for idx in ranked[:k]]
    dcg = _dcg(rels)
    ideal = _dcg([1.0] * min(len(gold_set), k))
    return dcg / ideal if ideal > 0 else 0.0


def groundedness(answer: str, cited_texts: Sequence[str]) -> float:
    """Proxy for faithfulness: fraction of answer tokens supported by citations.

    An extractive answer stitched from the cited passages scores ~1.0; any token
    not found in the cited context lowers the score, flagging hallucinated spans.
    """
    from .retrieval import tokenize

    ans_tokens = tokenize(answer)
    if not ans_tokens:
        return 0.0
    support: set[str] = set()
    for t in cited_texts:
        support |= set(tokenize(t))
    supported = sum(1 for tok in ans_tokens if tok in support)
    return supported / len(ans_tokens)


def aggregate(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

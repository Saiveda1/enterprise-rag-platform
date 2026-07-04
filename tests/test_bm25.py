"""Assertions on the Okapi BM25 implementation — math, not smoke."""
from __future__ import annotations

import math

import pytest

from ragplatform.retrieval import BM25Retriever, tokenize


def _manual_bm25(bm25: BM25Retriever, query_terms, doc_idx, texts):
    """Reference BM25 computed independently from raw texts."""
    toks = tokenize(texts[doc_idx])
    dl = len(toks)
    tf_map = {}
    for t in toks:
        tf_map[t] = tf_map.get(t, 0) + 1
    score = 0.0
    for term in set(query_terms):
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        idf = bm25.idf(term)
        denom = tf + bm25.k1 * (1 - bm25.b + bm25.b * dl / bm25.avgdl)
        score += idf * (tf * (bm25.k1 + 1)) / denom
    return score


def test_idf_formula():
    texts = ["alpha beta", "alpha gamma", "delta"]
    bm = BM25Retriever().fit(texts)
    # 'alpha' appears in 2 of 3 docs.
    expected = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    assert bm.idf("alpha") == pytest.approx(expected)
    # unseen term -> 0 idf.
    assert bm.idf("zzz") == 0.0


def test_score_doc_matches_reference():
    texts = [
        "the quick brown fox jumps",
        "the lazy dog sleeps all day",
        "quick foxes are quick and brown",
        "completely unrelated content here",
    ]
    bm = BM25Retriever(k1=1.5, b=0.75).fit(texts)
    query = tokenize("quick brown fox")
    for idx in range(len(texts)):
        got = bm.score_doc(query, idx)
        exp = _manual_bm25(bm, query, idx, texts)
        assert got == pytest.approx(exp, rel=1e-9), idx


def test_search_ranks_relevant_first():
    texts = [
        "machine learning models train on data",
        "the cat sat quietly on the warm mat",
        "deep learning is a subset of machine learning",
    ]
    bm = BM25Retriever().fit(texts)
    hits = bm.search("machine learning", k=3)
    assert hits[0].idx in (0, 2)
    # The unrelated cat doc must rank last (or be absent).
    ranked = [h.idx for h in hits]
    assert ranked.index(1) == len(ranked) - 1 if 1 in ranked else True


def test_search_scores_descending():
    texts = ["alpha beta gamma", "alpha alpha beta", "gamma delta", "beta beta beta"]
    bm = BM25Retriever().fit(texts)
    hits = bm.search("alpha beta", k=4)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_longer_doc_penalized_by_length_norm():
    # Same term frequency, but one doc is padded with unrelated tokens.
    texts = ["target target", "target target " + " ".join(["pad"] * 40)]
    bm = BM25Retriever().fit(texts)
    q = tokenize("target")
    assert bm.score_doc(q, 0) > bm.score_doc(q, 1)

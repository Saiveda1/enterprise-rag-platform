"""Retrieval backends: Dense (cosine), Okapi BM25, and Hybrid (RRF fusion).

All retrievers share the ``Hit`` result shape and support optional multi-tenant
filtering via an ``allowed`` boolean mask over chunk indices, so a single index
can serve many tenants with hard isolation at query time.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer shared by BM25 and reranker features."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    idx: int
    score: float


# ---------------------------------------------------------------------------
# BM25 (Okapi) — implemented from scratch.
# ---------------------------------------------------------------------------
class BM25Retriever:
    """Okapi BM25 built on a precomputed sparse term-weight matrix.

    score(D, Q) = Σ_t IDF(t) · tf(t,D)·(k1+1) / (tf(t,D) + k1·(1 − b + b·|D|/avgdl))
    IDF(t)      = ln(1 + (N − n_t + 0.5) / (n_t + 0.5))

    Key observation: the per-(term, doc) BM25 weight depends only on ``tf`` and
    the document length — **not** on the query. So we materialize a CSR matrix
    ``W`` of shape (n_docs, vocab) once at fit time; a query is then a 0/1
    indicator over term columns and scoring is a single sparse matrix-vector
    product ``W @ q`` — exact BM25, but milliseconds instead of a Python scan over
    long postings lists. This is what lets retrieval stay fast at 10^5+ chunks.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._vocab: dict[str, int] = {}
        self._doc_tf: list[dict[str, int]] = []
        self._idf: dict[str, float] = {}
        self._W: csr_matrix | None = None  # (n_docs, vocab) BM25 weights
        self._doc_len: np.ndarray = np.zeros(0, dtype=np.int32)
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    def fit(self, texts: Sequence[str]) -> "BM25Retriever":
        doc_len: list[int] = []
        df: dict[str, int] = defaultdict(int)
        self._doc_tf = []
        self._vocab = {}
        for i, text in enumerate(texts):
            toks = tokenize(text)
            doc_len.append(len(toks))
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            self._doc_tf.append(dict(tf))  # enables O(query) score_doc
            for term, freq in tf.items():
                df[term] += 1
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)

        self.n_docs = len(doc_len)
        self._doc_len = np.asarray(doc_len, dtype=np.int32)
        self.avgdl = float(self._doc_len.mean()) if self.n_docs else 0.0
        n = self.n_docs
        for term, n_t in df.items():
            self._idf[term] = math.log(1.0 + (n - n_t + 0.5) / (n_t + 0.5))

        # Build the sparse BM25 weight matrix W[d, t].
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        k1 = self.k1
        for d, tf in enumerate(self._doc_tf):
            denom_len = k1 * (1.0 - self.b + self.b * doc_len[d] / (self.avgdl or 1.0))
            for term, f in tf.items():
                rows.append(d)
                cols.append(self._vocab[term])
                data.append(self._idf[term] * (f * (k1 + 1.0)) / (f + denom_len))
        self._W = csr_matrix(
            (np.asarray(data, dtype=np.float64), (rows, cols)),
            shape=(max(1, n), max(1, len(self._vocab))),
        )
        return self

    def idf(self, term: str) -> float:
        return self._idf.get(term, 0.0)

    def score_doc(self, query_terms: Sequence[str], doc_idx: int) -> float:
        """Exact BM25 score of one document — used by tests and the reranker."""
        dl = float(self._doc_len[doc_idx])
        denom_len = self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
        doc_tf = self._doc_tf[doc_idx]
        score = 0.0
        for term in set(query_terms):
            tf = doc_tf.get(term, 0)
            if tf == 0:
                continue
            score += self._idf[term] * (tf * (self.k1 + 1.0)) / (tf + denom_len)
        return score

    def search(
        self, query: str, k: int = 10, allowed: np.ndarray | None = None
    ) -> list[Hit]:
        assert self._W is not None
        cols = [self._vocab[t] for t in set(tokenize(query)) if t in self._vocab]
        if not cols:
            return []
        # q is a sparse column indicator; W @ q sums the BM25 weights of the
        # query's term-columns per document — exact BM25 scores.
        q = csc_matrix(
            (np.ones(len(cols)), (cols, np.zeros(len(cols), dtype=int))),
            shape=(self._W.shape[1], 1),
        )
        scores = np.asarray((self._W @ q).todense()).ravel()
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        nz = scores > 0 if allowed is None else np.isfinite(scores) & (scores > 0)
        pool = int(nz.sum())
        if pool == 0:
            return []
        k_eff = min(k, pool)
        cand = np.flatnonzero(nz)
        order = cand[np.argsort(-scores[cand])[:k_eff]]
        return [Hit(int(i), float(scores[i])) for i in order]


# ---------------------------------------------------------------------------
# Dense retrieval — cosine via sklearn NearestNeighbors (+ masked numpy path).
# ---------------------------------------------------------------------------
class DenseRetriever:
    """Cosine kNN over L2-normalized dense embeddings.

    Because embeddings are unit vectors, cosine similarity is exactly the dot
    product, so search is a single ``embeddings @ q`` matrix-vector product plus a
    partial top-k selection — an exact brute-force cosine kNN. A ``NearestNeighbors``
    index is also fitted and exposed as ``ann`` (for a drop-in ANN swap and to
    show scoring parity), but the numpy path is the hot path because it is ~10x
    faster here and supports masking for tenant isolation.
    """

    def __init__(self, n_neighbors: int = 50, build_ann: bool = True) -> None:
        self.n_neighbors = n_neighbors
        self.build_ann = build_ann
        self._emb: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.ann: NearestNeighbors | None = None

    def fit(self, embeddings: np.ndarray) -> "DenseRetriever":
        self._emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        if self.build_ann:
            n_fit = min(self.n_neighbors, max(1, self._emb.shape[0]))
            self.ann = NearestNeighbors(
                n_neighbors=n_fit, metric="cosine", algorithm="brute"
            )
            self.ann.fit(self._emb)
        return self

    def search(
        self, query_vec: np.ndarray, k: int = 10, allowed: np.ndarray | None = None
    ) -> list[Hit]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        sims = self._emb @ q  # cosine (unit vectors)
        if allowed is not None:
            sims = np.where(allowed, sims, -np.inf)
            pool = int(allowed.sum())
        else:
            pool = self._emb.shape[0]
        k_eff = min(k, pool)
        if k_eff <= 0:
            return []
        top = np.argpartition(-sims, k_eff - 1)[:k_eff]
        top = top[np.argsort(-sims[top])]
        return [Hit(int(i), float(sims[i])) for i in top]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion.
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]], k: int = 10, rrf_k: int = 60
) -> list[Hit]:
    """Fuse multiple ranked lists. RRF(d) = Σ_l 1/(rrf_k + rank_l(d)).

    ``rank`` is 1-based. Documents absent from a list contribute nothing from it.
    """
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.idx] += 1.0 / (rrf_k + rank)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [Hit(idx, s) for idx, s in ranked]


class HybridRetriever:
    """Dense + BM25 fused with Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        rrf_k: int = 60,
        candidate_k: int = 50,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        k: int = 10,
        allowed: np.ndarray | None = None,
    ) -> list[Hit]:
        dense_hits = self.dense.search(query_vec, k=self.candidate_k, allowed=allowed)
        bm25_hits = self.bm25.search(query, k=self.candidate_k, allowed=allowed)
        return reciprocal_rank_fusion([dense_hits, bm25_hits], k=k, rrf_k=self.rrf_k)

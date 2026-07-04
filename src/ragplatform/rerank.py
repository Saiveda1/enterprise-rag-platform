"""Lightweight cross-encoder-style reranker.

Instead of a heavyweight transformer cross-encoder (which we cannot download
offline), we score each (query, candidate) pair with a small feature vector and
a logistic-regression relevance model trained on synthetic labels. Features mix
lexical, semantic, and length signals — exactly what a learned reranker keys on:

    * fraction of query terms present in the candidate
    * Jaccard token overlap
    * IDF-weighted overlap (rare shared terms count more)
    * BM25 score of the pair
    * dense cosine similarity
    * length prior (deviation from the average document length)

The trained model reorders the fused candidate set, lifting the true passage
above lexically-similar distractors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .retrieval import BM25Retriever, Hit, tokenize

N_FEATURES = 6


@dataclass
class Reranker:
    """Feature-based learned reranker over retrieval candidates."""

    bm25: BM25Retriever
    embeddings: np.ndarray
    chunk_texts: Sequence[str]
    seed: int = 42

    def __post_init__(self) -> None:
        self._model: LogisticRegression | None = None
        self._tok_cache: dict[int, set[str]] = {}

    def _doc_tokens(self, idx: int) -> set[str]:
        cached = self._tok_cache.get(idx)
        if cached is None:
            cached = set(tokenize(self.chunk_texts[idx]))
            self._tok_cache[idx] = cached
        return cached

    def featurize(self, query: str, query_vec: np.ndarray, idx: int) -> np.ndarray:
        q_terms = tokenize(query)
        q_set = set(q_terms)
        d_set = self._doc_tokens(idx)
        shared = q_set & d_set
        union = q_set | d_set

        frac_present = len(shared) / max(1, len(q_set))
        jaccard = len(shared) / max(1, len(union))
        idf_overlap = sum(self.bm25.idf(t) for t in shared)
        bm25_score = self.bm25.score_doc(q_terms, idx)
        cosine = float(self.embeddings[idx] @ np.asarray(query_vec, dtype=np.float32))
        dl = float(self.bm25._doc_len[idx]) if self.bm25.n_docs else 0.0
        len_prior = -abs(dl - self.bm25.avgdl) / (self.bm25.avgdl or 1.0)

        return np.array(
            [frac_present, jaccard, idf_overlap, bm25_score, cosine, len_prior],
            dtype=np.float64,
        )

    def train(
        self,
        examples: Sequence[tuple[str, np.ndarray, int, Sequence[int]]],
    ) -> "Reranker":
        """Fit the relevance model.

        Each example is (query, query_vec, gold_idx, candidate_idxs). The gold
        chunk is the positive; the other candidates are hard negatives.
        """
        X: list[np.ndarray] = []
        y: list[int] = []
        for query, qvec, gold_idx, cand_idxs in examples:
            for idx in cand_idxs:
                X.append(self.featurize(query, qvec, idx))
                y.append(1 if idx == gold_idx else 0)
        Xa = np.vstack(X)
        ya = np.asarray(y)
        # Standardize features for stable LR coefficients.
        self._mu = Xa.mean(axis=0)
        self._sigma = Xa.std(axis=0) + 1e-9
        Xs = (Xa - self._mu) / self._sigma
        self._model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=self.seed
        )
        self._model.fit(Xs, ya)
        return self

    def rerank(
        self, query: str, query_vec: np.ndarray, candidates: Sequence[Hit], k: int = 10
    ) -> list[Hit]:
        if self._model is None:
            raise RuntimeError("Reranker must be train()ed before rerank().")
        if not candidates:
            return []
        feats = np.vstack(
            [self.featurize(query, query_vec, h.idx) for h in candidates]
        )
        feats = (feats - self._mu) / self._sigma
        proba = self._model.predict_proba(feats)[:, 1]
        order = np.argsort(-proba)[:k]
        return [Hit(candidates[i].idx, float(proba[i])) for i in order]

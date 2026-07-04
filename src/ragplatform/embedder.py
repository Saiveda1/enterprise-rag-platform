"""Offline, deterministic embeddings behind a clean interface.

``TfidfSvdEmbedder`` builds dense vectors by fitting a TF-IDF vectorizer and
reducing it with TruncatedSVD (latent semantic analysis). The SVD-reduced,
L2-normalized rows are treated as dense embeddings — cosine similarity in this
space captures lexical *and* co-occurrence structure, so paraphrased queries
still land near their source passage.

Everything is deterministic (fixed ``random_state``) and needs no network or
GPU, so a real transformer encoder can later drop in behind the ``Embedder``
protocol without touching the retrieval code.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@runtime_checkable
class Embedder(Protocol):
    """Interface any embedding backend must satisfy."""

    dim: int

    def fit(self, texts: Sequence[str]) -> "Embedder": ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class TfidfSvdEmbedder:
    """TF-IDF + TruncatedSVD (LSA) embeddings, L2-normalized to unit length."""

    def __init__(
        self,
        dim: int = 256,
        *,
        min_df: int = 2,
        max_features: int = 50_000,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            min_df=min_df,
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._svd: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfSvdEmbedder":
        tfidf = self._vectorizer.fit_transform(texts)
        # SVD dim must be < n_features.
        n_features = tfidf.shape[1]
        eff_dim = int(min(self.dim, max(2, n_features - 1)))
        self._svd = TruncatedSVD(
            n_components=eff_dim,
            algorithm="randomized",
            n_iter=2,
            random_state=self.seed,
        )
        self._svd.fit(tfidf)
        self.dim = eff_dim
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted or self._svd is None:
            raise RuntimeError("Embedder must be fit() before encode().")
        tfidf = self._vectorizer.transform(texts)
        reduced = self._svd.transform(tfidf).astype(np.float32)
        # L2-normalize so dot product == cosine similarity.
        return normalize(reduced, norm="l2", axis=1)

    @property
    def explained_variance(self) -> float:
        if self._svd is None:
            return 0.0
        return float(self._svd.explained_variance_ratio_.sum())

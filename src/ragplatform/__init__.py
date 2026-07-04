"""Enterprise RAG Platform — production-grade retrieval-augmented generation.

Offline, deterministic, and dependency-light. Every component is real:
synthetic corpus generation, TF-IDF+SVD embeddings, dense / BM25 / hybrid
retrieval, a feature-based reranker, extractive synthesis with citations,
and a full evaluation harness.
"""
from __future__ import annotations

__version__ = "1.0.0"

SEED = 42

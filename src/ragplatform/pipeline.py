"""End-to-end RAG pipeline: index build, retrieval, rerank, eval, isolation.

Ties the components together into an ``RagIndex`` and provides the evaluation
harness that compares BM25 / Dense / Hybrid / Hybrid+Rerank on a synthetic QA
set with known gold chunks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .corpus import Chunk, generate_documents, _chunk_document
from .embedder import Embedder, TfidfSvdEmbedder
from .metrics import aggregate, groundedness, ndcg_at_k, recall_at_k, reciprocal_rank
from .qa import QAItem, build_qa_set
from .rerank import Reranker
from .retrieval import (
    BM25Retriever,
    DenseRetriever,
    Hit,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from .synthesis import synthesize

METHODS = ["BM25", "Dense", "Hybrid", "Hybrid+Rerank"]


@dataclass
class RagIndex:
    chunks: list[Chunk]
    texts: list[str]
    tenants: np.ndarray
    doc_records: list[dict]
    embedder: Embedder
    embeddings: np.ndarray
    dense: DenseRetriever
    bm25: BM25Retriever
    hybrid: HybridRetriever
    reranker: Reranker | None = None
    _chunk_index: dict[str, int] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    def chunk_lookup(self, idx: int) -> dict:
        c = self.chunks[idx]
        return {"text": c.text, "chunk_id": c.chunk_id, "doc_id": c.doc_id}

    def tenant_mask(self, tenant_id: str) -> np.ndarray:
        return self.tenants == tenant_id

    def encode_query(self, query: str) -> np.ndarray:
        return self.embedder.encode([query])[0]


def build_index(
    n_docs: int,
    *,
    seed: int = 42,
    dim: int = 256,
    chunk_tokens: int = 40,
    overlap_tokens: int = 12,
    candidate_k: int = 50,
) -> RagIndex:
    """Materialize the corpus, embed it, and construct all retrievers."""
    chunks: list[Chunk] = []
    texts: list[str] = []
    tenants: list[str] = []
    doc_records: list[dict] = []

    for doc in generate_documents(n_docs, seed=seed):
        start = len(chunks)
        for c in _chunk_document(
            doc["doc_id"],
            doc["tenant_id"],
            doc["source"],
            doc["timestamp"],
            doc["sentences"],
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        ):
            chunks.append(c)
            texts.append(c.text)
            tenants.append(c.tenant_id)
        idxs = list(range(start, len(chunks)))
        doc_records.append(
            {
                "doc_id": doc["doc_id"],
                "tenant_id": doc["tenant_id"],
                "source": doc["source"],
                "meta": doc["meta"],
                "chunk_indices": idxs,
            }
        )

    tenants_arr = np.asarray(tenants)

    embedder = TfidfSvdEmbedder(dim=dim, seed=seed).fit(texts)
    embeddings = embedder.encode(texts)

    dense = DenseRetriever(n_neighbors=candidate_k).fit(embeddings)
    bm25 = BM25Retriever().fit(texts)
    hybrid = HybridRetriever(dense, bm25, candidate_k=candidate_k)

    return RagIndex(
        chunks=chunks,
        texts=texts,
        tenants=tenants_arr,
        doc_records=doc_records,
        embedder=embedder,
        embeddings=embeddings,
        dense=dense,
        bm25=bm25,
        hybrid=hybrid,
    )


def _ranked_idx(hits: list[Hit]) -> list[int]:
    return [h.idx for h in hits]


def _union(*rankings: list[Hit]) -> list[Hit]:
    """Deduplicated union of ranked hit lists (first occurrence wins)."""
    seen: dict[int, Hit] = {}
    for ranking in rankings:
        for h in ranking:
            if h.idx not in seen:
                seen[h.idx] = h
    return list(seen.values())


def _candidate_pool(
    index: RagIndex, query: str, qvec: np.ndarray, candidate_k: int
) -> list[Hit]:
    """Union of BM25 and Dense top-``candidate_k`` (deduped).

    Gives the reranker a broad, high-recall pool instead of the RRF-truncated
    list, so its ceiling is max(BM25, Dense) recall rather than Hybrid's.
    """
    bm25 = index.bm25.search(query, k=candidate_k)
    dense = index.dense.search(qvec, k=candidate_k)
    return _union(bm25, dense)


def train_reranker(
    index: RagIndex, train_items: list[QAItem], *, candidate_k: int = 50
) -> Reranker:
    """Train the reranker on hybrid candidates of the training QA items."""
    reranker = Reranker(
        bm25=index.bm25, embeddings=index.embeddings, chunk_texts=index.texts
    )
    examples = []
    for item in train_items:
        qvec = index.encode_query(item.question)
        cands = _candidate_pool(index, item.question, qvec, candidate_k)
        cand_idxs = _ranked_idx(cands)
        # Guarantee the gold appears as a positive even if retrieval missed it.
        if item.gold_chunk_idx not in cand_idxs:
            cand_idxs = cand_idxs + [item.gold_chunk_idx]
        examples.append((item.question, qvec, item.gold_chunk_idx, cand_idxs))
    reranker.train(examples)
    index.reranker = reranker
    return reranker


def evaluate(
    index: RagIndex,
    test_items: list[QAItem],
    *,
    ks=(1, 3, 5, 10),
    ndcg_k: int = 10,
    candidate_k: int = 50,
) -> dict:
    """Evaluate all four methods on the test QA set. Returns a results dict."""
    max_k = max(max(ks), ndcg_k)
    results: dict[str, dict] = {m: _empty_acc(ks) for m in METHODS}

    for item in test_items:
        qvec = index.encode_query(item.question)
        gold = {item.gold_chunk_idx}

        # Compute each arm once at candidate depth, then derive the rest.
        depth = max(candidate_k, max_k)
        bm25_deep = index.bm25.search(item.question, k=depth)
        dense_deep = index.dense.search(qvec, k=depth)
        bm25_cands = bm25_deep[:candidate_k]
        dense_cands = dense_deep[:candidate_k]

        hybrid_hits = reciprocal_rank_fusion(
            [bm25_cands, dense_cands], k=max_k, rrf_k=index.hybrid.rrf_k
        )
        method_hits = {
            "BM25": bm25_deep[:max_k],
            "Dense": dense_deep[:max_k],
            "Hybrid": hybrid_hits,
        }
        if index.reranker is not None:
            pool = _union(bm25_cands, dense_cands)
            reranked = index.reranker.rerank(item.question, qvec, pool, k=max_k)
            method_hits["Hybrid+Rerank"] = reranked

        for method, hits in method_hits.items():
            ranked = _ranked_idx(hits)
            acc = results[method]
            for k in ks:
                acc["recall"][k].append(recall_at_k(ranked, gold, k))
            acc["mrr"].append(reciprocal_rank(ranked, gold))
            acc["ndcg"].append(ndcg_at_k(ranked, gold, ndcg_k))
            # Groundedness of the extractive answer built from this ranking.
            ans = synthesize(item.question, hits[:3], index.chunk_lookup)
            acc["grounded"].append(groundedness(ans.text, ans.cited_texts))

    summary = {}
    for method, acc in results.items():
        if not acc["mrr"]:
            continue
        summary[method] = {
            "recall": {k: aggregate(acc["recall"][k]) for k in ks},
            "mrr": aggregate(acc["mrr"]),
            "ndcg": aggregate(acc["ndcg"]),
            "grounded": aggregate(acc["grounded"]),
            "n": len(acc["mrr"]),
        }
    return {"summary": summary, "ks": list(ks), "ndcg_k": ndcg_k}


def _empty_acc(ks) -> dict:
    return {
        "recall": {k: [] for k in ks},
        "mrr": [],
        "ndcg": [],
        "grounded": [],
    }


def measure_latency(
    index: RagIndex,
    items: list[QAItem],
    *,
    ks=(1, 5, 10, 20, 50),
    candidate_k: int = 50,
) -> dict[str, dict[int, float]]:
    """Mean per-query latency (ms) for each method across retrieval depth k."""
    out: dict[str, dict[int, float]] = {m: {} for m in METHODS}
    qvecs = [index.encode_query(it.question) for it in items]
    for k in ks:
        cand = max(k, candidate_k)
        timings = {m: 0.0 for m in METHODS}
        for it, qvec in zip(items, qvecs):
            t = time.perf_counter()
            index.bm25.search(it.question, k=k)
            timings["BM25"] += time.perf_counter() - t

            t = time.perf_counter()
            index.dense.search(qvec, k=k)
            timings["Dense"] += time.perf_counter() - t

            t = time.perf_counter()
            index.hybrid.search(it.question, qvec, k=cand)
            timings["Hybrid"] += time.perf_counter() - t

            if index.reranker is not None:
                # End-to-end: build the union candidate pool, then rerank it.
                t = time.perf_counter()
                pool = _candidate_pool(index, it.question, qvec, cand)
                index.reranker.rerank(it.question, qvec, pool, k=k)
                timings["Hybrid+Rerank"] += time.perf_counter() - t

        n = max(1, len(items))
        for m in METHODS:
            out[m][k] = 1000.0 * timings[m] / n
    return out


def split_qa(items: list[QAItem], *, train_frac: float = 0.5) -> tuple[list, list]:
    """Deterministic train/test split (items already shuffled by build_qa_set)."""
    cut = int(len(items) * train_frac)
    return items[:cut], items[cut:]


def demo_tenant_isolation(index: RagIndex, tenant_id: str, query: str) -> dict:
    """Return proof that a tenant-scoped query never leaks other tenants' chunks."""
    mask = index.tenant_mask(tenant_id)
    qvec = index.encode_query(query)
    dense_hits = index.dense.search(qvec, k=10, allowed=mask)
    bm25_hits = index.bm25.search(query, k=10, allowed=mask)
    leaked = [
        index.chunks[h.idx].tenant_id
        for h in (dense_hits + bm25_hits)
        if index.chunks[h.idx].tenant_id != tenant_id
    ]
    return {
        "tenant": tenant_id,
        "dense_hits": len(dense_hits),
        "bm25_hits": len(bm25_hits),
        "leaked": leaked,
        "isolated": len(leaked) == 0,
    }

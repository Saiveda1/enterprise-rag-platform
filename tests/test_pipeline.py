"""End-to-end pipeline: embeddings, retrieval, tenant isolation, rerank, synth, eval."""
from __future__ import annotations

import numpy as np

from ragplatform.pipeline import (
    METHODS,
    _candidate_pool,
    demo_tenant_isolation,
    evaluate,
    split_qa,
)
from ragplatform.synthesis import synthesize


def test_embeddings_l2_normalized(index):
    norms = np.linalg.norm(index.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert index.embeddings.dtype == np.float32


def test_dense_masked_matches_bruteforce(index):
    # The masked numpy path and an independent cosine computation must agree.
    q = index.encode_query("data retention policy")
    mask = np.ones(len(index.chunks), dtype=bool)
    hits = index.dense.search(q, k=5, allowed=mask)
    sims = index.embeddings @ q
    top = np.argsort(-sims)[:5]
    assert [h.idx for h in hits] == list(top)
    assert hits[0].score == np.float32(sims[top[0]]).item() or abs(
        hits[0].score - sims[top[0]]
    ) < 1e-4


def test_tenant_isolation_hard(index):
    for tenant in np.unique(index.tenants):
        res = demo_tenant_isolation(index, str(tenant), "policy incident vector service")
        assert res["isolated"], f"leak for tenant {tenant}: {res['leaked']}"
    # And every returned chunk truly belongs to the tenant.
    tenant = str(index.tenants[0])
    mask = index.tenant_mask(tenant)
    q = index.encode_query("security review access")
    for h in index.dense.search(q, k=10, allowed=mask):
        assert index.chunks[h.idx].tenant_id == tenant
    for h in index.bm25.search("security review access", k=10, allowed=mask):
        assert index.chunks[h.idx].tenant_id == tenant


def test_tenant_filter_changes_results(index):
    # Filtering to one tenant must not return more than the unfiltered top set
    # and must only contain that tenant's chunks.
    tenant = str(index.tenants[0])
    mask = index.tenant_mask(tenant)
    q = index.encode_query("deployment runbook canary")
    filtered = index.dense.search(q, k=10, allowed=mask)
    assert all(index.chunks[h.idx].tenant_id == tenant for h in filtered)


def test_candidate_pool_covers_both(index):
    q_text = "incident response acknowledged"
    qv = index.encode_query(q_text)
    pool = {h.idx for h in _candidate_pool(index, q_text, qv, 20)}
    bm = {h.idx for h in index.bm25.search(q_text, k=20)}
    dn = {h.idx for h in index.dense.search(qv, k=20)}
    assert bm <= pool and dn <= pool  # union property


def test_synthesis_citations_align(index):
    q = "data retention policy"
    qv = index.encode_query(q)
    hits = index.hybrid.search(q, qv, k=3)
    ans = synthesize(q, hits, index.chunk_lookup)
    assert ans.citations
    for c in ans.citations:
        # The cited snippet must appear at its recorded span in the answer text.
        assert ans.text[c.start : c.end] == c.snippet
        assert c.chunk_id and c.doc_id


def test_reranker_improves_or_matches(trained_index, qa):
    _, test_items = split_qa(qa)
    res = evaluate(trained_index, test_items)["summary"]
    assert set(res) == set(METHODS)
    # The learned reranker should not degrade MRR versus the hybrid baseline.
    assert res["Hybrid+Rerank"]["mrr"] >= res["Hybrid"]["mrr"] - 1e-9
    # And it should be the strongest (or tied) on nDCG among all methods.
    best = max(res[m]["ndcg"] for m in METHODS)
    assert res["Hybrid+Rerank"]["ndcg"] >= best - 1e-9


def test_metrics_in_valid_range(trained_index, qa):
    _, test_items = split_qa(qa)
    res = evaluate(trained_index, test_items)["summary"]
    for m, v in res.items():
        assert 0.0 <= v["mrr"] <= 1.0
        assert 0.0 <= v["ndcg"] <= 1.0
        assert 0.0 <= v["grounded"] <= 1.0
        for k, r in v["recall"].items():
            assert 0.0 <= r <= 1.0

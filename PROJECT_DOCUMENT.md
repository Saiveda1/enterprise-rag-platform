# Enterprise RAG Platform Project Document

**Prepared For:** Sai Veda  
**GitHub Publishing Account:** Nikeshk834  
**Repository Slug:** `01-enterprise-rag-platform`  
**Verified Test Count From Portfolio Index:** 28  

## Background

**Production-grade retrieval-augmented generation — offline, deterministic, and evaluated.**

Enterprise knowledge lives in policies, support tickets, and wiki pages scattered
across tenants. A useful assistant has to *find the right passage* before it can
answer, and it has to prove the answer came from a real source without leaking one
tenant's data to another. This project is the retrieval and evaluation core of
such a system, built end-to-end: a streaming corpus generator, an embedder behind
a swappable interface, three retrieval backends (BM25 / Dense / Hybrid) plus a
learned reranker, extractive answers with citation spans, hard multi-tenant
isolation, and an evaluation harness that measures Recall@k, MRR, nDCG@10, and a
groundedness proxy — then charts the results.

Everything runs with **zero API keys, zero model downloads, zero network** and is
fully seeded, so `make all` reproduces the numbers below bit-for-bit.

> **Headline (real run, 60,000 chunks / 500 QA items, 250 held-out, seed 42):**
> **Hybrid+Rerank** tops every metric — **nDCG@10 0.962**, **MRR 0.952**, **Recall@5 0.988** —
> beating a strong BM25 baseline (nDCG 0.950) and lifting Recall@1 from 0.888 to **0.920**.
> Full index build + eval runs in **~125 s** on 4 CPUs, fully offline. Generation streams at
> **~144k chunks/s** (5.8 M tokens/s) at bounded memory.

## Project Purpose

This repository is part of the AI engineering portfolio and focuses on the following problem space:

- Hybrid retrieval (BM25 + dense) + reranking + eval harness
- Headline result from the portfolio index: Hybrid+Rerank: Recall@1 **0.92**, nDCG@10 **0.96**

## What This Project Solves

**The learned reranker wins, and it wins honestly.** On this corpus BM25 is a
*very* strong baseline (nDCG@10 0.950): the planted facts carry rare, high-IDF
tokens (document codes, entity code-names) that exact lexical matching nails.
The learned reranker still beats it on every ranking metric — nDCG@10 0.962,
MRR 0.952, and Recall@1 from 0.888 → 0.920 — because it has BM25 *as one feature*
plus dense cosine, term-overlap, and length priors, and learns to weight them.

**Dense-only (LSA) is the weak arm, and that's the point of hybrid.** An offline
TF-IDF+SVD embedder has no external synonym knowledge, and at 60k chunks its
semantic vectors get dominated by shared topic language, so rare-entity precision
suffers (Recall@1 0.188). This is the textbook motivation for hybrid retrieval:
RRF fusion rescues dense's recall dramatically — Recall@10 climbs 0.244 → 0.792 —
and the union candidate pool it feeds the reranker is what lets the reranker
recover BM25-level top-1 precision *and* exceed it. A real transformer encoder
(dropped in behind the `Embedder` interface) would raise the dense arm and likely
push Hybrid past BM25 on its own; the architecture is unchanged.

**Latency is dominated by the reranker, not retrieval.** BM25 (sparse matvec) and
Dense (one `embeddings @ q`) both answer in ~8–12 ms at 60k chunks; hybrid roughly
doubles that (two arms + fusion) and reranking the candidate pool adds a few more
ms — still ~18–27 ms end-to-end, and independent of corpus size because the
reranker only ever sees a few hundred candidates.

**Groundedness ~0.92 across the board** confirms the extractive synthesizer keeps
answers traceable to their cited passages regardless of which retriever fed it.

## Technical Approach

```
docs ─► stream + chunk (overlap, tenant metadata) ─► TF-IDF+SVD embeddings
                                                          │
              ┌───────────────────────────────────────────┤
              ▼                     ▼                       ▼
          BM25 (Okapi)        Dense (cosine kNN)       RRF hybrid
              └────────── union candidate pool ──────────┘
                                   │
                       Reranker (LogReg, 6 feats)
                                   │
                   extractive synthesis + citations
                                   │
              eval: Recall@k · MRR · nDCG@10 · groundedness
```

- **Streaming corpus** — `stream_chunks` yields chunks one at a time (O(one-doc)
  memory), so the generator scales to millions of chunks. Sliding-window chunking
  (40-token windows, 12-token overlap) with `tenant_id / source / doc_id / timestamp`.
- **Embeddings** — TF-IDF + TruncatedSVD (LSA), L2-normalized so dot = cosine.
  Deterministic; behind an `Embedder` protocol so a real encoder drops in.
- **BM25** — Okapi, implemented from scratch (inverted index, exact IDF,
  length-normalized saturation).
- **Hybrid** — Reciprocal Rank Fusion of the Dense and BM25 rankings.
- **Reranker** — logistic regression over lexical + semantic + length features,
  trained on synthetic relevance labels, reranking the union candidate pool.
- **Synthesis** — extractive; every span carries char offsets + `chunk_id`/`doc_id`.
- **Multi-tenant isolation** — retrieval takes a tenant mask; isolation is enforced
  *inside* the retriever. Tests assert zero cross-tenant leakage.

## Benchmark And Validation Evidence

The portfolio root documents **28 passing tests** for this project, and the repo quickstart uses `make test` as the standard validation path. The benchmark outputs committed in `benchmarks/` and the generated visuals in `assets/` are the evidence package for this delivery.

### retrieval_quality.md

# Retrieval Quality (held-out QA)

| method | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 | grounded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.888 | 0.988 | 0.988 | 0.988 | 0.937 | 0.950 | 0.920 |
| Dense | 0.188 | 0.216 | 0.220 | 0.244 | 0.206 | 0.215 | 0.922 |
| Hybrid | 0.300 | 0.540 | 0.664 | 0.792 | 0.450 | 0.532 | 0.921 |
| Hybrid+Rerank | 0.920 | 0.984 | 0.988 | 0.988 | 0.952 | 0.962 | 0.920 |

## Visual Artifacts Reviewed

- `assets/quality_comparison.png`: Retrieval quality — 4 methods compared.
- `assets/scorecard.png`: Evaluation scorecard.
- `assets/latency_vs_k.png`: Query latency vs retrieval depth.
- `assets/reranker_gain.png`: Reranker lift & ranking quality.

## Engineering Notes

The primary design and scale decisions are documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md). The benchmark markdown in [`benchmarks/`](./benchmarks) and the generated figures in [`assets/`](./assets) should be read together: the markdown gives the measured numbers, and the screenshots make those results easier to inspect quickly during review.

## Files Included In This Repo

- [`README.md`](./README.md) for project overview, quickstart, and headline results
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) for system design and scaling choices
- [`benchmarks/`](./benchmarks) for measured results from the committed runs
- [`assets/`](./assets) for generated screenshots and dashboards
- [`tests/`](./tests) for the automated validation suite

## Delivery Summary

This project document was prepared for **Sai Veda** so the repository reads like a real project handoff: what the system is for, what problem it solves, what evidence supports it, and where the benchmark and test artifacts live inside the repo.

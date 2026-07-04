# Architecture

Enterprise RAG Platform is a retrieval-augmented generation stack designed the
way a production system is: an ingestion path that streams, an index layer with
pluggable retrieval backends, a learned reranker, extractive synthesis with
citations, an evaluation harness that gates quality, and hard multi-tenant
isolation. Everything runs offline and deterministically so the whole pipeline
is reproducible from a single seed.

## Component map

```
                          ┌──────────────────────────────────────────────┐
                          │              INGESTION (streaming)             │
   docs (policies,        │  generate_documents ─► sliding-window chunker  │
   tickets, wiki) ───────►│  chunk_tokens=40, overlap=12, O(1-doc) memory  │
                          │  metadata: tenant_id, source, doc_id, ts       │
                          └───────────────────────┬──────────────────────┘
                                                  │ chunks
                          ┌───────────────────────▼──────────────────────┐
                          │                 INDEX LAYER                    │
                          │  Embedder (TF-IDF + TruncatedSVD/LSA, L2)      │
                          │  Dense (cosine kNN)   BM25 (Okapi inverted)    │
                          └───────────────────────┬──────────────────────┘
                                                  │
              query ─► embed ─┬───────────────────┼─────────────────┐
                              ▼                    ▼                 ▼
                         Dense top-k          BM25 top-k        RRF fusion
                              └─────── union candidate pool ───────┘
                                                  │
                                    ┌─────────────▼─────────────┐
                                    │  Reranker (LogReg on 6     │
                                    │  lexical+semantic feats)   │
                                    └─────────────┬─────────────┘
                                                  ▼
                                    Extractive synthesis + citations
                                                  │
                                    ┌─────────────▼─────────────┐
                                    │  Eval harness: Recall@k,   │
                                    │  MRR, nDCG@10, groundedness│
                                    └───────────────────────────┘
```

## Design decisions and trade-offs

### Offline, deterministic embeddings behind an interface
Real transformer encoders need model downloads / GPUs that aren't available in a
sealed environment, and they make results non-reproducible. We use **TF-IDF +
TruncatedSVD (latent semantic analysis)**, L2-normalized so a dot product equals
cosine similarity. This captures both lexical and co-occurrence structure and is
completely deterministic (fixed `random_state`, `n_iter=2` randomized SVD).

The encoder sits behind the `Embedder` protocol (`fit` / `encode` / `dim`), so a
`SentenceTransformerEmbedder` could drop in without touching retrieval, fusion,
reranking, or eval. That is the important architectural property: **the offline
model is an implementation detail, not a design constraint.**

Trade-off: LSA has no external world knowledge, so it cannot resolve true
synonyms the corpus never co-locates. In our benchmark this shows up as a strong
lexical BM25 baseline that pure dense retrieval does not beat — an honest result.
The value of the dense arm is *robustness* (it recovers documents whose exact
query terms are missing) and *feature diversity* for the reranker.

### Three retrieval backends + RRF
- **BM25 (Okapi)** — implemented from scratch with exact IDF and
  length-normalized term saturation. Because the per-(term, doc) BM25 weight is
  query-independent, we materialize a sparse CSR weight matrix `W` (docs × vocab)
  once at fit time; a query is a 0/1 column indicator and scoring is a single
  sparse matvec `W @ q` — exact BM25, but ~10x faster than a Python scan over long
  postings lists (measured: 90ms → 8ms per query at 60k chunks). Per-document
  term-frequency maps additionally give the reranker an O(query-terms) `score_doc`.
- **Dense** — cosine kNN. Unfiltered queries use a fitted `NearestNeighbors`
  (brute, exact); tenant-filtered queries use a masked numpy dot product. Both
  paths return identical scoring semantics.
- **Hybrid** — **Reciprocal Rank Fusion**: `RRF(d) = Σ 1/(rrf_k + rank_l(d))`.
  RRF is score-scale agnostic (BM25 scores and cosines are not comparable), needs
  no tuning, and rewards documents both arms agree on.

Why RRF over learned weighting for the *fusion* step: it is parameter-free and
robust. The learned component lives one layer up, in the reranker, where features
are individually meaningful.

### Reranker: a cross-encoder in spirit, a logistic model in practice
A transformer cross-encoder is the standard reranker but is undownloadable here.
We approximate its *behavior* — jointly scoring a (query, doc) pair — with six
features: fraction of query terms present, Jaccard overlap, IDF-weighted overlap,
BM25 score, dense cosine, and a length prior; scored by a `LogisticRegression`
trained on synthetic relevance labels (the gold chunk is the positive, the other
retrieved candidates are hard negatives). It reranks the **union** of the BM25
and Dense candidate pools, so its recall ceiling is `max(BM25, Dense)` rather than
the RRF-truncated list. Swapping in a real cross-encoder means implementing the
same `train`/`rerank` surface.

### Extractive synthesis with citations
We select the most query-relevant sentence from each top passage and stitch them,
recording each span's character offsets and provenance (`chunk_id`, `doc_id`).
Extractive (not generative) synthesis is a deliberate choice for an offline,
faithful system: every answer token is traceable to a source, which the
groundedness metric verifies. A generative LLM would slot in behind the same
`Answer` contract.

### Multi-tenant isolation
Every chunk carries a `tenant_id`. Queries pass a boolean mask over chunk indices;
BM25 skips disallowed postings and Dense masks disallowed rows to `-inf` before
top-k. Isolation is enforced **inside** the retriever, not by post-filtering
results (which would leak ranking signal and waste the k budget). `test_pipeline`
asserts zero cross-tenant leakage for every tenant, and tenant scoping also
improves precision by shrinking the candidate space to the relevant partition.

In production this maps to per-tenant index shards or a partition key
(`WHERE tenant_id = ?`) pushed into the vector store / inverted index, giving both
isolation and data-locality.

## Scaling to millions of docs / 1B tokens

The system is built so the *same code path* extrapolates; what changes at scale is
where the index physically lives.

**Ingestion is already streaming.** `stream_chunks` / `count_chunks` yield one
chunk at a time with O(one-document) memory, so corpus size is bounded by time,
not RAM. `generate_data.py` writes Parquet in row-group batches (zstd). The
`benchmark_scale.py` script measures generation throughput at increasing sizes and
extrapolates to 1B tokens without ever materializing the corpus — see
`benchmarks/scale.csv`.

**Where each layer goes at 1B tokens (~25M chunks):**

| Layer | Prototype (this repo) | Production at 1B tokens |
| --- | --- | --- |
| Corpus store | Parquet, zstd | Partitioned Parquet on object storage; DuckDB / Polars-lazy for out-of-core scans |
| Embeddings | in-RAM float32 matrix | Sharded memory-mapped `.npy` / on-disk vector store; encode in batches |
| Dense ANN | exact brute `NearestNeighbors` | HNSW / IVF-PQ (FAISS-class) per shard; brute force is O(N·d) and only fits ~1M in RAM |
| BM25 | in-memory sparse CSR weight matrix | Segment-based inverted index (Lucene/Tantivy) or DuckDB full-text; postings on disk |
| Reranker | LogReg over 6 features | Same features, or a GPU cross-encoder over the top ~100 candidates only |
| Tenancy | boolean mask | Per-tenant shards / partition pruning pushed to the store |

The key scaling insight is that **retrieval is embarrassingly shardable**:
partition by tenant (and by time), retrieve top-k per shard, then RRF-merge across
shards — the same fusion primitive already used for BM25+Dense. Memory per node
stays bounded; throughput scales with node count. The reranker only ever sees a
few hundred candidates per query, so its cost is independent of corpus size.

Honest measurement boundary: this repo builds and evaluates a real in-memory index
of ~10^5 chunks and streams-benchmarks generation to the 10^5–10^6 range in
seconds. The 1B-token figure is an architected extrapolation (streaming generator
+ sharded ANN), not an in-RAM run — consistent with the portfolio's
"impressive but truthful" rule.

## Reproducibility
Single seed (42) drives corpus, embeddings (`random_state`), QA sampling, the
train/test split, and the reranker. BLAS threads are pinned to 1 (the sandbox
oversubscribes threads, turning a 2s SVD into ~100s). Re-running `make run`
reproduces the benchmark tables bit-for-bit.

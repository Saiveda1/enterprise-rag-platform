"""Build the index, run the four-method comparison, and write benchmark tables.

    python scripts/run_eval.py --docs 30000 --queries 500

Outputs:
    benchmarks/retrieval_quality.csv / .md   — method x metric table
    benchmarks/latency.csv                   — latency (ms) vs k
    benchmarks/results.json                  — full run payload for charts
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

from ragplatform.pipeline import (
    METHODS,
    _candidate_pool,
    build_index,
    demo_tenant_isolation,
    evaluate,
    measure_latency,
    split_qa,
    train_reranker,
)
from ragplatform.qa import build_qa_set
from ragplatform.synthesis import synthesize

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"


def _quality_table(summary: dict, ks) -> tuple[list[str], list[list[str]]]:
    header = ["method"] + [f"recall@{k}" for k in ks] + ["mrr", "ndcg@10", "grounded"]
    rows = []
    for m in METHODS:
        if m not in summary:
            continue
        v = summary[m]
        row = [m] + [f"{v['recall'][k]:.3f}" for k in ks]
        row += [f"{v['mrr']:.3f}", f"{v['ndcg']:.3f}", f"{v['grounded']:.3f}"]
        rows.append(row)
    return header, rows


def _write_md(path: Path, header, rows, title: str) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    path.write_text("\n".join(lines) + "\n")


def _write_csv(path: Path, header, rows) -> None:
    import csv

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=int, default=30_000)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    BENCH.mkdir(parents=True, exist_ok=True)

    print(f"[run] building index for {args.docs:,} docs ...")
    t0 = time.perf_counter()
    index = build_index(args.docs, dim=args.dim, seed=args.seed)
    build_s = time.perf_counter() - t0
    n_chunks = len(index.chunks)
    print(
        f"[run] {n_chunks:,} chunks | embed dim {index.embedder.dim} | "
        f"build {build_s:.2f}s | explained var {index.embedder.explained_variance:.3f}"
    )

    qa = build_qa_set(index.doc_records, index.texts, n_queries=args.queries, seed=args.seed)
    train_items, test_items = split_qa(qa)
    print(f"[run] QA: {len(qa)} items ({len(train_items)} train / {len(test_items)} test)")

    t0 = time.perf_counter()
    train_reranker(index, train_items)
    print(f"[run] reranker trained in {time.perf_counter() - t0:.2f}s")

    result = evaluate(index, test_items)
    summary, ks = result["summary"], result["ks"]

    header, rows = _quality_table(summary, ks)
    _write_csv(BENCH / "retrieval_quality.csv", header, rows)
    _write_md(BENCH / "retrieval_quality.md", header, rows, "Retrieval Quality (held-out QA)")
    print("\n[results] retrieval quality")
    print("  " + "  ".join(f"{h:>10}" for h in header))
    for r in rows:
        print("  " + "  ".join(f"{c:>10}" for c in r))

    # Latency sweep.
    lat = measure_latency(index, test_items[: min(80, len(test_items))])
    lat_ks = sorted(next(iter(lat.values())).keys())
    lhdr = ["method"] + [f"k={k}" for k in lat_ks]
    lrows = [[m] + [f"{lat[m][k]:.3f}" for k in lat_ks] for m in METHODS]
    _write_csv(BENCH / "latency.csv", lhdr, lrows)
    print("\n[results] mean per-query latency (ms)")
    print("  " + "  ".join(f"{h:>10}" for h in lhdr))
    for r in lrows:
        print("  " + "  ".join(f"{c:>10}" for c in r))

    # Tenant isolation demo.
    tenant = index.chunks[0].tenant_id
    iso = demo_tenant_isolation(index, tenant, "password rotation policy security review")
    print(f"\n[isolation] tenant={iso['tenant']} isolated={iso['isolated']} "
          f"leaked={len(iso['leaked'])} chunks")

    # Extractive answer demo with citations.
    demo_item = test_items[0]
    qv = index.encode_query(demo_item.question)
    pool = _candidate_pool(index, demo_item.question, qv, 50)
    hits = index.reranker.rerank(demo_item.question, qv, pool, k=3)
    ans = synthesize(demo_item.question, hits, index.chunk_lookup)
    print("\n[synthesis] demo answer")
    print("  Q:", demo_item.question)
    print("  A:", ans.text[:300])
    print("  citations:", [(c.marker, c.chunk_id) for c in ans.citations])

    payload = {
        "config": {"docs": args.docs, "chunks": n_chunks, "dim": index.embedder.dim,
                   "queries": len(qa), "seed": args.seed, "build_s": round(build_s, 3),
                   "explained_variance": round(index.embedder.explained_variance, 4)},
        "quality": summary,
        "ks": ks,
        "latency": {m: {str(k): lat[m][k] for k in lat_ks} for m in METHODS},
        "latency_ks": lat_ks,
        "isolation": {"tenant": iso["tenant"], "isolated": iso["isolated"],
                      "leaked": len(iso["leaked"])},
        "answer_demo": {"question": demo_item.question, "answer": ans.text,
                        "citations": [c.chunk_id for c in ans.citations]},
    }
    (BENCH / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[run] wrote benchmarks/ (quality, latency, results.json)")


if __name__ == "__main__":
    main()

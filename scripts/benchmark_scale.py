"""Streaming scale benchmark: chunk-generation throughput at increasing sizes.

Demonstrates the generator is O(1)-memory and linear in documents, so the same
code path extrapolates to millions of chunks / a billion tokens. Only counts
chunks (never materializes the corpus), so memory stays flat.

    python scripts/benchmark_scale.py --sizes 5000 20000 80000 200000
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import time
from pathlib import Path

from ragplatform.corpus import count_chunks

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"

# Rough token count per chunk (chunk_tokens window) for a tokens/s figure.
APPROX_TOKENS_PER_CHUNK = 40


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[5_000, 20_000, 80_000, 200_000])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    BENCH.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'docs':>10} {'chunks':>12} {'seconds':>10} {'chunks/s':>12} {'Mtokens/s':>10}")
    for n_docs in args.sizes:
        t0 = time.perf_counter()
        n_chunks = count_chunks(n_docs, seed=args.seed)
        dt = time.perf_counter() - t0
        cps = n_chunks / dt
        mtps = cps * APPROX_TOKENS_PER_CHUNK / 1e6
        print(f"{n_docs:>10,} {n_chunks:>12,} {dt:>10.2f} {cps:>12,.0f} {mtps:>10.2f}")
        rows.append([n_docs, n_chunks, round(dt, 3), round(cps, 1), round(mtps, 3)])

    with (BENCH / "scale.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["docs", "chunks", "seconds", "chunks_per_s", "Mtokens_per_s"])
        w.writerows(rows)

    # Extrapolate to 1B tokens at the largest measured throughput.
    best_cps = max(r[3] for r in rows)
    tokens_1b = 1e9
    est_s = tokens_1b / (best_cps * APPROX_TOKENS_PER_CHUNK)
    print(f"\n[extrapolation] 1B tokens (~{tokens_1b / APPROX_TOKENS_PER_CHUNK:,.0f} chunks) "
          f"at {best_cps:,.0f} chunks/s ≈ {est_s / 60:,.1f} min of generation (bounded memory)")
    print(f"[bench] wrote {BENCH / 'scale.csv'}")


if __name__ == "__main__":
    main()

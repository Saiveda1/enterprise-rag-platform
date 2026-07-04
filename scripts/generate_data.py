"""Generate a synthetic enterprise corpus and stream it to Parquet.

Streaming keeps memory O(one batch), so the generator scales to millions of
chunks. Writes ``data/corpus.parquet`` (chunk text + metadata).

    python scripts/generate_data.py --docs 20000 --out data/corpus.parquet
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must precede numpy import)

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ragplatform.corpus import stream_chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=int, default=20_000, help="number of documents")
    ap.add_argument("--out", type=str, default="data/corpus.parquet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-tokens", type=int, default=40)
    ap.add_argument("--overlap-tokens", type=int, default=12)
    ap.add_argument("--batch", type=int, default=50_000, help="rows per row-group")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            ("chunk_id", pa.string()),
            ("doc_id", pa.string()),
            ("tenant_id", pa.string()),
            ("source", pa.string()),
            ("timestamp", pa.string()),
            ("text", pa.string()),
        ]
    )

    t0 = time.perf_counter()
    total = 0
    buf: list[dict] = []
    writer = pq.ParquetWriter(out, schema, compression="zstd")
    try:
        for chunk in stream_chunks(
            args.docs,
            seed=args.seed,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
        ):
            buf.append(chunk.as_dict())
            if len(buf) >= args.batch:
                writer.write_table(pa.Table.from_pylist(buf, schema=schema))
                total += len(buf)
                buf.clear()
        if buf:
            writer.write_table(pa.Table.from_pylist(buf, schema=schema))
            total += len(buf)
    finally:
        writer.close()

    dt = time.perf_counter() - t0
    print(
        f"[generate] {args.docs:,} docs -> {total:,} chunks in {dt:.2f}s "
        f"({total / dt:,.0f} chunks/s) -> {out}"
    )


if __name__ == "__main__":
    main()

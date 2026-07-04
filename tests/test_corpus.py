"""Corpus generation: determinism, chunking with overlap, streaming, metadata."""
from __future__ import annotations

from ragplatform.corpus import (
    SOURCES,
    TENANTS,
    _chunk_document,
    count_chunks,
    generate_documents,
    stream_chunks,
)
from ragplatform.retrieval import tokenize


def test_deterministic_generation():
    a = [c.chunk_id + "|" + c.text for c in stream_chunks(50, seed=1)]
    b = [c.chunk_id + "|" + c.text for c in stream_chunks(50, seed=1)]
    assert a == b
    c = [c.chunk_id for c in stream_chunks(50, seed=2)]
    assert a != c  # different seed -> different corpus


def test_metadata_present_and_valid():
    for chunk in stream_chunks(30, seed=3):
        assert chunk.tenant_id in TENANTS
        assert chunk.source in SOURCES
        assert chunk.doc_id and chunk.chunk_id.startswith(chunk.doc_id)
        assert chunk.text
        # timestamp is ISO-8601 parseable.
        from datetime import datetime

        datetime.fromisoformat(chunk.timestamp)


def test_chunk_overlap():
    # A long document must split into overlapping windows.
    sentences = [" ".join(f"word{i}" for i in range(100))]
    chunks = list(
        _chunk_document("d0", "acme", "wiki", "2023-01-01T00:00:00",
                        sentences, chunk_tokens=40, overlap_tokens=12)
    )
    assert len(chunks) >= 2
    toks0 = tokenize(chunks[0].text)
    toks1 = tokenize(chunks[1].text)
    # Overlap region: tail of chunk0 shares tokens with head of chunk1.
    assert set(toks0) & set(toks1)
    # Window size respected.
    assert len(toks0) <= 40


def test_count_matches_stream():
    assert count_chunks(40, seed=5) == sum(1 for _ in stream_chunks(40, seed=5))


def test_streaming_is_lazy():
    # Taking a few items from a huge request must not materialize everything.
    gen = stream_chunks(10_000_000, seed=9)
    first = [next(gen) for _ in range(5)]
    assert len(first) == 5


def test_gold_fact_is_recoverable():
    # Every document's planted fact sentence must survive into some chunk.
    docs = list(generate_documents(30, seed=4))
    for doc in docs:
        fact_tokens = set(tokenize(doc["meta"]["fact_sentence"]))
        chunks = list(
            _chunk_document(doc["doc_id"], doc["tenant_id"], doc["source"],
                            doc["timestamp"], doc["sentences"],
                            chunk_tokens=40, overlap_tokens=12)
        )
        covered = set()
        for c in chunks:
            covered |= set(tokenize(c.text))
        assert fact_tokens <= covered

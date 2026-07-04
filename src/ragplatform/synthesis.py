"""Extractive answer synthesis with citation spans.

We select the most query-relevant sentence from each of the top passages and
stitch them into an answer. Every stitched span records its character offsets in
the answer *and* its provenance (chunk_id, doc_id), so the UI can render inline
citations and the eval harness can measure groundedness.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .retrieval import Hit, tokenize

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Citation:
    marker: int
    chunk_id: str
    doc_id: str
    start: int  # char offset in the answer text
    end: int
    snippet: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)

    @property
    def cited_texts(self) -> list[str]:
        return [c.snippet for c in self.citations]


def _best_sentence(text: str, query_terms: set[str]) -> str:
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return text.strip()
    best = max(
        sentences,
        key=lambda s: len(query_terms & set(tokenize(s))),
    )
    return best


def synthesize(
    query: str,
    hits: Sequence[Hit],
    chunk_lookup,
    *,
    max_passages: int = 3,
) -> Answer:
    """Build an extractive answer from the top ``max_passages`` hits.

    ``chunk_lookup(idx)`` returns a mapping with ``text``, ``chunk_id`` and
    ``doc_id`` for a chunk index.
    """
    query_terms = set(tokenize(query))
    parts: list[str] = []
    citations: list[Citation] = []
    cursor = 0
    for marker, hit in enumerate(hits[:max_passages], start=1):
        rec = chunk_lookup(hit.idx)
        snippet = _best_sentence(rec["text"], query_terms)
        piece = f"{snippet} [{marker}]"
        start = cursor
        end = start + len(snippet)
        citations.append(
            Citation(
                marker=marker,
                chunk_id=rec["chunk_id"],
                doc_id=rec["doc_id"],
                start=start,
                end=end,
                snippet=snippet,
            )
        )
        parts.append(piece)
        cursor = len(" ".join(parts)) + 1
    return Answer(text=" ".join(parts), citations=citations)

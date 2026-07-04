"""Synthetic QA set with known gold chunks.

For a sampled subset of documents we synthesize a natural-language question and
resolve its *gold chunk*: the chunk whose tokens overlap most with the document's
planted fact sentence. Questions come in two flavors:

  * ``lexical``    — include the document code (favors BM25 exact matching)
  * ``paraphrase`` — describe the fact in other words (favors dense retrieval)

The mix means no single retriever dominates, which is what makes the
BM25/Dense/Hybrid/Rerank comparison meaningful.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .retrieval import tokenize


@dataclass(frozen=True)
class QAItem:
    question: str
    gold_chunk_idx: int
    tenant_id: str
    doc_id: str
    kind: str  # "lexical" | "paraphrase"


_POLICY_Q = {
    "lexical": "What does policy {code} specify about {topic}?",
    "paraphrase": "For the {entity} program, what is the rule on {topic}?",
}
_TICKET_Q = {
    "lexical": "How was support ticket {code} resolved?",
    "paraphrase": "On the {entity} service, what fixed the issue where {topic}?",
}
_WIKI_Q = {
    "lexical": "What does wiki page {code} document?",
    "paraphrase": "How does the {entity} {topic} component behave?",
}
_TEMPLATES = {"policy": _POLICY_Q, "ticket": _TICKET_Q, "wiki": _WIKI_Q}


def _gold_chunk_for_doc(
    fact_sentence: str, chunk_indices: list[int], chunk_texts: list[str]
) -> int:
    """Pick the chunk maximizing token overlap with the planted fact sentence."""
    fact_tokens = set(tokenize(fact_sentence))
    best_idx = chunk_indices[0]
    best_overlap = -1
    for ci in chunk_indices:
        overlap = len(fact_tokens & set(tokenize(chunk_texts[ci])))
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = ci
    return best_idx


def build_qa_set(
    doc_records: list[dict],
    chunk_texts: list[str],
    *,
    n_queries: int = 400,
    seed: int = 42,
) -> list[QAItem]:
    """Build up to ``n_queries`` QA items from materialized documents.

    ``doc_records`` entries carry ``doc_id``, ``tenant_id``, ``source``,
    ``meta`` (with ``code``/``topic``/``fact_sentence``) and ``chunk_indices``.
    """
    rng = random.Random(seed)
    pool = list(doc_records)
    rng.shuffle(pool)
    items: list[QAItem] = []
    for rec in pool:
        if len(items) >= n_queries:
            break
        meta = rec["meta"]
        source = rec["source"]
        kind = "lexical" if rng.random() < 0.5 else "paraphrase"
        template = _TEMPLATES[source][kind]
        question = template.format(
            code=meta["code"], topic=meta["topic"], entity=meta["entity"]
        )
        gold = _gold_chunk_for_doc(
            meta["fact_sentence"], rec["chunk_indices"], chunk_texts
        )
        items.append(
            QAItem(
                question=question,
                gold_chunk_idx=gold,
                tenant_id=rec["tenant_id"],
                doc_id=rec["doc_id"],
                kind=kind,
            )
        )
    return items

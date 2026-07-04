"""Synthetic enterprise corpus generator.

Produces three document families that mimic an enterprise knowledge base:
  * policy     — HR / security / finance policies
  * ticket     — support tickets with a problem + resolution
  * wiki       — internal engineering wiki pages

The generator is a *streaming* generator: documents are yielded one at a time
and immediately chunked, so the working-set memory is O(one document) and the
corpus can scale to millions of chunks without OOM.

Each chunk carries metadata: ``tenant_id``, ``source`` (doc family), ``doc_id``,
``chunk_id``, ``timestamp`` and the raw ``text``. Chunks are produced with a
sliding window (token count + overlap) to preserve context across boundaries.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Iterator

# ---------------------------------------------------------------------------
# Vocabulary pools — deterministic, no external data.
# ---------------------------------------------------------------------------
TENANTS = ["acme", "globex", "initech", "umbrella", "hooli"]
SOURCES = ["policy", "ticket", "wiki"]

_POLICY_TOPICS = [
    ("data retention", "records are retained for {n} days before secure deletion"),
    ("password rotation", "credentials must be rotated every {n} days"),
    ("expense approval", "expenses above ${n} require director sign-off"),
    ("remote work", "employees may work remotely up to {n} days per week"),
    ("incident response", "critical incidents must be acknowledged within {n} minutes"),
    ("vendor onboarding", "new vendors undergo a {n}-day security review"),
    ("access review", "privileged access is re-certified every {n} days"),
    ("data classification", "restricted data must be encrypted with AES-{n}"),
]
_TICKET_SYMPTOMS = [
    "the dashboard fails to load after login",
    "export to CSV times out for large tenants",
    "single sign-on returns an invalid token error",
    "the ingestion pipeline drops records intermittently",
    "webhook deliveries are delayed by several minutes",
    "the mobile client crashes on the settings screen",
    "search results omit recently indexed documents",
    "billing invoices show a duplicated line item",
]
_TICKET_FIXES = [
    "clearing the cache and reissuing the service token resolved it",
    "increasing the batch timeout to {n} seconds fixed the failures",
    "rotating the SSO signing key restored authentication",
    "adding a retry with backoff eliminated the dropped records",
    "scaling the delivery workers to {n} replicas cleared the backlog",
    "patching the client to version 4.{n} stopped the crash",
    "forcing an index refresh returned the missing documents",
    "de-duplicating the ledger job corrected the invoices",
]
_WIKI_TOPICS = [
    ("service mesh", "routes traffic between microservices with mutual TLS"),
    ("feature store", "serves precomputed features with a {n} ms p99 budget"),
    ("deployment runbook", "canary rollout shifts {n} percent of traffic per step"),
    ("observability stack", "traces are sampled at {n} percent under load"),
    ("data lakehouse", "partitions are compacted every {n} hours"),
    ("vector index", "shards hold up to {n} million vectors each"),
    ("auth gateway", "issues short-lived tokens valid for {n} minutes"),
    ("cost controls", "idle clusters scale to zero after {n} minutes"),
]
_FILLER = [
    "This document is maintained by the platform team and reviewed quarterly.",
    "Refer to the linked runbook for step-by-step remediation.",
    "Stakeholders are notified through the standard change-management channel.",
    "Historical context is preserved for audit and compliance purposes.",
    "The owning team tracks exceptions in the central governance registry.",
    "Escalation paths are defined in the on-call rotation schedule.",
    "Metrics for this area are surfaced on the reliability dashboard.",
    "Related configuration lives in the infrastructure-as-code repository.",
]

_CONS = "bcdfghjklmnprstvw"
_VOW = "aeiou"


def _entity_name(k: int) -> str:
    """Deterministic pronounceable name for entity slot ``k`` (e.g. 'Kavodu').

    Fixed 3-syllable form gives up to 17*5*17*5*17*5 ≈ 6.1M distinct names, so
    entity slots stay unique within a pool while reading as real code-names.
    """
    parts = []
    for _ in range(3):
        parts.append(_CONS[k % len(_CONS)])
        k //= len(_CONS)
        parts.append(_VOW[k % len(_VOW)])
        k //= len(_VOW)
    return "".join(parts).capitalize()


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with tenant + provenance metadata."""

    chunk_id: str
    doc_id: str
    tenant_id: str
    source: str
    timestamp: str
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


def _tokenize_words(text: str) -> list[str]:
    return text.split()


def _chunk_document(
    doc_id: str,
    tenant_id: str,
    source: str,
    timestamp: str,
    sentences: list[str],
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> Iterator[Chunk]:
    """Sliding-window chunking over the concatenated sentence tokens.

    Uses a token window with overlap so a fact that straddles two windows is
    still recoverable. Yields lazily to keep memory bounded.
    """
    words = _tokenize_words(" ".join(sentences))
    if not words:
        return
    step = max(1, chunk_tokens - overlap_tokens)
    idx = 0
    start = 0
    while start < len(words):
        window = words[start : start + chunk_tokens]
        text = " ".join(window)
        yield Chunk(
            chunk_id=f"{doc_id}::c{idx}",
            doc_id=doc_id,
            tenant_id=tenant_id,
            source=source,
            timestamp=timestamp,
            text=text,
        )
        idx += 1
        if start + chunk_tokens >= len(words):
            break
        start += step


def _make_policy(
    rng: random.Random, doc_key: int, entity: str
) -> tuple[list[str], str, dict]:
    topic, template = rng.choice(_POLICY_TOPICS)
    n = rng.randint(7, 365)
    code = f"POL-{doc_key:06d}"
    fact = f"Under the {entity} policy {code}, {topic}: {template.format(n=n)}."
    sents = [
        f"Corporate {topic} policy {code} governs the {entity} program.",
        fact,
        rng.choice(_FILLER),
        f"Compliance with {code} is mandatory for all {entity} business units.",
        rng.choice(_FILLER),
    ]
    meta = {"topic": topic, "code": code, "n": n, "entity": entity, "fact_sentence": fact}
    return sents, "policy", meta


def _make_ticket(
    rng: random.Random, doc_key: int, entity: str
) -> tuple[list[str], str, dict]:
    symptom = rng.choice(_TICKET_SYMPTOMS)
    fix = rng.choice(_TICKET_FIXES)
    n = rng.randint(2, 90)
    code = f"TCK-{doc_key:06d}"
    fix_text = fix.format(n=n) if "{n}" in fix else fix
    fact = f"For the {entity} service, ticket {code}: {symptom}; {fix_text}."
    sents = [
        f"Support ticket {code} was opened for the {entity} service.",
        f"Reported symptom on {entity}: {symptom}.",
        fact,
        rng.choice(_FILLER),
        f"Ticket {code} on {entity} was closed after verification.",
    ]
    meta = {"topic": symptom, "code": code, "n": n, "entity": entity, "fact_sentence": fact}
    return sents, "ticket", meta


def _make_wiki(
    rng: random.Random, doc_key: int, entity: str
) -> tuple[list[str], str, dict]:
    topic, template = rng.choice(_WIKI_TOPICS)
    n = rng.randint(1, 99)
    code = f"WIKI-{doc_key:06d}"
    fact = f"The {entity} {topic} component ({code}) {template.format(n=n)}."
    sents = [
        f"Engineering wiki page {code} documents the {entity} {topic}.",
        fact,
        rng.choice(_FILLER),
        f"Design notes for the {entity} {topic} are versioned in {code}.",
        rng.choice(_FILLER),
    ]
    meta = {"topic": topic, "code": code, "n": n, "entity": entity, "fact_sentence": fact}
    return sents, "wiki", meta


_MAKERS = {"policy": _make_policy, "ticket": _make_ticket, "wiki": _make_wiki}


def generate_documents(
    n_docs: int,
    *,
    seed: int = 42,
    tenants: list[str] | None = None,
) -> Iterator[dict]:
    """Yield document descriptors (pre-chunk) with metadata + a gold fact.

    Kept separate from chunking so the QA generator can reuse the fact
    sentences to build questions with known gold chunks.
    """
    rng = random.Random(seed)
    tenants = tenants or TENANTS
    base_ts = datetime(2023, 1, 1)
    # Entity code-names are drawn from a bounded pool so each entity is shared by
    # a handful of documents — creating realistic distractors that reward
    # combining lexical + semantic signals rather than a single exact match.
    entity_pool = max(50, n_docs // 6)
    for i in range(n_docs):
        source = SOURCES[i % len(SOURCES)]
        maker = _MAKERS[source]
        entity = _entity_name(rng.randrange(entity_pool))
        sents, src, meta = maker(rng, i, entity)
        tenant = tenants[rng.randrange(len(tenants))]
        ts = (base_ts + timedelta(hours=rng.randint(0, 24 * 900))).isoformat()
        doc_id = f"{src}-{i:07d}"
        yield {
            "doc_id": doc_id,
            "tenant_id": tenant,
            "source": src,
            "timestamp": ts,
            "sentences": sents,
            "meta": meta,
        }


def stream_chunks(
    n_docs: int,
    *,
    seed: int = 42,
    chunk_tokens: int = 40,
    overlap_tokens: int = 12,
    tenants: list[str] | None = None,
) -> Iterator[Chunk]:
    """Stream chunks for ``n_docs`` documents. O(1 document) memory."""
    for doc in generate_documents(n_docs, seed=seed, tenants=tenants):
        yield from _chunk_document(
            doc["doc_id"],
            doc["tenant_id"],
            doc["source"],
            doc["timestamp"],
            doc["sentences"],
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )


def count_chunks(
    n_docs: int,
    *,
    seed: int = 42,
    chunk_tokens: int = 40,
    overlap_tokens: int = 12,
) -> int:
    """Count chunks without materializing them (streaming, bounded memory)."""
    total = 0
    for _ in stream_chunks(
        n_docs, seed=seed, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens
    ):
        total += 1
    return total

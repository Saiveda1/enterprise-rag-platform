"""Shared test fixtures. Pins BLAS threads (sandbox oversubscription) and builds
one small index reused across the suite."""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from ragplatform.pipeline import build_index, split_qa, train_reranker
from ragplatform.qa import build_qa_set


@pytest.fixture(scope="session")
def index():
    return build_index(600, dim=96, seed=7)


@pytest.fixture(scope="session")
def qa(index):
    return build_qa_set(index.doc_records, index.texts, n_queries=120, seed=7)


@pytest.fixture(scope="session")
def trained_index(index, qa):
    train_items, _ = split_qa(qa)
    train_reranker(index, train_items)
    return index

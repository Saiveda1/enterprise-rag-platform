# Enterprise RAG Platform — one-command workflows.
# Everything is offline and deterministic (seeded). BLAS pinned to 1 thread to
# avoid sandbox oversubscription; override on real hardware if desired.

PY ?= python3
DOCS ?= 30000
QUERIES ?= 500
export PYTHONPATH := src
export MPLBACKEND := Agg
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1

.PHONY: help setup data run test bench screenshots all clean

help:
	@echo "make setup        install dependencies"
	@echo "make data         generate synthetic corpus  (DOCS=$(DOCS))"
	@echo "make run          build index + evaluate 4 methods (writes benchmarks/)"
	@echo "make test         run pytest suite"
	@echo "make bench        streaming scale benchmark"
	@echo "make screenshots  render dashboards into assets/"
	@echo "make all          run + screenshots (the full demo)"

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/generate_data.py --docs $(DOCS) --out data/corpus.parquet

run:
	$(PY) scripts/run_eval.py --docs $(DOCS) --queries $(QUERIES)

test:
	$(PY) -m pytest -q

bench:
	$(PY) scripts/benchmark_scale.py --sizes 5000 20000 80000 200000

screenshots:
	$(PY) scripts/make_screenshots.py

all: run screenshots

clean:
	rm -rf data/*.parquet __pycache__ src/ragplatform/__pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

UV ?= uv
PY := .venv/bin/python
SEED ?= 42
N ?= 600
DEV ?= 400

.PHONY: install data data-report baseline test lint typecheck check clean

install:
	$(UV) venv --python 3.12
	$(UV) pip install -e ".[dev]"

## Regenerate both splits from scratch and verify every invariant.
data:
	rm -rf data/dev data/holdout
	$(PY) -m recon.cli generate --out data --seed $(SEED) --n $(N) --dev $(DEV)

data-report:
	$(PY) -m recon.cli data-report data/dev
	$(PY) -m recon.cli data-report data/holdout

## Run the Phase 2 deterministic baselines and score them.
baseline:
	$(PY) -m recon.cli baseline --data-root data --split both

test:
	$(PY) -m pytest tests -q

lint:
	.venv/bin/ruff check src tests

typecheck:
	.venv/bin/mypy src

check: lint typecheck test

clean:
	rm -rf data/dev data/holdout reports .pytest_cache .mypy_cache .ruff_cache

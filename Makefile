.PHONY: help install install-dev test lint fmt typecheck example smoke \
        reference reference-check precommit scan clean

PY ?= python

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:         ## Install the package
	$(PY) -m pip install -e .

install-dev:     ## Install with dev extras
	$(PY) -m pip install -e ".[dev]"

test:            ## Run the test suite
	$(PY) -m pytest

lint:            ## Lint (ruff)
	$(PY) -m ruff check src tests scripts

fmt:             ## Auto-format (ruff)
	$(PY) -m ruff format src tests scripts

typecheck:       ## Static types (mypy, advisory)
	$(PY) -m mypy

example:         ## Run the end-to-end toy example
	$(PY) examples/run_toy.py --out outputs/toy

smoke:           ## Run the toy example and assert the expected verdict
	$(PY) examples/run_toy.py --out outputs/toy --assert-verdict

reference:       ## Regenerate the committed reference results (review the diff first!)
	$(PY) scripts/make_reference.py

reference-check: ## Verify the committed reference results still hold
	$(PY) -m pytest -q tests/test_reference.py

precommit:       ## Run every check CI runs, locally
	$(PY) -m ruff check src tests scripts
	$(PY) -m pytest -q
	$(PY) scripts/scan_repo.py .

scan:            ## Repository hygiene scan (secrets / oversized files)
	$(PY) scripts/scan_repo.py .

clean:           ## Remove build and runtime output
	rm -rf outputs .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
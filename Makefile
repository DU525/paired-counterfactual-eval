.PHONY: help install install-dev test lint fmt typecheck example smoke clean scan

PY ?= python

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:         ## Install the package
	$(PY) -m pip install -e .

install-dev:     ## Install with dev extras
	$(PY) -m pip install -e ".[dev]"

test:            ## Run the test suite
	$(PY) -m pytest

lint:            ## Lint (ruff)
	$(PY) -m ruff check src tests

fmt:             ## Auto-format (ruff)
	$(PY) -m ruff format src tests

typecheck:       ## Static types (mypy)
	$(PY) -m mypy

example:        ## Run the end-to-end toy example
	$(PY) examples/run_toy.py --out outputs/toy

smoke:          ## Run the toy example and assert the expected verdict
	$(PY) examples/run_toy.py --out outputs/toy --assert-verdict

scan:           ## Repository hygiene scan (secrets / oversized files)
	$(PY) scripts/scan_repo.py .

clean:          ## Remove build and runtime output
	rm -rf outputs .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
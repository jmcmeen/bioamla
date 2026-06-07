# Makefile for bioamla development tasks.
# Dependency resolution and environment management use uv (https://docs.astral.sh/uv/).

.DEFAULT_GOAL := help

UV ?= uv
PKG := bioamla
SRC := src
TESTS := tests

.PHONY: help install sync lock upgrade dev shell \
        test test-fast cov bench \
        lint fmt fmt-check check \
        build publish publish-test clean docs

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## --- Environment & dependencies (uv) ---

install: ## Create venv and install the project with dev extras
	$(UV) sync --extra dev

sync: ## Sync the environment to the lockfile (all extras)
	$(UV) sync --all-extras

lock: ## Resolve and write uv.lock
	$(UV) lock

upgrade: ## Upgrade locked dependencies to latest allowed versions
	$(UV) lock --upgrade

shell: ## Spawn a subshell with the project venv activated
	$(UV) run $${SHELL:-bash}

## --- Testing ---

test: ## Run the full test suite
	$(UV) run pytest

test-fast: ## Run tests, skipping slow/integration markers
	$(UV) run pytest -m "not slow and not integration"

cov: ## Run tests with coverage report
	$(UV) run pytest --cov=$(PKG) --cov-report=term-missing

bench: ## Run benchmark tests only
	$(UV) run pytest --benchmark-only

## --- Linting & formatting (ruff) ---

lint: ## Lint with ruff
	$(UV) run ruff check $(SRC) $(TESTS)

fmt: ## Auto-format and apply lint fixes
	$(UV) run ruff format $(SRC) $(TESTS)
	$(UV) run ruff check --fix $(SRC) $(TESTS)

fmt-check: ## Check formatting without modifying files
	$(UV) run ruff format --check $(SRC) $(TESTS)

check: lint fmt-check test ## Run lint, format check, and tests

## --- Packaging ---

build: clean ## Build sdist and wheel
	$(UV) build

publish-test: build ## Publish to TestPyPI
	$(UV) run twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	$(UV) run twine upload dist/*

## --- Docs ---

docs: ## Build Sphinx documentation
	$(UV) run --extra docs sphinx-build -b html docs/source docs/_build/html

## --- Housekeeping ---

clean: ## Remove build artifacts and caches
	rm -rf dist build *.egg-info src/*.egg-info
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

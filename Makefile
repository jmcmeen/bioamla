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
        docs docs-serve \
        clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## --- Environment & dependencies (uv) ---

install: ## Create venv and install the project with dev tooling
	$(UV) sync --extra dev

dev: ## Install the full stack + dev tooling (runtime deps are all in base)
	$(UV) sync --extra dev

sync: ## Sync the environment to the lockfile (incl. dev tooling)
	$(UV) sync --extra dev

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

## --- Documentation (mkdocs) ---

docs: ## Build the documentation site into ./site
	$(UV) run --extra dev mkdocs build

docs-serve: ## Serve docs locally with live reload at http://127.0.0.1:8000
	$(UV) run --extra dev mkdocs serve

## --- Housekeeping ---

clean: ## Remove caches and test/coverage/docs build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov site
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

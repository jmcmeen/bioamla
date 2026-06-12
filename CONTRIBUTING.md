# Contributing to bioamla

Thanks for your interest in improving bioamla! This guide covers the setup,
conventions, and checks for contributing code, docs, or bug reports.

## Getting started

bioamla uses [uv](https://docs.astral.sh/uv/) for environments and
[ruff](https://docs.astral.sh/ruff/) + [pytest](https://docs.pytest.org/) for
quality. Everything is driven through the `Makefile`.

```bash
git clone https://github.com/jmcmeen/bioamla
cd bioamla
make install      # uv sync --extra dev — full runtime stack + tooling
make check        # lint + format-check + tests (the gate before a PR)
```

You'll also need **ffmpeg** on your system for audio I/O
(`scripts/install-deps.sh` handles common platforms).

## Development workflow

- Branch off `main` (development happens on `dev-*` branches).
- Make your change with a test that covers it.
- Run `make fmt` to auto-format, then `make check` — it must pass.
- Open a PR against `main`. CI runs lint, format-check, and the test matrix
  (Python 3.10–3.13) on pull requests.

```bash
make test         # pytest
make test-fast    # skip slow/integration markers
make fmt          # auto-format + ruff --fix
make docs         # mkdocs build --strict
```

## Project conventions

These keep changes consistent with the design (see `CLAUDE.md` for the full set):

- **Errors are exceptions.** Functions return plain data and raise on failure;
  every exception derives from `bioamla.exceptions.BioamlaError`. No error
  sentinels, `None`-on-failure, or `(ok, value)` tuples.
- **Organize by domain, not by layer.** New functionality goes in the relevant
  domain subpackage (`audio`, `viz`, `indices`, `detect`, `cluster`, `catalogs`,
  `datasets`, `ml`, `system`). No services/models/repository split.
- **`import bioamla` stays fast.** Heavy third-party deps (torch, librosa, umap,
  hdbscan, transformers, …) are imported *inside* the functions that use them,
  never at module top level of an eagerly-imported module.
- **The public surface of a domain is its `__init__.py`** — re-export the
  intended API there with an explicit `__all__`; internal modules are `_`-prefixed.
- **Keep the CLI thin.** Logic lives in the domain; command bodies parse args,
  call the domain, and print. Domain imports in command bodies are function-local.
- **Type hints + docstrings** on public functions; modern syntax (`X | None`,
  `list[str]`); line length 100 (the formatter owns wrapping).

## Testing

- `tests/` mirrors the domain layout — put a new test beside its domain.
- Reuse the synthetic fixtures in `tests/conftest.py` (no binary fixture files).
- Mark tests with `slow` / `integration` / `requires_cuda` as appropriate, and
  guard heavy/optional backends with `pytest.importorskip(...)` so the suite
  stays green on slim and full installs alike.
- Add a test with every behavior change.

## Reporting bugs / requesting features

Open an issue using the templates under
[`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/). For security issues, see
[SECURITY.md](SECURITY.md) — please do **not** open a public issue.

## Code of conduct

Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).

# CLAUDE.md

Guidance for working in this repo. These are the project's invariants and patterns — follow
them so changes stay consistent with the design. When something here conflicts with what you
see in the code, trust the code and flag the drift.

## What this is

`bioamla` is a Python library + CLI for **bioacoustics and machine learning**: audio I/O and
signal processing, acoustic indices, event detection, spectrogram visualization, embedding
clustering, species catalogs, dataset tooling, and AST-based ML inference.

- **0.2.x is a ground-up rebuild.** The old layered architecture (core/services/models/
  repository) was torn out and replaced by flat, domain-oriented packages. APIs may still
  change. When in doubt, favor **removing/simplifying over adding** — keep the core lean.
- Python **≥ 3.10**, `src/` layout, package `bioamla`, CLI entry point `bioamla` →
  `bioamla.cli.cli:main`.
- Tooling: **uv** for envs/deps, **ruff** for lint+format, **pytest** for tests, **mkdocs**
  for docs. Drive everything through the `Makefile`.

## Invariants (do not break these)

1. **Errors are exceptions.** Functions return plain data and raise on failure. Every
   exception derives from `bioamla.exceptions.BioamlaError`. Never return error sentinels,
   `None`-on-failure, or `(ok, value)` tuples. Add a new subclass to `exceptions.py` rather
   than raising bare `Exception`/`ValueError` from library code.
2. **Organize by domain, not by layer.** New functionality goes in the relevant domain
   subpackage. Do **not** reintroduce a services/models/repository/core split. Do file I/O
   directly with `pathlib` — there is no repository layer.
3. **`import bioamla` stays fast.** Domain subpackages are lazy (PEP 562 `__getattr__` in
   `bioamla/__init__.py`); heavy third-party deps (torch, librosa, umap, hdbscan,
   sounddevice, transformers) are **imported inside the functions that use them**, never at
   module top level of an eagerly-imported module. Keep `bioamla --help` snappy.
4. **Batteries included, but lazy.** A single `pip install bioamla` installs the full runtime
   stack (it's all in `[project.dependencies]` — there are no runtime extras; only `[dev]`).
   `DependencyError` is therefore reserved for genuine environment breakage (missing system
   lib like ffmpeg, broken install), not for "optional package not installed."
5. **The public surface of a domain is its `__init__.py`.** Re-export the intended API there
   with an explicit `__all__`. Internal-only modules are prefixed `_` (e.g. `_models.py`,
   `_io.py`, `_metadata.py`).
6. **No TUI here.** The TUI ("magpy-lite", `textual`) was extracted to a separate repo. Do
   not add `textual` or TUI code back.
7. **ruff-clean and tests green before declaring done.** `make check` must pass.

## Architecture

Flat domain packages under `src/bioamla/`:

| Package | Responsibility |
|---|---|
| `audio` | audio I/O, discovery, analysis, signal processing (filter/denoise/normalize/resample/segment), playback. `AudioData` is the core DTO. |
| `viz` | spectrograms, mel/MFCC, waveform plots |
| `indices` | acoustic indices — ACI, ADI, AEI, BIO, NDSI, spectral/temporal entropy |
| `detect` | event detection — energy, RIBBIT, peaks, accelerating-pattern |
| `cluster` | embedding dim-reduction (umap), clustering (hdbscan), novelty detection |
| `catalogs` | external providers — Xeno-canto, iNaturalist, eBird, Macaulay, HuggingFace |
| `datasets` | dataset merge / augment / licensing / annotations / partition / stats |
| `ml` | Audio Spectrogram Transformer (AST) inference, training, embeddings |
| `system` | configuration, dependency checks, environment info |

Cross-cutting modules (no domain owns them):

- `bioamla/exceptions.py` — the single exception hierarchy. Read it before adding errors.
- `bioamla/batch.py` — the generic batch engine (`run_batch`, `run_csv_batch`,
  `discover_files`, CSV helpers, `BatchConfig`/`BatchResult`/`SegmentInfo`). Supports two
  input modes: **directory** (`--input-dir`) and **CSV-metadata** (`--input-file` with a
  `file_name` column; results merge back into the CSV). Each domain that supports batch has a
  thin `batch.py` that reuses this engine — don't write a parallel one.
- `bioamla/common/` — shared low-level helpers: `config`, `constants`, `files`, `http`,
  `progress`.
- `bioamla/cli/` — thin Click layer; see below.

### Data transfer objects

DTOs are `@dataclass`es, typically mixing in `ToDictMixin` (`audio/data.py`) for recursive
`to_dict()` serialization. Prefer a typed dataclass result over a bare dict for anything
returned to a caller (e.g. `AcousticIndices`, `AudioData`, `AudioMetadata`).

### CLI layer (`bioamla/cli/`)

- `cli.py` defines the root Click group and registers each command group; `main()` runs Click
  in `standalone_mode=False` and **centrally catches `BioamlaError`** → friendly `Error: ...`
  on exit 1 (`cli/errors.py`).
- One file per command group in `cli/commands/` (`audio.py`, `indices.py`, …).
- **Command bodies import domain functions lazily** (function-local `from bioamla.audio import
  …`) to preserve fast startup — match this; don't add domain imports at module top of a
  command file.
- Convention in command bodies: wrap the domain call in `try/except BioamlaError as e: raise
  click.ClickException(str(e)) from e`. (The central handler in `main()` is the backstop; the
  per-command wrap gives Click-native usage/exit behavior.) Keep CLI code thin — logic lives
  in the domain, the command just parses args, calls, and prints.

## Conventions

- **Type hints** on public functions; modern syntax (`X | None`, `list[str]`) — ruff `UP` is
  on, `target-version = py310`.
- **ruff**: line-length 100 (formatter owns wrapping; `E501` ignored), isort with
  `known-first-party = ["bioamla"]`. Run `make fmt` to auto-fix.
- **Docstrings**: module docstring explaining the domain + per-public-function docstrings with
  Attributes/Args; many modules cite literature references (keep that for indices/detect).
- Use `logging.getLogger(__name__)`, not `print`, in library code. The CLI uses
  `click.echo` / `rich` for user output.
- **Config / secrets**: `.env` in CWD (or a parent) is auto-loaded on import (`override=False`,
  so real env vars win). Catalog keys: `EBIRD_API_KEY`, `XC_API_KEY`, `HF_TOKEN`. Read keys at
  call time; never hardcode or commit them.

## Testing

- `tests/` mirrors the domain layout (`tests/audio/`, `tests/indices/`, …). Put a new test
  beside its domain.
- Shared fixtures live in `tests/conftest.py` (e.g. `sample_audio_data`, `test_audio_path`,
  `test_audio_dir`) — reuse them; they generate synthetic sine-wave audio, no fixture files.
- Markers (strict — `--strict-markers`): `slow`, `integration`, `requires_cuda`, `benchmark`.
  Mark accordingly. `make test-fast` deselects `slow`/`integration`.
- Guard tests that need heavy/optional backends with `pytest.importorskip("torch")` so the
  suite stays green on a slim install **and** a full one.
- Add a test with every behavior change. `make cov` for coverage.

## Dev workflow

```bash
make install      # uv sync --extra dev  (full runtime stack + tooling)
make test         # pytest
make test-fast    # skip slow/integration
make check        # lint + format-check + test  ← gate before "done"
make fmt          # auto-format + ruff --fix
make docs         # mkdocs build --strict
```

**Commits/branches:** Development happens on `dev-*` branches (default branch is `main`).
**The user handles commits** — make your changes in the working tree and stop there; don't
commit or offer to unless explicitly asked.

## Recommended skills

When working here, reach for these (invoke with `/<name>`):

- **`/code-review`** — review the current diff for correctness bugs and cleanup opportunities
  before handing work back. Use after any non-trivial change.
- **`/simplify`** — applies reuse/simplification/altitude cleanups; fits this repo's
  "simplify over add" direction. Good after getting something working.
- **`/verify`** / **`/run`** — actually run the CLI/library to confirm a change works, not
  just that tests pass. The CLI is the primary user surface — exercise it (`bioamla <group>
  <cmd> --help`, then a real invocation).
- **`/security-review`** — for changes touching `catalogs`/`http`/credential handling or file
  writes.
- **`/init`** — only to regenerate this doc wholesale; prefer editing in place.

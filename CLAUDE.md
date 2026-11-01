# CLAUDE.md

Guidance for working in this repo. These are the project's invariants and patterns — follow
them so changes stay consistent with the design. When something here conflicts with what you
see in the code, trust the code and flag the drift.

## What this is

`bioamla` is a Python library + CLI for **bioacoustics and machine learning**: audio I/O and
signal processing, acoustic indices, event detection, spectrogram visualization, embedding
clustering, species catalogs, dataset tooling, and AST-based ML inference.

- **Flat, domain-oriented packages.** The old layered architecture (core/services/models/
  repository) was torn out in 0.2.x. When in doubt, favor **removing/simplifying over
  adding** — keep the core lean.
- Python **≥ 3.11**, `src/` layout, package `bioamla`, console script `bioamla` →
  `bioamla._cli_entry:main` (a shim that runs `bioamla.cli.cli:main`).
- Tooling: **uv** for envs/deps, **ruff** for lint+format, **pytest** for tests, **mkdocs**
  for docs. Drive everything through the `Makefile`.

## Invariants (do not break these)

1. **Errors are exceptions.** Functions return plain data and raise on failure. Every
   exception derives from `bioamla.exceptions.BioamlaError`. Never return error sentinels,
   `None`-on-failure, or `(ok, value)` tuples. Add a new subclass to `exceptions.py` rather
   than raising bare `Exception`/`ValueError` from library code.
2. **Organize by domain, not by layer.** New functionality goes in the relevant domain
   subpackage. Do **not** reintroduce a services/models/repository/core split. Do file I/O
   directly with `pathlib` — there is no repository layer. **Operations** (0.4.0) are not a
   layer: each domain declares its own (typed request, function, typed result) next to its
   code, and `bioamla.operations` is only an index of them. See
   `docs/architecture/workspaces-and-remote.md`.
3. **`import bioamla` stays fast.** Domain subpackages are lazy (PEP 562 `__getattr__` in
   `bioamla/__init__.py`); heavy third-party deps (torch, librosa, umap, hdbscan,
   sounddevice, transformers) are **imported inside the functions that use them**, never at
   module top level of an eagerly-imported module. Keep `bioamla --help` snappy.
4. **Slim base, opt-in extras.** `pip install bioamla` installs only the light core
   (numpy, scipy, pandas, soundfile, pydub, riffy, requests, the catalog clients). Heavy
   dependencies live in extras: `audio` (librosa, sounddevice), `viz`, `indices`, `detect`,
   `ml` (torch, transformers, …), `cluster`, `server` (fastapi, uvicorn), and `all`. `bioamla/_extras.py` maps each extra
   to its modules. A domain package that needs an extra wholesale calls `require_extra(...)`
   at the top of its `__init__.py`; a function elsewhere imports its heavy dependency with
   `import_extra("librosa", "audio")`. Both raise `DependencyError` with the install command,
   so `DependencyError` covers "extra not installed" as well as environment breakage (missing
   ffmpeg). When adding a heavy dependency, put it in an extra, add it to `EXTRAS`, and import
   it through these helpers. `[cli]` (`click`, `rich`) is only for the `bioamla` command; the
   library never imports it. The console script entry point is the dependency-light shim
   `bioamla/_cli_entry.py`, which prints an install hint when run without `[cli]`, and
   `cli.main()` turns a missing-extra `ModuleNotFoundError` into the same kind of hint.
   `[dev]` pulls in `[all,cli]` so the test suite exercises everything.
5. **The public surface of a domain is its `__init__.py`.** Re-export the intended API there
   with an explicit `__all__`. Internal-only modules are prefixed `_` (e.g. `_models.py`,
   `_io.py`, `_metadata.py`).
6. **No TUI here.** The TUI ("magpy-lite", `textual`) was extracted to a separate repo. Do
   not add `textual` or TUI code back.
7. **ruff-clean and tests green before declaring done.** `make check` must pass.
8. **The library is the API; operations are thin adapters over it.** Every capability is a
   public, documented library function first (explicit arguments, exported from its domain's
   `__init__`). An operation only maps a typed request onto those calls and wraps the result;
   if an operation (or a command) grows real logic, move it into a library function. Anything
   the CLI or the server can do must be doable, and tested, by importing bioamla directly.

## Architecture

Flat domain packages under `src/bioamla/`:

| Package | Responsibility |
|---|---|
| `audio` | audio I/O, discovery, analysis, signal processing (filter/denoise/normalize/resample/segment), playback. `AudioData` is the core DTO. |
| `viz` | spectrograms, mel/MFCC, waveform plots, scrolling spectrogram videos (`video.py`, encodes through the `ffmpeg` CLI) |
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
- **Commands are front ends to operations** (0.4.0, being converted group by group): a command
  builds the operation's request from its options, runs the operation, and prints the result.
  Anything a command does beyond that (CSV-mode merging, writing outputs) moves into public
  library functions that the operation calls, because the server runs the same operation
  without the CLI, and scripts use the library without either.

## Conventions

- **Type hints** on public functions; modern syntax (`X | None`, `list[str]`) — ruff `UP` is
  on, `target-version = py311`.
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
- Markers (strict — `--strict-markers`): `slow`, `integration`, `requires_cuda`.
  Mark accordingly. `make test-fast` deselects `slow`/`integration`; CI runs everything.
- `tests/integration/` chains real commands through `bioamla --json`, each step reading the
  previous envelope's outputs. When a change touches what one command writes and another reads
  (annotation columns, metadata CSVs, config files), add or extend a pipeline there.
- Guard tests that need heavy/optional backends with `pytest.importorskip("torch")` so the
  suite stays green on a slim install **and** a full one.
- Add a test with every behavior change. `make cov` for coverage.

## Dev workflow

```bash
make install      # uv sync --extra dev  (every extra + tooling)
make test         # pytest
make test-fast    # skip slow/integration
make check        # lint + format-check + test  ← gate before "done"
make fmt          # auto-format + ruff --fix
make docs         # mkdocs build --strict
```

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
  <cmd> --help`, then a real invocation). For an end-to-end check, the project ships a
  **`run-bioamla`** skill (`.claude/skills/run-bioamla/`) whose `smoke.sh` drives config /
  audio / indices / detect / visualize against a synthesized WAV.
- **`/security-review`** — for changes touching `catalogs`/`http`/credential handling or file
  writes.
- **`/init`** — only to regenerate this doc wholesale; prefer editing in place.
- **`/deep-research`** — multi-source, adversarially verified research report. Reach for this
  before adding a new detection algorithm, acoustic index, or ML model — compare literature,
  evaluate alternatives (e.g. BirdNET vs PANNs vs AST), or explore catalog API capabilities.
- **`/review`** — review a GitHub PR. Distinct from `/code-review` (which reviews the working
  diff); use this on a PR before it merges to `main`.
- **`/claude-api`** — Claude API / Anthropic SDK reference. Auto-triggers on LLM-shaped tasks
  in the `ml` or `catalogs` packages; also useful when designing embedding-based or
  LLM-augmented features.
- **`/fewer-permission-prompts`** (periodic) — scans transcripts to find un-allowlisted tool
  calls and adds them to `.claude/settings.local.json`. Run after significant changes to the
  test suite or CI workflow to keep prompt fatigue low.

# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]

A ground-up rebuild. `bioamla` is now a flat, **domain-oriented** library with a thin CLI,
designed to be a stable core that other applications can build on.

**License:** relicensed from **GPLv3** (0.0.x–0.1.x) to the **MIT License** starting with this
release. The project is sole-authored, so the relicense applies to all current and future code.

### Added

- **Domain packages**: `audio`, `viz`, `indices`, `detect`, `cluster`, `catalogs`,
  `datasets`, `ml`, `system`, plus a shared `batch` engine — each importable directly
  (`from bioamla.indices import compute_all_indices`).
- **Exception hierarchy** (`bioamla.exceptions`): functions return plain data and raise from
  `BioamlaError` (e.g. `AudioLoadError`, `InvalidInputError`, `DependencyError`). The CLI
  catches it centrally.
- **Batteries-included install**: a single `pip install bioamla` brings the full runtime stack
  (audio/viz/indices/detect, the PyTorch + AST ML stack, clustering, playback, tracking). The
  only optional group is `[dev]` (test/lint/docs tooling). Heavy imports stay lazy so
  `import bioamla` and `bioamla --help` remain fast.
- Batch CLI supports both directory mode and **CSV-metadata mode** (a `file_name` column;
  results merged back into the CSV, paths resolved relative to the CSV).
- **HuggingFace dataset pull** — `catalogs.huggingface.pull_dataset` /
  `bioamla catalogs hf pull-dataset` fetch a Hub audio dataset (e.g. `ashraq/esc50`) and
  materialize it into the labeled-folder + `metadata.csv` layout that `dataset` / `models ast
  train` consume directly.
- **Audio editing transforms** in `bioamla.audio` (and `bioamla audio` CLI): `pitch_shift`,
  `time_stretch`, `add_noise`, `apply_gain` — deterministic single-file ops, distinct from the
  randomized pre-training augmentation layer.
- **`ml.train_ast`** — AST fine-tuning is now a parameter-driven library function returning a
  `TrainResult`; the `models ast train` command is a thin wrapper that also accepts a
  `--config <file.toml>` (explicit flags override the file, which overrides defaults). It trains
  off a Hub dataset id directly (grab-and-go), a local metadata CSV, or a dataset directory —
  and a directory carrying a `metadata.csv` (from `partition`/`pull-dataset`) is loaded via that
  CSV so a fixed `split` column and the `train/val/test/<label>/` layout are honored.
- **`catalogs hf cache`** + `catalogs.huggingface.scan_cache` / `purge_cache` — inspect and purge
  the local HuggingFace cache that repeat pulls/loads populate (replaces `config purge`, keeping
  HF concerns in the `hf` group).
- **Experiment tracking**: `tensorboard` is bundled and is the default `--report-to` for
  `models ast train`; `mlflow` is also available, and `--report-to`/`[training].report_to`
  accept `none`, `tensorboard`, `mlflow`, or a comma-separated combination.
- **`models ast annotate`** + `datasets.predictions_to_annotations` — turn segmented inference
  output into an editable annotation file to seed manual review (the predict → review → dataset loop).
- `cluster cluster` now exposes **HDBSCAN** (the default) and `--min-cluster-size`, matching the
  core / `batch cluster` capabilities.
- Example end-to-end workflows under `examples/`, and a full offline+gated `dev-data/cli_test.sh`.
- Contributor docs: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and GitHub
  issue/PR templates.
- Documentation site (MkDocs + Material + mkdocstrings) and GitHub Actions for CI, PyPI
  publish, and docs deployment.

### Changed

- Replaced the layered architecture (`core/` → `services/` → `models/`/`repository/`) with
  domain packages; removed the `ServiceResult` wrapper and the repository dependency-injection
  layer in favor of plain returns + exceptions and direct `pathlib` I/O.
- `__version__` is now derived from installed package metadata (single source of truth in
  `pyproject.toml`).
- CLI startup is lazy — `import bioamla` no longer eagerly imports heavy domains.

### Removed

- **TensorFlow** and the BirdNET / custom-CNN / bioacoustics-model-zoo stack (`tensorflow`,
  `tf-keras`, `ai-edge-litert`, `onnx`, `timm`, `bioacoustics-model-zoo`). ML inference is
  AST-only for now.
- The TUI (extracted to its own repository).
- A large body of dead/experimental code (the old `dev/` package), preserved on the
  `experimental/dev-archive` branch.

### Fixed

- Infinite recursion in `get_audio_info`.
- Batch parallel mode (`max_workers > 1`) now uses a thread pool, so closure-based
  per-item functions work in parallel and the `fork()`-in-multithreaded-process
  DeprecationWarning is gone.
- AST training sets `TENSORBOARD_LOGGING_DIR` instead of the deprecated
  `TrainingArguments(logging_dir=...)` kwarg (removed in transformers v5.2).
- Pinned `transformers>=5.10.2`, which ships the AST checkpoint key-conversion so the
  pretrained encoder from old Hub checkpoints (e.g. `MIT/ast-finetuned-audioset-...`)
  actually loads — older versions silently fine-tuned a randomly-initialized encoder.
  Also dropped a redundant `model.init_weights()` call in training (a verified no-op).
- Several batch-CLI paths that previously raised `AttributeError`
  (`batch models predict`, `batch cluster`).
- Output commands now create parent directories before writing.

[Unreleased]: https://github.com/jmcmeen/bioamla/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jmcmeen/bioamla/releases/tag/v0.2.0

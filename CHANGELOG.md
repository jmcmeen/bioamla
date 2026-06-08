# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]

A ground-up rebuild. `bioamla` is now a flat, **domain-oriented** library with a thin CLI,
designed to be a stable core that other applications can build on.

### Added
- **Domain packages**: `audio`, `viz`, `indices`, `detect`, `cluster`, `catalogs`,
  `datasets`, `ml`, `system`, plus a shared `batch` engine — each importable directly
  (`from bioamla.indices import compute_all_indices`).
- **Exception hierarchy** (`bioamla.exceptions`): functions return plain data and raise from
  `BioamlaError` (e.g. `AudioLoadError`, `InvalidInputError`, `DependencyError`). The CLI
  catches it centrally.
- **Slim core + optional extras**: base install is lightweight; `pip install "bioamla[ml]"`,
  `[detect]`, `[cluster]`, `[playback]`, `[all]`, `[docs]` add heavier capabilities. Heavy
  dependencies load lazily and raise `DependencyError` with the extra to install.
- Batch CLI supports both directory mode and **CSV-metadata mode** (a `file_name` column;
  results merged back into the CSV, paths resolved relative to the CSV).
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
- Several batch-CLI paths that previously raised `AttributeError`
  (`batch models predict`, `batch cluster`).
- Output commands now create parent directories before writing.

[Unreleased]: https://github.com/jmcmeen/bioamla/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jmcmeen/bioamla/releases/tag/v0.2.0

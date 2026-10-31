# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `bioamla --json <command>`: machine-readable output for scripts and agents. stdout gets
  exactly one JSON envelope (`{"bioamla", "command", "ok", "result"}`, or `"error": {"type",
  "message"}` on failure), and everything meant for people goes to stderr. `annotation summary`,
  `audio metadata` and `dataset stats` report results so far; the other commands follow.
- `--json` results for `detect energy|ribbit|peaks|accelerating` (`detections`, `count`, the
  options used and any `-o` path), `indices compute|temporal|aci|adi|aei|bio|ndsi|entropy`,
  `audio info` and `audio list`. Reported options are keyed by option name (`--threshold-db` as
  `threshold_db`), not by internal parameter name.

### Deprecated

- The per-command `--json` flags of `annotation summary`, `audio metadata` and `dataset stats`.
  They still print bare JSON (now with a note on stderr); use `bioamla --json <command>`.

## [0.3.1] - 2026-10-31

### Added

- Config objects are validated. `BatchConfig`, `ReductionConfig`, `ClusteringConfig`,
  `ModelConfig`, `InferenceConfig`, `EmbeddingConfig` and the augmentation configs check their
  fields on construction and raise `ConfigValidationError` (a `ConfigError` and an
  `InvalidInputError`) naming each bad or unknown field. They remain dataclasses: positional
  arguments, `asdict` and `replace` work as before. pydantic is a new base dependency; it loads
  when a config is first used, so `import bioamla` and the CLI start as fast as before.
- `bioamla models ast train --config` validates the whole file before training. An unknown
  section or key, or a value of the wrong type or out of range, is reported with its
  `section.key` path instead of being ignored or failing mid-run.
- JSON Schema for every config: `bioamla system schema [NAME]`, `config_schema()` and
  `validate_config()` in `bioamla.common.config`. The schemas are also published under
  `docs/schemas/`, so editors can complete and check a training config file.
- Structured fields on log records. The batch engine and every batch function attach
  `operation`, `file`, `index`, `total`, `elapsed_s` (and `error` or `status`); `train_ast` attaches
  `run_id`, the step as `index`/`total`, `epoch` and a final `status` (finished, cancelled or
  failed). The names are documented in `bioamla.common.log.LOG_FIELDS`. `run_batch` and
  `run_csv_batch` take an `operation` name, and `TrainResult` has the run's `run_id`.
- `bioamla --log-format json` writes log records to stderr as JSON lines with those fields
  (`bioamla.common.log.JsonFormatter`), and `--log-level` sets the level for any command.

- `bioamla.protocols`: the interfaces bioamla accepts from callers, as `typing.Protocol`s —
  `OnProgress` (the `on_progress(completed, total)` hook), `OnBatchProgress`, `OnTrainProgress`,
  `ShouldStop`, `OnDownloadProgress` and `AudioModel`. Function signatures now use them, and
  `common.progress.ProgressCallback` and `catalogs.inat.ProgressCallback` are aliases of them.
  The module imports only the standard library.
- Experiment-tracker hook for `train_ast`: `tracker=` takes one or more
  `bioamla.protocols.ExperimentTracker`s, which receive the run's parameters (including the
  augmentation config and label list), every logged metric, a `bioamla_run.json` summary written
  next to the best model, and the final status: finished, cancelled or failed. `MLflowTracker`
  and `TensorBoardTracker` are included in `bioamla.ml`. A tracker that fails only logs a
  warning.

### Changed

- With MLflow in `report_to` (or `"all"` with MLflow installed), `train_ast` now reports to
  MLflow through an `MLflowTracker` instead of transformers' MLflow callback, so each run has one
  MLflow writer. Runs still end `KILLED` when cancelled and `FAILED` on error. The run's params
  are now bioamla's (dataset, base model, hyperparameters, augmentation, labels, run id) rather
  than transformers' full `TrainingArguments` dump, and training inside an already-active MLflow
  run (e.g. a sweep) gets its own nested run. Param values longer than MLflow allows are
  truncated.

### Deprecated

- `bioamla.datasets.AugmentationConfig` is renamed `WaveformAugmentationConfig`, and
  `bioamla.ml.AugmentationConfig` is renamed `SpectrogramAugmentationConfig`. The two were
  unrelated classes with the same name. The old names still work and emit a
  `DeprecationWarning`; they will be removed in a future minor release.

### Fixed

- `bioamla --help` (and every command) starts in about 50 ms instead of about 400 ms. The
  `annotation` and `batch` command modules imported the datasets package, numpy and the batch
  engine at load time; they now import them when a command runs.
- `bioamla system deps` suggested `bioamla config deps --install`, a command that doesn't exist;
  it now says `bioamla system deps --install`.

## [0.3.0] - 2026-10-31

### Changed

- **Breaking: the heavy dependencies are now extras.** `pip install bioamla` installs a slim
  base (audio I/O and metadata, resampling, discovery, batch, datasets, catalogs). The
  librosa-based signal processing and playback need `bioamla[audio]`; `bioamla.viz`, `bioamla.indices`, `bioamla.detect`,
  `bioamla.ml` and `bioamla.cluster` need the extra of the same name. `bioamla[all]` installs
  everything, as a plain install did before. A missing extra raises `DependencyError` with the
  `pip install` command, and the `bioamla` command prints the same hint.
- `resample_audio` (and everything that resamples through it: `load_audio_data(sample_rate=...)`,
  `save_audio(target_sample_rate=...)`, clip extraction, augmentation) uses scipy's polyphase
  resampler instead of librosa, so it works on the base install. Output differs slightly from
  before.

### Fixed

- On Python 3.13 and later, pydub's `audioop` replacement (`audioop-lts`) is now a direct
  dependency instead of arriving through librosa.

## [0.2.15] - 2026-10-27

### Added

- CI tests and PyPI classifiers cover Python 3.14.

## [0.2.10] - 2026-10-17

### Added

- **riffy reads WAV headers.** `riffy` is now a dependency. `get_audio_info`,
  `get_wav_metadata` and `bioamla audio info` read PCM WAV headers with it, so WAV
  metadata needs no FFmpeg, libsndfile or torch and always reports bit depth. Other
  formats, and WAVs riffy cannot parse (non-PCM), still go through the pydub / ffprobe
  backend. `bioamla.audio.info.read_audio_info_dict` is the shared routing function.
- **Recorder metadata.** `bioamla.audio.get_recording_metadata` reads the deployment
  details field recorders embed in WAV files (GUANO, AudioMoth, Wildlife Acoustics WAMD,
  `bext`, RIFF INFO) and flattens them into one `RecordingMetadata`: `recorded_at`,
  `latitude`/`longitude`, `device_make`/`device_model`/`device_id`, `gain`,
  `temperature_c`, `battery_voltage` and the `sources` found. `dump_recording_metadata`
  returns every decoded field per standard. `bioamla audio metadata FILE [--json]` prints
  either view, and `bioamla audio info --metadata` appends the flat view when present.
- **WAV pre-flight and metadata in batch.** `bioamla.audio.validate_wav` checks that a file
  is well-formed PCM WAV from the header alone. `bioamla batch audio validate` runs it over
  a directory or manifest, writes `validation_report.csv` (`file_name`, `valid`, `error`),
  marks non-WAV files as skipped and exits 1 if anything is invalid, so it works as a
  pre-flight step with no FFmpeg, torch or GPU. `bioamla batch audio metadata` writes
  `recorder_metadata.csv` with the flat recorder fields per WAV, or merges them into a
  manifest with `--input-file`.
- **Recorder metadata in datasets.** `bioamla.datasets.add_recording_metadata` fills the
  recorder columns of a dataset's `metadata.csv` from its WAV files in place. When those
  columns are present, `build_manifest_from_metadata` (and `bioamla dataset manifest`)
  records `recorded_from`/`recorded_to`, the distinct `devices`, and the distinct `sites`
  with coordinates rounded to 0.01 deg; the dataset card gains a "Recording" section.
  Recorder columns are written after the catalog columns in `metadata.csv`. Gain is also
  read from Wildlife Acoustics' `WA|Song Meter|Audio settings` GUANO field, and when no
  standard carries a timestamp it falls back to a `PREFIX_YYYYMMDD_HHMMSS` file name
  (`sources` then includes `filename`). riffy's parse warnings for off-spec chunks are
  logged at debug level instead of printed per file.

### Changed

- **`bioamla audio info` no longer decodes the file.** It read the whole clip into memory
  to print four numbers; it now reads the header and also prints format, bit depth and
  subtype.

## [0.2.9] - 2026-10-17

### Added

- **Dataset remixes.** `bioamla.datasets.remix_dataset` builds a derived dataset from a
  labeled source directory: keep or drop classes, add a class from another clip directory,
  or replace a class's examples. The output is organized by label with its own
  `metadata.csv` and a `LICENSE` generated from the attribution columns, so provenance
  follows the clips. `RemixRecipe.from_toml` loads a small TOML description (`[remix]` with
  `keep`/`drop`, plus `[[add]]` and `[[replace]]` tables), and `bioamla dataset remix` is the
  CLI front end, taking a recipe, `--keep`/`--drop` flags, or both.

## [0.2.8] - 2026-10-17

### Added

- **Spectrogram and waveform videos.** `bioamla.viz.generate_spectrogram_video` renders an
  audio file as a video in sync with its audio: a mel or STFT spectrogram, or the waveform
  alone, either scrolling past a fixed playhead or held still page by page while the playhead
  sweeps across (the page can be the whole recording). Optional waveform strip, frequency
  ticks, time readout, and `start`/`end` to render just a stretch of a long file. Frames are
  composed in numpy and piped to the `ffmpeg` CLI, so there is no video library dependency;
  the audio is muxed in. `bioamla audio video` is the CLI front end. Folded in from the
  audio-spectre and spectrogram-video-creator prototypes.

## [0.2.4] - 2026-10-10

### Fixed

- **`ASTInference` loads checkpoints that ship no `preprocessor_config.json`.** It now falls
  back to the default AST feature extractor, as `ASTModel` and `extract_embeddings_file`
  already did. Previously `predict_file`, `batch_predict_files`/`batch_predict_segments`, and
  the `ASTInference` engine itself raised `ModelError` for such models, including the default
  `bioamla/scp-frogs`.
- **AST inference works on Apple Silicon.** `ASTInference` and `ASTModel` loaded weights with
  `device_map="auto"` (which selects MPS) but sent inputs to the CPU device, so every prediction
  failed with a device mismatch. The model is now moved to the device inference runs on.
- **Model paths decode WAV/FLAC/OGG without a torchcodec-compatible FFmpeg.**
  `load_waveform_tensor` now falls back to libsndfile when `torchaudio`'s FFmpeg backend
  cannot load (FFmpeg missing, or a major version outside the 4–8 that torchcodec supports),
  so AST prediction, embedding, and training keep working on those formats. Formats only
  FFmpeg can decode still require it.

## [0.2.3] - 2026-06-29

### Changed

- **Colorful CLI output.** Every command group now renders through a shared Rich-based
  semantic palette (green ✓ for completed actions, bold headers/values, cyan paths, dim for
  empty results) instead of plain monochrome text. `cli/progress.py` was renamed to
  `cli/console.py` to reflect that it owns general output, not just progress bars, and its
  helpers (`echo`, `print_success`, `print_header`, `print_path`, `print_kv`, …) are hardened
  to never wrap data lines and to escape interpolated text.
- **Pipe-safe by design.** Output styling is auto-stripped on non-TTY/`NO_COLOR`, and
  machine-readable JSON/CSV dumps bypass Rich entirely, so redirected/piped output stays plain.
  Errors and warnings now go to **stderr** (previously some went to stdout), keeping a piped
  stdout stream clean.
- **Unified error formatting.** The central `BioamlaError` handler and per-command validation
  messages now share one bold-red `Error:` prefix, matching Click's own error style.
- **Gradient `--version` banner.** The ASCII logo is rendered with a vertical green → cyan →
  indigo colour gradient (smooth on truecolor terminals, gracefully downsampled elsewhere).

## [0.2.2] - 2026-06-28

- **CLI update**: Updated --version cli to show a ascii-art banner

## [0.2.1] - 2026-06-24

### Added

- **Expanded acoustic measurements** — `datasets.compute_measurements` now spans the time,
  amplitude, power, frequency, and entropy domains plus a summarized peak-frequency contour
  (~24 new metrics: `zero_crossing_rate`, `peak_time`, `rms_db`/`peak_db`/`crest_factor_db`/
  `dynamic_range`, `avg_power`/`max_power`/`energy`, `peak_frequency`, frequency
  quartiles/percentiles (`freq_q1`/`freq_q3`/`freq_5`/`freq_95`) and `bandwidth_90`/
  `bandwidth_iqr`, `spectral_entropy`/`temporal_entropy`, and `pfc_*` contour summaries). New
  frequency metrics honour each annotation's `low_freq`/`high_freq` band, matching the existing
  `centroid`/`rolloff`. Pass `metrics="all"` for the full set; `ALL_METRICS` and the unchanged
  `DEFAULT_METRICS` are exported from `bioamla.datasets`. Uncomputable metrics are omitted, not
  emitted as `NaN`.

### Changed

- **CLI dependencies are now an optional `[cli]` extra.** `click` and `rich` moved out of the
  base install into `pip install bioamla[cli]`; the library itself never imports them. The
  `bioamla` console command now prints an install hint instead of a raw `ModuleNotFoundError`
  when the extra is absent. `[dev]` includes `[cli]`, so contributor installs are unchanged.

## [0.2.0] - 2026-06-12

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
- **On-the-fly training augmentation** — `models ast train` can apply an `audiomentations`
  pipeline (Gaussian noise, time-stretch, pitch-shift, gain, gain-transition, clipping-distortion).
  Every layer is **off by default** and opt-in, per layer, via a flag (`--add-noise`,
  `--pitch-shift`, …) or an `[augmentation]` TOML section, each with its own range, per-sample
  probability, and a pipeline-level probability; `--augment-multiplier` repeats the train split.
  Training logs exactly which transforms are active and with what parameters.
- **`models ast init-config`** — write a documented `ast_training.toml` starter
  (`[models]`/`[training]`/`[augmentation]`) to drive `train --config`; CLI flags still override
  the file.
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
- **No system-wide settings file.** Domain/library functions are entirely parameter-driven with
  sensible defaults — none read a global config. The `bioamla config` group is now
  system-information only (`version`, `devices`, `deps`). The only configurable surface is the
  explicit training `--config` file (generated by `models ast init-config`); the Xeno-canto API
  key resolves from `set_xc_api_key()` then the `XC_API_KEY` env var / `.env` (no config-file
  fallback).

### Removed

- **TensorFlow** and the BirdNET / custom-CNN / bioacoustics-model-zoo stack (`tensorflow`,
  `tf-keras`, `ai-edge-litert`, `onnx`, `timm`, `bioacoustics-model-zoo`). ML inference is
  AST-only for now.
- The TUI (extracted to its own repository).
- A large body of dead/experimental code (the old `dev/` package), preserved on the
  `experimental/dev-archive` branch.
- The pre-rebuild **system-wide config cascade** and its CLI: `config init` / `config show` /
  `config path`, the `Config`/`DEFAULT_CONFIG` machinery, the `~/.config`/`/etc` discovery
  cascade, the bundled `*.toml` project templates, and the unused `@config_aware` helpers — a
  holdover that mostly advertised settings nothing consumed.

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

[Unreleased]: https://github.com/jmcmeen/bioamla/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/jmcmeen/bioamla/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/jmcmeen/bioamla/compare/v0.2.26...v0.3.0
[0.2.15]: https://github.com/jmcmeen/bioamla/compare/v0.2.14...v0.2.15
[0.2.10]: https://github.com/jmcmeen/bioamla/compare/v0.2.9...v0.2.10
[0.2.9]: https://github.com/jmcmeen/bioamla/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/jmcmeen/bioamla/compare/v0.2.7...v0.2.8
[0.2.4]: https://github.com/jmcmeen/bioamla/compare/v0.2.3...v0.2.4
[0.2.0]: https://github.com/jmcmeen/bioamla/releases/tag/v0.2.0

# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `bioamla.workspace`: project workspaces. A workspace is a directory with a `.bioamla/` folder
  (`workspace.toml`, `config.toml`, `artifacts.json`, and `annotations/`, `imported/`,
  `datasets/`, `models/`, `runs/`); `create_workspace()`, `open_workspace()` and
  `find_workspace()` (walks up from a directory, like git). Artifacts are linked (referenced in
  place; paths inside the workspace stored relative so it can move) or imported (copied in).
  `WorkspaceError` for malformed workspaces and ones written by a newer bioamla. First piece of
  the 0.4.0 workspace and remote work; the format is shared with MagPy.
- `bioamla workspace init|status|add|import|list|remove`: create a workspace, see what it holds,
  link files or folders in place (`add`) or copy them in (`import`), list artifacts (flagging
  any whose linked file has gone missing), and forget one (`--delete-import` also removes an
  imported copy). Commands find the workspace from the current directory or `-C PATH` (relative
  paths then resolve against it, as with `git -C`); several PATHS are added all-or-nothing; all
  report `--json` results. Library: `Workspace.link_many()` and `import_many()`.
- Settings: any command option can take its default from `~/.config/bioamla/config.toml`, the
  workspace's `.bioamla/config.toml`, or a `BIOAMLA_<COMMAND>_<OPTION>` environment variable, in
  that order, with the command-line flag winning over all of them. Tables name commands
  (`[detect.energy]`) and keys name options by their long name. `bioamla workspace config
  list|get|set|unset [--user]` manages them; unknown keys are reported. See the new Settings
  page. Library: `bioamla.common.settings`.
- Run history: every command run inside a workspace is recorded in `.bioamla/history.jsonl`
  (time, duration, command line, working folder, host, user, bioamla version, success, exit
  code, error, and the output files its result names). Values of secret-looking options are
  hidden; `--help`/`--version` runs aren't recorded, and a problem writing the file never fails
  the command. `bioamla workspace history [-n N] [--failed] [--all]` shows it, and
  `workspace status` shows the last 5 runs (`last_runs` in its `--json` result). Library:
  `Workspace.record_run()` and `Workspace.history()`.
- `BIOAMLA_PROGRESS=jsonl`: commands with a progress bar write `start`, `progress` (`done`,
  `total`, `description`) and `end` (`ok`) events as JSON lines on stderr instead of drawing the
  bar: one per whole percent (every item of a small batch), at least every 0.1 s, plus the final
  count before `end`; `end`'s `ok` is false if the work failed or stopped early. Sent with
  `--quiet` too; a closed stderr stops the events, not the command or its count. Documented
  under "Progress events" in the CLI reference.
- `bioamla system commands [PATH...]`: the command catalog. Lists every command, or describes
  the ones under PATH; with `--json`, every group and command with its help, arguments and
  options (flags, type, choices, path and range constraints, default, required, multiple), each
  option's `BIOAMLA_*` variable and settings key, and the global options. Repeatable options
  default to `[]`; the table marks required options; a broken settings file is a warning, not an
  error, so the catalog still lists. Library: `bioamla.cli.catalog.command_catalog()` (raises
  `NotFoundError` for an unknown PATH).

## [0.3.8] - 2026-10-31

### Security

- Locked dependencies updated past open security advisories (all indirect): pillow 13.0.0,
  anyio 4.14.2, aiohttp 3.14.5, cryptography 50.0.2, filelock 4.2.6, fsspec 2026.7.0,
  urllib3 2.8.0, starlette 1.10.0, graphql-core 3.2.13, sqlparse 0.6.0, mako 1.4.3,
  multidict 7.2.1, opentelemetry 1.45.2, werkzeug 3.1.9, setuptools 84.2.1 and, for the docs
  build, pymdown-extensions 11.0.2 (held below 12, which breaks mkdocstrings). On Python 3.10,
  pillow and filelock stay at 12.3.0 and 4.1.0, the last releases that support it.

## [0.3.6] - 2026-10-31

### Fixed

- Raven selection tables with numeric scores and custom `Confidence` values share one column.
  An explicit numeric score, including zero, takes precedence over a custom value on the same
  annotation; text values are kept when no numeric score is set.
- CSV-mode batch commands run with `--output-dir` write the updated metadata CSV there, and its
  paths now resolve from there: relative `file_name`, `parent_file`, `features_file` and
  `embedding_file` entries are re-rooted instead of staying relative to the input CSV's folder,
  so the next batch command can read the CSV.
- CSV-mode `batch models embed` adds an `embedding_file` column, and `batch cluster
  --input-file` reads it, so embed -> cluster works over one metadata CSV.
- `batch models embed` no longer overwrites the embeddings of files that share a name
  (`a1.wav` and `b/a1.wav`, or `rec.wav` and `rec.flac`) while reporting every file as done.
- CSV mode: a row whose embedding or feature table fails this run gets a blank `embedding_file`
  / `features_file` instead of keeping an earlier run's, and `batch cluster` skips rows with a
  blank `embedding_file` instead of trying the audio file as an embedding.

### Changed

- `batch models embed` (directory mode) mirrors the input tree under `--output-dir`:
  embeddings of files in subfolders land in matching subfolders instead of all at the top level.
  `batch cluster --input-dir` searches subfolders by default, so embed -> cluster is unaffected;
  scripts that expect every embedding at the top level need to look in subfolders.

## [0.3.5] - 2026-10-31

### Added

- Arbimon annotations: an `arbimon` format for the CSV layout of Arbimon's pattern-matching ROI
  export (`x1`/`x2` seconds, `y1`/`y2` Hz, `species`, `songtype`, `recording`, `validated`,
  `score`, ...). An input file is recognized from its header wherever annotations are read
  (`annotation convert|summary|filter|remap|generate-labels|generate-frame-labels`, `dataset
  extract-clips`), and `filter`/`remap` keep it in Arbimon layout. Write it with `annotation
  convert ... --to arbimon`, naming the recording with `--recording` (or from a bioamla `.json`
  input's `audio_file`). Library: `load_arbimon_annotations()`, `save_arbimon_annotations()`,
  `ARBIMON_COLUMNS`.
- Annotation files that name each row's recording (a `recording` column, as Arbimon exports
  have) can cover many recordings. `dataset extract-clips` uses only the rows for the audio file
  being cut (matched by name, case-insensitively, with or without the extension), and
  `annotation generate-labels|generate-frame-labels` take `--recording` and refuse a file that
  covers several recordings without it.

### Fixed

- Raven selection tables keep confidence (a `Confidence` column, written when any annotation has
  one and read back when numeric; text values such as `high` stay a custom column) and write
  frequencies to three decimals instead of rounding to 0.1 Hz.

- `audio video -t trajectory`: on narrow frames (e.g. `--size 1080x1920`) the title no longer
  overprints the time: it moves to the line below, and is shortened or left out when even that
  is too narrow. The view is fitted to the 1-99th percentile of the path instead of the 2-98th,
  so far less of a dense, loud passage runs off the frame.
- Detections used as annotations keep their frequency band: the annotation readers accept the
  `frequency_low` / `frequency_high` columns `bioamla detect -o` writes, so `dataset
  extract-clips --annotations det.csv` fills `low_freq` / `high_freq` and `--bandpass` works.
- Detections saved as JSON (`bioamla detect ... -o det.json`, a JSON list) are accepted wherever
  annotations are read. Records without `start_time` / `end_time` (peak detections are points in
  time) are rejected with a clear error rather than read as empty annotations.
- `dataset partition` warns when a split it was asked to fill comes out empty (too few groups to
  go round), and lists those splits as `empty_splits` in its result; its help no longer claims
  the default only writes a `split` column (the default reorganizes into subdirs).
- `dataset stats` and `dataset partition` accept the path of the metadata CSV as well as the
  dataset directory.
- `dataset partition` sizes splits from the fractions without rounding a zero split up: with
  `--test 0` no group goes to test (round-half-to-even could put one there), and split sizes
  can differ by one group from before for the same seed.
- `models ast train --config` help names `models ast init-config`, which creates the file, and
  all three sections it reads.

## [0.3.4] - 2026-10-31

### Added

- `frame_features()` (`bioamla.audio`): per-frame spectral descriptors (level, centroid, spread,
  flatness, crest, rolloff, flux, peak frequency and the band edges within 20 dB of the peak), a
  log-mel spectrogram and MFCCs, with optional stationary noise-floor subtraction. numpy only, so
  it works on the base install. Settings are a validated `FrameFeatureConfig` (schema
  `frame-features`); `FrameFeatures.to_dataframe()` gives one row per frame.
- `mfcc_trajectory()` (`bioamla.audio`): a recording's path through timbre space. Frames above a
  level gate are projected onto the principal axes of their MFCCs (c0 left out), giving
  per-frame coordinates, the explained variance per axis and the axes themselves.
- `bioamla audio features FILE`: the per-frame table (time, descriptors, MFCCs, and `pc1`..`pc3`
  with `--trajectory`) as `-o out.csv|out.parquet`, or a per-descriptor summary on screen.
  `--json` reports a summary, plus every frame when there is no `-o`. Analysis settings are
  options (`--no-denoise`, `--n-fft`, `--hop`, `--n-mels`, `--n-mfcc`, `--f-min`, `--f-max`,
  `--gate-db`).
- `bioamla batch audio features`: one `<name>.features.csv` (or `-f parquet`) per file, mirroring
  the input tree, plus a `frame_features.csv` index; in CSV mode the `features_file` (relative to
  the CSV where possible) and `n_frames` columns are added. Library: `frame_features_file()`,
  `write_frame_table()`, `batch_frame_features_files()` and `frame_feature_table_paths()`.
- `bioamla audio video -t trajectory` and `generate_trajectory_video()` (`bioamla.viz`): the
  MFCC-PCA timbre trajectory drawn as the recording plays, in sync with its audio. The path is
  coloured by spectral centroid and weighted by level, silent stretches break it, and the view
  turns slowly (`--no-orbit` holds it); `--trail SECONDS` fades older path, and a scrolling mel
  spectrogram strip runs along the bottom (`--no-spectrogram` drops it).

### Changed

- The ffmpeg encoding loop behind `generate_spectrogram_video()` is shared with the trajectory
  video. A `--start`/`--end` clip at the file's own sample rate now decodes only that window, so
  a short clip of a long recording no longer loads the whole file.

## [0.3.3] - 2026-10-31

### Added

- `models ast evaluate` reports per-class precision, recall, F1 and support and the files it
  couldn't evaluate (on screen, in every `--format` report and in `--json`), and the confusion
  matrix with its label order (in the `txt` and `json` reports and `--json`). `--format csv` writes
  an overall row and one row per class. `EvaluationResult` gains `labels`, `per_class` and
  `skipped`, and its `confusion_matrix` is now filled in.
- `batch_detect_files()` (`bioamla.detect`): detection over an explicit list of files, writing
  the same `detections_<method>.json` as `batch_detect_dir()` and returning per-file counts.
- `bioamla --json <command>`: machine-readable output for scripts and agents. stdout gets
  exactly one JSON envelope (`{"bioamla", "command", "ok", "result"}`, or `"error": {"type",
  "message"}` on failure), and everything meant for people goes to stderr. `annotation summary`,
  `audio metadata` and `dataset stats` report results so far; the other commands follow.
- `--json` results for `detect energy|ribbit|peaks|accelerating` (`detections`, `count`, the
  options used and any `-o` path), `indices compute|temporal|aci|adi|aei|bio|ndsi|entropy`,
  `audio info` and `audio list`. Reported options are keyed by option name (`--threshold-db` as
  `threshold_db`), not by internal parameter name.
- `--json` results for `models ast predict` (predictions with times, label and confidence),
  `evaluate` (the metrics), `embed` (output path and shape; the vectors stay in the file) and
  `info` (every class label), and for `cluster reduce|fit|analyze|novelty` (shapes, cluster sizes,
  quality scores, novel sample indices).
- `--json` results for the `annotation` commands (`template`, `convert`, `remap`, `filter`,
  `generate-labels`, `generate-frame-labels`: counts, labels, output paths) and the `dataset`
  commands (`merge`, `extract-clips`, `manifest`, `partition`, `build`, `license`, `augment`,
  `remix`: their stats as returned by the library).
- `--json` results for the 19 `batch` commands: counts, timing, output files and each failure's
  file and error, the output directory, `results_file` (the aggregate file the command wrote,
  only if it exists) and, in CSV mode, `metadata_csv`, the updated metadata CSV. `batch audio
  validate` with invalid files is a failure envelope with the batch result in `error.details`.
- `batch_generate_spectrograms()` also returns `output_files` and `errors`.
- `--json` results for the rest of the CLI: the `audio` transforms, `segment`, `visualize`
  and `video` (input, output, options), `system version|devices|deps|schema`,
  `util download|unzip|zip`, `models ast init-config|train` (the `TrainResult`) and all 15
  `catalogs` commands (search results, downloads, Hub pushes and pulls, cache contents). Every
  command now reports a result; `audio play` refuses `--json`, and `system deps --install` and
  `catalogs hf cache --purge` need `--yes` with it (a purge that fails for some repos is a
  failure envelope with the details). Per-file results all use `file`/`output`, and
  `inat search` reports the same rows it saves to CSV. A test walks the command tree so a new
  command can't skip the contract.
- `get_model_info()` returns `labels`, every class label in class-index order. `classes` is still
  the first 10.

### Fixed

- `models ast evaluate --batch-size` was accepted and ignored; `evaluate_directory()` now takes
  `batch_size` and classifies files in batches. Files that can't be loaded are listed instead of
  silently dropped from the sample count.
- AST feature extraction passed multi-channel audio to the feature extractor as several inputs,
  so a stereo file failed or was misread in `predict`, segmented predict, embeddings and evaluate.
  Channels are now averaged to mono first (`bioamla.ml.ast.mono_waveform`).
- CSV-mode `batch detect` ran every detector and threw the detections away. Each row now gains
  a `num_detections` column (blank for a file that failed), and the detections are written to the
  same `detections_<method>.json` as in directory mode, beside the output CSV.
- CSV-mode `batch cluster` wrote its input CSV back without the cluster labels. Each row now gains
  `cluster` and `cluster_labels` (`;`-separated; both blank for a file that wasn't clustered).
- `batch cluster` exited 0 when clustering failed or no embedding could be loaded; it now fails
  with a `ClusteringError`.
- A batch command's `results_file` no longer names a file left over from an earlier run when
  this run wrote nothing.
- `catalogs inat search --project-id` was accepted but never passed on, so it searched without
  the project filter; `inat.search()` now takes `project_id`.
- `models ast init-config` on an existing file and a failed `system deps --install` printed
  an error and exited; they raise a regular CLI error now, so `--json` gets the message.
  `system deps`'s help examples named a `bioamla config deps` command that doesn't exist.
- The `annotation` and `dataset` commands that printed an error and exited now raise a regular
  CLI error: the same `Error: ...` line for people, and the actual message (not "exited with
  status 1") in a `--json` failure envelope. `dataset license --batch` with failed datasets now
  names them in the error, and its `--json` envelope carries the per-dataset results in
  `error.details`.
- `annotation generate-labels` miscounted clips through float error (0.3 s of audio in 0.1 s
  clips gave 2, not 3), reported a negative count when the clip was longer than the audio, and
  crashed on `--hop-length 0`. Those inputs are now errors, and the count is right. With
  `--format numpy`, both label commands report the `.npy` and `.labels.csv` files they actually
  write, and a failed write is an error, not a traceback.
- `dataset augment` range options accept negative ranges written with a dash (`-2-2`,
  `-6--1`) and report unusable input as a bad option value instead of a traceback.
- The `cluster` commands report an unreadable `.npy` input (missing, empty or the wrong shape)
  as `InvalidInputError` and a failed write as an error, instead of a traceback;
  `models ast predict|evaluate|embed` do the same for a failed write.
- `get_model_info()` listed classes in the config's key order, which for more than 10 classes is
  often string order ("0", "1", "10", "2", ...); they now follow the class index.
- `models ast evaluate --format csv|txt -o FILE` wrote a Python dict repr. `csv` now writes a
  header and one row of the scalar metrics; `txt` one `name: value` line per metric.

### Changed

- `cluster_embedding_files()` and `cluster_batch_files()` raise `ClusteringError` when clustering
  fails or no embedding file can be loaded, instead of returning a result with the error in
  `errors`. They cluster a file listed twice only once, and each entry in
  `metadata["file_assignments"]` (and `cluster_assignments.json`) has a `cluster`: the most
  common non-noise label, ties to the smallest.
- The `annotation` commands check that their input file exists through Click, so a missing
  file is a usage error (exit status 2, `Error: Invalid value for 'INPUT_FILE': ... does not
  exist`) instead of exit status 1.

### Deprecated

- `catalogs inat search --has-sounds`: it never had an effect; the search only returns
  observations with sounds.
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

[Unreleased]: https://github.com/jmcmeen/bioamla/compare/v0.3.8...HEAD
[0.3.8]: https://github.com/jmcmeen/bioamla/compare/v0.3.7...v0.3.8
[0.3.6]: https://github.com/jmcmeen/bioamla/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/jmcmeen/bioamla/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/jmcmeen/bioamla/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/jmcmeen/bioamla/compare/v0.3.2...v0.3.3
[0.3.1]: https://github.com/jmcmeen/bioamla/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/jmcmeen/bioamla/compare/v0.2.26...v0.3.0
[0.2.15]: https://github.com/jmcmeen/bioamla/compare/v0.2.14...v0.2.15
[0.2.10]: https://github.com/jmcmeen/bioamla/compare/v0.2.9...v0.2.10
[0.2.9]: https://github.com/jmcmeen/bioamla/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/jmcmeen/bioamla/compare/v0.2.7...v0.2.8
[0.2.4]: https://github.com/jmcmeen/bioamla/compare/v0.2.3...v0.2.4
[0.2.0]: https://github.com/jmcmeen/bioamla/releases/tag/v0.2.0

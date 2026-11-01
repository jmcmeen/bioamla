# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `models ast predict --top-k K` ranks each prediction's K most likely labels: the JSON
  gains `top` (`[{label, confidence}, ...]`, best first) and `margin` (top-1 minus top-2
  confidence), the CSV gains `margin` and `top` columns, and the human output lists them.
  Without the option the output is unchanged. In the library: `classify_file(top_k=...)`,
  `ASTInference.predict(top_k=...)`/`predict_segments(top_k=...)`, and
  `ASTPredictionResult.margin`/`top()`. A negative `top_k` is refused when the request
  is checked. The labeling skill now records each candidate's `margin`, and gives no
  label to an event that no prediction segment covers (it took the first segment's).

- `batch_generate_spectrograms()` gains `show_colorbar` and `continue_on_error`,
  `batch_visualize()` gains `legend`, and the `batch.audio.visualize` request a `legend` field.

- `audio convert --bit-depth` and the new `batch audio convert --bit-depth` write that many bits
  per sample: 8, 16, 24 or 32 for WAV, 8, 16 or 24 for FLAC. A depth the format can't hold,
  or any depth for MP3 or OGG, is refused before anything is written; the option used to be
  accepted and ignored (#239). **Behavior change:** without it, a conversion keeps the
  source's bit depth when the output format can hold it (a 24-bit WAV to FLAC stays 24-bit;
  it used to drop to 16), else writes 16 bits. The library gains `save_audio(bit_depth=...)`
  and `convert_audio_file(target_bit_depth=...)`; the `audio.convert` and
  `batch.audio.convert` requests gain a `bit_depth` field. WAV and FLAC are now always
  written through soundfile, so FLAC output no longer needs ffmpeg.

- `bioamla.audio.load_audio_channels()`: load a file keeping its channels (1-D for mono,
  `(n_samples, n_channels)` otherwise), where `load_audio()` downmixes to mono.

- Agent skills: a convention for skills that let an AI agent drive bioamla
  (`skills/<name>/SKILL.md` and `scripts/`). Scripts run `bioamla --json` and act on the
  JSON (so a skill works locally or against a server), decide by thresholds on objective
  measurements, send anything below threshold to a person, write to
  `runs/<skill>/<timestamp>/`, and use only the standard library. A test harness runs each
  skill's scripts in a synthetic workspace as a pass/fail eval. The first skill,
  `inventory`, lists a folder's recordings and flags short or unreadable files. See the new
  "Agent skills" docs page.
- The `label` agent skill, for a first pass of labeling, one script per step. `candidates.py`
  detects events in a frequency band (`detect energy`) and pairs each with the AST
  prediction of the segment it overlaps most, flagging labels outside the given
  vocabulary. `evidence.py` measures each candidate: signal-to-noise ratio in the
  detector's band (from `audio features`), detector score, model confidence, band and
  duration. `crops.py` cuts a clip and a spectrogram image around each candidate,
  recording the event's box in the clip. `annotate.py` accepts the candidates that clear
  confidence, SNR and detector-score thresholds and merges them into an annotations CSV
  with `confidence`, `method` and `labeled_by=agent`; rows a person made are never
  changed, and earlier agent rows for the same recordings are replaced. `review.py`
  queues every candidate below a threshold, with its reasons, evidence, clip and image,
  in `review_queue.json`. An end-to-end eval runs the five steps on synthetic recordings with
  known events and checks what is annotated, what is queued, and that a person's row is
  untouched.

- "Running the server" docs page: installing and starting `bioamla serve`, the token file,
  reaching it over an SSH tunnel or a VPN, a systemd unit, and linking recordings already on
  the machine with `--allow-link`.

- `audio visualize --box START,END[,LOW,HIGH][,LABEL]` (repeatable) and `--annotations FILE`
  outline regions on the picture: a box spans its band on the STFT and mel spectrograms, and
  the full height without one (and on MFCC and waveform plots); labels are written at its
  corner, in white or black to suit the colormap. The picture underneath is unchanged.
  `--annotations` reads any format bioamla reads (Raven, bioamla JSON, CSV, Arbimon; an
  export covering several recordings gives only this one's rows, and a row that can't be a
  box is skipped with a warning), so `detect ... -o det.csv` feeds it directly. The `audio.visualize` request
  gains `boxes` and `annotations`; the library gains `generate_spectrogram(boxes=...)`,
  `bioamla.viz.SpectrogramBox` and `bioamla.viz.annotation_boxes()`.

- Integration tests run a pipeline on a real server in CI (upload, detect, annotation
  summary, clip extraction, dataset stats, download), once through `bioamla.client` and once
  through `bioamla --remote`, each step reading the paths the previous one reported, and
  checking streamed progress and the server workspace's history (nothing recorded locally).

- `bioamla --remote NAME:WORKSPACE <command>` runs the command on a server: the CLI builds
  the operation's request as usual, the server runs it, progress and notices stream back,
  and the command prints exactly what a local run prints (`--json` included; parity is
  tested; paths in results come back relative to the remote workspace, as a local run
  reports them). The remote can also come from `BIOAMLA_REMOTE`, or from top-level `remote` and
  `workspace` settings in a local workspace. Paths are the server's (local existence checks
  are skipped). Local-only commands refuse an explicit `--remote`; command-line credentials
  are never sent; remote runs are recorded in the server's history, not locally.
- `bioamla.common.wire` (`to_wire()`, `from_wire()`): results as JSON that turns back into
  the same objects (bioamla dataclasses and enums, paths, NumPy arrays, tuples, NaN, ...);
  rebuilding only constructs bioamla types. The server sends it with `?wire=true` (operation
  routes and `GET /v1/jobs/{id}`), and `Client.run(typed=True)` returns the rebuilt objects
  (`UntypedResult` when this machine can't load the result's type); with
  `workspace_relative=True`, paths inside the workspace come back relative to it.

- `bioamla.client`: work on a server from another machine. `Client(url, token)` lists and
  creates workspaces, reads and changes their settings, lists, uploads and downloads files,
  runs operations (`run()` waits for a job, following its events and progress; `submit()`
  doesn't; a dropped event stream is resumed after the last event), and lists, follows and
  cancels jobs (by id or a unique prefix). A refusal raises `RemoteError` with the
  server's message (`RemoteNotFound`, `RemoteInvalidInput`; `JobFailed` for a failed or
  cancelled job). It needs only `requests`, so the base install can drive a server.
- Named remotes in the user's settings file (`[remotes.NAME]` with `url` and `token_env`;
  the token stays in the environment variable): `add_remote()`, `remotes()`,
  `get_remote()`, `remove_remote()`, `connect()`.
- `bioamla remote add|list|remove|ping|workspaces|upload|download|jobs|cancel`. Reads of a
  server (`list`, `ping`, `workspaces`, `jobs`) aren't recorded in a local workspace's
  history.

- The server's OpenAPI document is committed as `docs/api/openapi.json` (its version is
  the API's, `v1`), with a test that fails while it's out of date. Regenerate it with
  `bioamla serve --openapi docs/api/openapi.json`, which writes it and exits (it needs no
  other option, and refuses to write a partial document when an extra is missing);
  `bioamla.server` has `openapi_spec()` and `write_openapi_spec()`.

- Server workspace and file routes: list the workspaces under the server root and create
  one (`GET`/`POST /v1/workspaces`), a workspace's status (`GET /v1/workspaces/{ws}`), its
  settings (`GET .../config`, `PUT`/`DELETE .../config/{key}`, keys checked against the
  operations' request fields), and its files: list a folder (`GET .../files?path=&recursive=`),
  download (`GET .../files/{path}`) and upload (`PUT .../files/{path}`, streamed to disk,
  `?overwrite=true` to replace; the file appears whole or not at all, and two uploads of a
  new file can't both win). Every path must stay inside the workspace root or a linkable
  folder; uploads go only into the workspace named in the URL. Listings leave out
  workspace metadata and unfinished uploads, and stop at 10,000 entries (`truncated`).
- `Workspace.settings()`, `set_setting()`, `unset_setting()` (one spelling per key: `-`
  and `_` are alike) and `list_files()`; `bioamla.operations.check_setting_key()`,
  `bioamla.common.settings.check_setting_value()` and
  `bioamla.common.files.AtomicFileWriter`.

- Server jobs: a job-kind operation posted to the server is queued and answered at once
  (`202`, with the job), then runs in its own worker process, so a crash or a cancel can't
  take the server down. `GET /v1/jobs` and `/v1/jobs/{id}` report status, progress and the
  result or error; `/v1/jobs/{id}/events` streams progress, notices and log lines as
  server-sent events (resumable with `Last-Event-ID`); `/v1/jobs/{id}/log` gives the log as
  text; `POST /v1/jobs/{id}/cancel` drops a queued job or stops a running one (its cancel
  check first, then the worker is ended). At most `bioamla serve --max-jobs` jobs run at
  once (default 2), and one machine-learning job at a time. A request with a path outside
  the allowed folders is refused (`403`) before it's queued. Each job is recorded in its
  workspace's history with its id.
- `bioamla.operations.prepare()`: a request validated and its paths checked, as `run()`
  would run it, without running it.

- Every `cluster` and `models ast` command runs through an operation (`cluster.reduce|fit|
  analyze|novelty`, `models.ast.init-config|predict|train|evaluate|embed|info`), so every
  command group now has its operations. Training reports progress (steps) through the
  operation context and stops cleanly when cancelled.
- Library functions behind them: in `bioamla.cluster`, `reduce_embeddings_file()`,
  `cluster_embeddings_file()`, `analyze_clusters_file()`, `novelty_file()`,
  `load_array_file()` and `save_array_file()`; in `bioamla.ml`, `classify_file()`,
  `evaluate_model()` (with `evaluation_text()` and `evaluation_csv_rows()`), `embed_file()`,
  and `train_ast_from_options()` with `resolve_train_options()` and `AST_TRAIN_DEFAULTS`
  (explicit options, then a config file, then defaults; the rule `ast train` follows; an
  option of the wrong type, or a probability outside 0-1, is refused up front).

- Every `catalogs` command runs through an operation (`catalogs.inat.*`, `catalogs.hf.*`,
  `catalogs.xc.*`, `catalogs.ml.*`, `catalogs.ebird.*`): searches and lookups are quick,
  downloads, pushes and pulls are jobs. API keys never travel in a request: an operation
  takes a key from its context's secrets (locally, a flag such as `--api-key`), else from the
  environment where it runs. `catalogs.hf.cache-purge` is local-only.
- `OperationContext.secrets` and `OperationContext.on_notice`: credentials for a run that
  never enter a request, and messages an operation sends while it runs (the CLI prints
  them; `hf push-dataset` reports its card and location warnings this way, before the
  upload).
- Library functions behind them, in `bioamla.catalogs`: `inat_search()` (rows, optional
  CSV), `inat_project_stats()` (optional JSON), `push_dataset_folder()` (card, location
  warnings, push) with `dataset_push_warnings()`, and `ebird_nearby()` (optional CSV).

- `system`, `util` and `workspace` commands run through operations (`system.version|devices|
  deps|deps-install|schema|export-schemas`, `util.download|unzip|zip`, `workspace.init|status|
  add|import|list|remove|history`), so every command that does work has one.
  `system.deps-install` and `util.download` are local-only. `system commands` and
  `workspace config` stay command-line only: they describe and configure the CLI itself.
- Library: `bioamla.workspace.current_workspace()`, `Workspace.artifact_row()`,
  `artifact_rows()` and `status()`; `bioamla.common.files.download_to_dir()`, `unzip_to()`
  and `zip_path()`; `bioamla.common.config.config_schema_report()` and
  `export_config_schemas()`; `bioamla.system.install_missing()`.

- Server: one endpoint per remote-safe operation, generated from the registry:
  `POST /v1/workspaces/{ws}/operations/{name}` with the operation's request as the body (its
  schema is in the OpenAPI spec). Every path in a request is relative to the workspace and
  must resolve inside the server's allowed folders; quick operations answer with the result
  envelope; local-only operations have no route. `GET /v1/operations` lists them with their
  request schemas and the domains a missing extra makes unavailable. Each run is recorded in
  the workspace history (`via: "server"`, with the operation and request), failed or not,
  except the reads of the workspace itself (`workspace.status|list|history`). A server run
  works on its own workspace: a request's `directory` must be inside it.
- `Workspace.output_paths()`: the files a run's result says it wrote (what history records),
  and `Workspace.run_record()`, the history record that both the CLI and the server write.

### Changed

- **Breaking:** `annotation generate-labels` and `generate-frame-labels` with `--format
  numpy` name the label map after the array file: `run.v1` writes `run.v1.npy` and
  `run.v1.labels.csv` (both used to write `run.labels.csv`, so outputs that differed only
  before their last dot overwrote each other's map), and a `.csv` output name becomes
  `.npy`: `labels.csv` writes `labels.npy` (it was `labels.csv.npy`) and
  `labels.labels.csv`. Names ending in `.npy`, or without a dot, are unchanged. The human
  output now shows the label map's path. `bioamla.common.files.npy_path()` gives the file
  `numpy.save` writes for a name.

- **Behavior change:** `audio visualize` and `batch audio visualize` draw the same picture
  of a file: one set of defaults (150 dpi, the `magma` colormap, 40 MFCCs, as `bioamla.viz`
  already used), so `audio visualize` pictures are larger (was 100 dpi) and use another
  colormap (was `viridis`) and MFCC count (was 20). `batch audio visualize` gains the single
  command's drawing options (`--n-fft`, `--hop-length`, `--n-mels`, `--n-mfcc`, `--cmap`,
  `--dpi`), as does its request; the defaults live in `bioamla.common.constants`
  (`PICTURE_N_MELS`, `PICTURE_N_MFCC`, `PICTURE_CMAP`, `PICTURE_DPI`). A bad setting (an
  unknown colormap, more MFCCs than mel bands, a size below 1) is refused before any file
  is drawn (`bioamla.viz.check_picture_settings()`), and `--n-mels` now shapes MFCC
  pictures too (it was ignored for `-t mfcc`).

- **Behavior change:** `audio trim`, `audio resample` and `audio gain`, `batch audio
  resample` and `batch audio trim` by time range keep a file's channels; they mixed every
  file down to mono. `batch audio trim --trim-silence` still mixes down, so every channel is
  cut the same. The library functions behind them (`trim_audio_file()`,
  `resample_audio_file()`, `gain_audio_file()`, `batch_resample()`) do the same, and
  `batch_transform()`, `batch_transform_files()` and `process_file()` take
  `keep_channels`. New: `bioamla.audio.each_channel()`.

- **Behavior change:** `audio segment` and `batch audio segment` cut the same clips (#240):
  named `<stem>_seg0000.<format>` (`--prefix` replaces the stem for `audio segment`), every
  clip exactly the duration long, with a shorter last clip left out unless `--keep-short`,
  in WAV, MP3, FLAC or OGG (`batch audio segment` gains `--format`, and `-d` defaults to 3 s
  there too). `audio segment` used to name clips `<prefix>_0000`, and `batch audio segment`
  kept the short last clip and wrote only WAV. A file shorter than one clip is an error that
  names `--keep-short` (counted as a failed file in a batch) rather than silently giving no
  clips. `segment_audio_file()` leaves the short clip out by default (`include_partial=True`
  keeps it) and loses its `label` parameter (clips are always `_seg0000`); the
  `audio.segment` and `batch.audio.segment` requests gain `include_partial` (and the batch
  one `format`).

- **Behavior change:** `audio convert` and `batch audio convert` keep a file's channels: a
  stereo file stays stereo (each channel as it was) unless `--channels` says otherwise.
  They used to downmix every file to mono, and `-c 2` duplicated that downmix into both
  channels (#238). A file with more than two channels converted to MP3 (which holds two)
  needs `--channels 1` or `2`; a channel count other than 1 or 2 is refused up front.

- `cluster reduce|fit` say "Reducing embeddings to 2D using pca..." / "Clustering embeddings
  using hdbscan..." (the input size is read inside the operation now).
- The `ast init-config` "already exists" error names the fix itself (`force=True` /
  `--force`).
- `workspace status` and `workspace list` are no longer recorded in the workspace history
  (like `workspace history`): they only read it, and recording them buried the real runs.
- `util unzip` checks every entry before writing anything: an entry that would leave the
  folder (including through a link already in it) or touch a `.bioamla` folder is refused,
  as is an archive that would extract to more than 50 GiB (`unzip_to(max_bytes=...)`).
- `models.ast.train` puts the base model and a local MLflow store (a path or `file:` URI)
  through the path check, including when they come from the config file, and a model name
  that fails the check is refused rather than retried as a Hugging Face ID.

### Removed

- `models ast train --id-column`, which training never used: labels are numbered from the
  sorted `--label-column` values (#254). Passing it is now a usage error.

Deprecated in 0.3.x:

These are breaking changes:

- **Python 3.10 support.** bioamla now requires Python 3.11 or newer. 3.10 reached end of life in
  October 2026, and the security fixes for pillow and filelock need 3.11. The `tomli`
  dependency is gone (`tomllib` is in the standard library from 3.11).

These were deprecated in 0.3.x and are breaking changes:

- The old `AugmentationConfig` names: `bioamla.datasets.AugmentationConfig` (use
  `WaveformAugmentationConfig`) and `bioamla.ml.AugmentationConfig` (use
  `SpectrogramAugmentationConfig`), with their aliases in `bioamla.datasets.augmentation`
  and `bioamla.ml.preprocessing`.
- The per-command `--json` flag of `annotation summary`, `audio metadata` and
  `dataset stats`, which printed bare JSON. Use `bioamla --json <command>`; a `--json`
  after the command is now a usage error.
- `inat search --has-sounds`, which did nothing (searches only return observations with
  sounds).
- `bioamla.cli.output.legacy_json`, the helper behind those flags.

### Fixed

- `models ast train` failed with FFmpeg 9 (and any FFmpeg that torchcodec can't load):
  `datasets` decoded the training audio through torchcodec. Training audio is now decoded
  by bioamla's own loader (`bioamla.ml.training.decode_example_audio`), local datasets keep
  plain file paths, and a Hub dataset's audio column is left undecoded for it (#256). A
  folder of class subfolders is read by bioamla too, so its classes keep their folder
  names as labels. Training with DataLoader workers on macOS and Windows (spawned workers)
  failed to pickle the dataset's transform; it is now a module-level
  `bioamla.ml.training.BatchPreprocessor`.

- Conversion and segmenting edge cases: `batch audio convert` refuses an unsupported
  channel count (as well as a bit depth) before writing anything (`check_convert_options()`
  in `bioamla.audio.convert`); `--bit-depth`'s help gives the real default (the source's
  when the output holds it, else 16); a 32-bit float WAV keeps 32 bits and a 32-bit source
  to FLAC gets 24, instead of 16; MP3 with more than two channels is refused by
  `save_audio()` for every caller; segmenting with `--keep-short` and an overlap writes one
  shorter last clip, not several; an empty file is reported as such. A step that would
  give a file's channels different lengths raises `ProcessingError`.

- `batch audio visualize --no-legend` draws the bare picture (no axes, title or colorbar);
  it used to be accepted and ignored (#246). `batch.audio.visualize` also honours
  `continue_on_error`, which it ignored.

- `convert_audio_file()` called without `target_format` wrote WAV data whatever the output's
  extension (a `.flac` path got a WAV file); it now takes the format from the extension.

- A 32-bit PCM file read through ffmpeg (formats soundfile can't read) that was nearly
  silent was taken for float audio and came out at full scale; it is scaled as integers.

- `audio list` of a folder that doesn't exist (and `list_audio_files()`) raises
  `NotFoundError` instead of reporting an empty folder (#262).

- A server request can no longer read or write a workspace's metadata (its `.bioamla`
  folder and the files directly in it) through an operation's path fields; the path
  policy refuses them (403). Subfolders such as `.bioamla/imported` are still allowed.
- `util zip` of a file dated before 1980, `util download` of a malformed URL and `system schema
  NAME -o` to an unwritable path stopped with a traceback; they report an error. A missing
  archive for `util unzip` is a not-found error.
- `cluster reduce|fit|novelty` and `models ast embed` with an output name without `.npy`
  reported that name, though NumPy wrote `NAME.npy`; they report the file written.
- `models ast evaluate` and single-file AST prediction failed on Apple Silicon: the input
  stayed on the CPU while the model ran on MPS (#255).

## [0.3.11] - 2026-11-01

### Added

- Every `batch audio` command runs through an operation (`batch.audio.info|validate|metadata|
  features|convert|resample|normalize|trim|filter|denoise|segment|visualize`), in directory and
  CSV mode, with unchanged output. Each takes a `BatchRequest` (`input_dir` or `input_file`,
  `output_dir`, `recursive`, `max_workers`, `continue_on_error`) plus its options; all are jobs
  and report progress where the library does.
- Library functions behind them, each handling both modes from a `BatchConfig`, in
  `bioamla.audio.batch` (also exported from `bioamla.audio`): `batch_audio_info()`,
  `batch_validate_wavs()` (invalid files in `metadata["invalid_files"]`),
  `batch_recording_metadata()`, `batch_frame_features()`, `batch_convert()`,
  `batch_resample()`, `batch_transform()` with `normalize_processor()`, `trim_processor()`,
  `filter_processor()` and `denoise_processor()`, `batch_segment()` and `batch_visualize()`.
- The CSV-mode runners in `bioamla.batch`: `run_csv_merge()`, `run_csv_transform()`,
  `run_csv_process()`, `run_csv_segment()` and `run_csv_predict_segments()`, and
  `BatchRequest`. Operation names may now nest (`batch.audio.*`): the registry loads the module
  for the longest registered prefix, so one domain's missing extra doesn't hide another's.

- `batch detect energy|ribbit|peaks|accelerating`, `batch index`, `batch models predict|embed`
  and `batch cluster` run through operations (`batch.detect.*`, `batch.index`,
  `batch.models.*`, `batch.cluster`), so every batch command now does; output is unchanged.
- Library functions behind them, each handling both modes from a `BatchConfig`:
  `bioamla.detect.batch_detection()`, `bioamla.indices.batch_indices()`,
  `bioamla.ml.batch_predict()` and `batch_embed()` (which write `predictions.json` /
  `predictions.csv` and the embeddings themselves), and `bioamla.cluster.batch_cluster()`;
  and `bioamla.batch.config_csv_context()`, `require_input_dir()` and `require_output_dir()`.

- Every `annotation` command runs through an operation (`annotation.template|convert|summary|
  remap|filter`, quick; `annotation.generate-labels|generate-frame-labels`, jobs), with
  unchanged output apart from error messages, which now name the parameter and its flag.
- Library functions behind them, in `bioamla.datasets`: `annotation_template()`,
  `convert_annotation_file()`, `summarize_annotation_file()`, `remap_annotation_file()`,
  `filter_annotation_file()`, `clip_labels_file()` and `frame_labels_file()`, with the result
  types `AnnotationTemplate`, `ConvertedAnnotations`, `AnnotationFileSummary`,
  `RewrittenAnnotations` and `GeneratedLabels`.

- Every `dataset` command runs through an operation (`dataset.merge|extract-clips|stats|manifest|
  partition|build|license|augment|remix`; `stats` is quick, the rest are jobs), with unchanged
  output.
- Library functions for the multi-step ones, in `bioamla.datasets`: `write_dataset_manifest()`,
  `build_dataset()` (extract, partition, manifest and attributions in one call),
  `write_dataset_license()`, `augment_dataset()` (ranges as `(min, max)` tuples) and
  `remix_from()` (a recipe plus overrides).

### Changed

- `batch audio visualize --no-legend` prints a warning that it's ignored; it always was (#246).
- `annotation generate-labels|generate-frame-labels` problems (no annotations, no duration, an
  ambiguous recording, a clip longer than the audio) are `InvalidInputError` in the `--json`
  envelope, naming the parameter (`audio_duration`, `recording`, ...).
- `batch models predict --input-file` reports where it wrote the metadata CSV, like the other
  CSV-mode commands.
- Batch directory mode honours `continue_on_error` in `batch_audio_info()`,
  `batch_validate_wavs()`, `batch_recording_metadata()` and `batch_segment()`.

### Fixed

- `batch audio resample|normalize|trim|filter|denoise --input-dir` crashed before processing
  anything (`batch_transform_files()` got an unexpected `operation` argument); it now runs, and
  `batch_transform_files()` takes `operation` for its log records like the other batch runners.
- Spectral denoising (`audio denoise`, `batch audio denoise`, `spectral_denoise()`) no longer
  drops the last few milliseconds of the audio.
- `annotation generate-frame-labels` dropped the last frame to float error (0.3 s in 0.1 s
  frames gave 2) and wrote an empty file for a frame longer than the audio; it keeps the last
  frame and refuses a non-positive or too-long frame (#250). With `--format numpy` both label
  commands print the file they wrote (`<name>.npy`).
- `OptionConflictError` (an `InvalidInputError`): arguments that can't be used together; the CLI
  reports it as a usage error.
- `batch models predict` without `--output-dir` ran the model on every file and kept nothing;
  predictions now go beside the input (the input folder, or the metadata CSV's folder), as
  `batch detect`'s do. In CSV mode `predictions.json` follows the CSV's order, also with
  `--max-workers` above 1.

## [0.3.10] - 2026-11-01

### Added

- Every `detect` and `indices` command runs through an operation (`detect.energy|ribbit|peaks|
  accelerating`, `indices.compute|temporal|aci|adi|aei|bio|ndsi|entropy`), so the server can run
  them without the CLI. Output and `--json` results are unchanged. Each is declared a `job`,
  since its cost grows with the recording's length.
- Library functions behind those commands, for use without the CLI:
  - `bioamla.detect`: `detect_peaks_file()` and `FilePeaks`, `export_peaks()`,
    `PeakDetection.to_event()` and `PEAK_EVENT_COLUMNS` (the peak rows the command reports and
    saves).
  - `bioamla.indices`: `index_file()` (every index of one file, optionally saved as JSON),
    `temporal_index_file()` (indices over consecutive windows), `index_values_file()` (selected
    indices, loading the file once; index parameters only with a single index), with
    `FileIndices`, `FileTemporalIndices`, `FileIndexValues` and `SEGMENT_COLUMNS`.
  - An empty output path saves nothing, as the commands always treated it.

- Every `audio` command runs through an operation (`audio.info|metadata|list|play|features|
  convert|segment|trim|normalize|resample|filter|denoise|pitch-shift|time-stretch|add-noise|
  gain|visualize|video`), with unchanged output. Reading a header or listing a folder is
  `quick`; anything that decodes audio is a `job`; `audio.play` is local-only. `audio.video`
  reports progress through the operation context and stops when cancelled.
- Library functions behind those commands, in `bioamla.audio`: `trim_audio_file()`,
  `normalize_audio_file()`, `resample_audio_file()`, `filter_audio_file()`,
  `denoise_audio_file()`, `pitch_shift_audio_file()`, `time_stretch_audio_file()`,
  `add_noise_audio_file()`, `gain_audio_file()` (each returns a `ProcessedFile`),
  `describe_audio_file()`, `recording_metadata_report()`, and the result types
  `AudioFileDescription`, `RecordingMetadataReport`, `AudioFileList` and `SegmentedFile`.
- `segment_audio_file()` takes `format`, `include_partial` and `label`, and
  `convert_audio_file()` takes `target_format=None` (format from the output name), so
  `audio segment` and `audio convert` use them instead of their own copies.

### Changed

- `detect.energy` is declared a `job` rather than `quick`, like the other per-file analyses.
- `audio convert --bit-depth` prints a warning that it's ignored; it always was (#239).
- `audio filter`, `audio trim` and `audio segment` report bad option combinations as
  `InvalidInputError` in the `--json` envelope, naming the parameters (`end`, `duration`,
  `bandpass_low`, ...) rather than the flags. `audio trim` refuses a negative `--start` or a
  non-positive `--duration`; `audio filter` treats a 0 Hz cutoff as given.
- `segment_audio_file()` (so `audio segment` and `batch audio segment`) refuses a `prefix`
  containing a path, so clips stay inside the output folder, and creates the output folder
  only after the audio loads.
- `audio video --cmap` has no fixed default in `--help`: each renderer picks its own (magma,
  or plasma for `-t trajectory`), as before.

### Fixed

- `indices ndsi --n-fft` (and `compute_index(..., "ndsi", n_fft=...)`) is used instead of being
  ignored; NDSI's default stays 1024, so default results don't change.
- `temporal_indices` (and `indices temporal`) raises `InvalidInputError` for a window or hop
  shorter than one sample instead of looping forever.
- `detect peaks -o name.json` writes JSON, as the other detectors do, instead of CSV.

## [0.3.9] - 2026-11-01

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
- `bioamla serve` and the `server` extra (`pip install 'bioamla[server]'`: fastapi, uvicorn):
  the start of the HTTP API (`/v1`) for working in this machine's workspaces from elsewhere.
  This release has `/v1/health` (no token), `/v1/version` and the OpenAPI spec at
  `/v1/openapi.json` (token required; it declares the bearer scheme, and there are no
  interactive docs pages); responses and errors, unknown routes included, use the `--json`
  envelope. One user, one bearer token (a file created atomically, with mode 600 on macOS and
  Linux; a malformed or shared token file is refused), localhost by default; every path must
  resolve inside the workspace root or a folder allowed with `--allow-link`, and a workspace's
  `.bioamla` metadata can't link outside it. The server doesn't need the CLI.
  Library: `bioamla.server`; `to_jsonable` now lives in `bioamla.common.jsonable`.
- `bioamla.operations`: a registry of operations, the shared contract the CLI and the server
  call. Each operation has a typed request (a validated config whose path fields are marked
  `PathStr` and appear in its JSON Schema with `"format": "path"`), a function declared in its
  domain package, and a result. `operations.run(name, request, context)` validates the request,
  passes every path through the context's check, and runs it; `OperationContext` carries the
  workspace, progress and cancel hooks. Domains load on first use, and `all_operations()` lists
  what's available and which domains need an extra. Request paths also accept `pathlib.Path`
  and `list[PathStr]`; a path hidden where it can't be checked is refused at registration. `detect energy` is the first command run
  through one (`detect.energy`), with unchanged output.
- `bioamla.detect.detect_file(detector, file, output=None)`: run any detector on one file and
  optionally save the detections (JSON for `.json`, otherwise CSV), returning `FileDetections`
  (`file`, `detections`, `count`, `output`, `to_dict()`).

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

[Unreleased]: https://github.com/jmcmeen/bioamla/compare/v0.3.11...HEAD
[0.3.11]: https://github.com/jmcmeen/bioamla/compare/v0.3.10...v0.3.11
[0.3.10]: https://github.com/jmcmeen/bioamla/compare/v0.3.9...v0.3.10
[0.3.9]: https://github.com/jmcmeen/bioamla/compare/v0.3.8...v0.3.9
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

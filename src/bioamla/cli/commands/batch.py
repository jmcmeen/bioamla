"""Batch processing operations.

Each subcommand supports two input modes:

- Directory mode (``--input-dir``): discover and process every audio file under
  a directory, writing results (transformed audio, aggregated JSON/CSV) to
  ``--output-dir``.
- CSV-metadata mode (``--input-file``): process each file listed in a metadata
  CSV (paths resolved relative to the CSV's directory) and write the results
  back to a CSV — in-place when no ``--output-dir`` is given, else into
  ``--output-dir`` — merging new result columns, updating processed-file paths,
  or expanding one input row into many for segmentation.

Subcommands wire to the NEW domain helpers (audio / detect / indices / ml /
cluster) and the generic engine in :mod:`bioamla.batch`. Each body wraps its
call in ``try/except BioamlaError -> ClickException`` so the central CLI error
handler reports failures cleanly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import click

from bioamla.cli.batch_options import batch_input_options, batch_output_options
from bioamla.cli.options import (
    frame_feature_options,
    max_workers_option,
    quiet_option,
    recursive_option,
)
from bioamla.exceptions import BioamlaError, InvalidInputError

if TYPE_CHECKING:
    # Kept out of module load so `bioamla --help` doesn't import numpy or the
    # batch engine; the helpers below import what they use when they run.

    from bioamla.batch import BatchConfig, CSVBatchContext

# =============================================================================
# Shared helpers
# =============================================================================


def _build_config(
    input_dir: str | None,
    input_file: str | None,
    output_dir: str | None,
    recursive: bool,
    max_workers: int,
    quiet: bool,
) -> BatchConfig:
    """Build a :class:`BatchConfig` from the shared batch CLI options."""
    from bioamla.batch import BatchConfig

    return BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir or "",
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )


def _operation() -> str:
    """The running command's path without the program name (e.g. ``batch audio info``),
    used as the ``operation`` field on the batch engine's log records."""
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return "batch"
    names = []
    while ctx.parent is not None:
        names.append(ctx.info_name or "")
        ctx = ctx.parent
    return " ".join(reversed(names)) or "batch"


def _require_input_dir(config: BatchConfig) -> str:
    """Return the input directory, or raise if it was not provided."""
    if not config.input_dir:
        raise InvalidInputError("--input-dir is required for this command.")
    return config.input_dir


def _batch_json(result, output_dir: str | None, saved_hint: str | None) -> dict:
    """The ``--json`` result for a BatchResult.

    Counts, timing, output files and each failure's error; ``metadata_csv`` (CSV
    mode); ``output_dir``; and ``results_file``, the aggregate file the command
    wrote, only when it exists. Per-file rows, predictions and segments the
    library keeps in ``metadata`` are left out: they are in the results files.
    """
    if "output_file" in result.metadata:  # the library says what it wrote (None: nothing)
        results_file = result.metadata["output_file"]
    elif output_dir and saved_hint:
        results_file = str(Path(output_dir) / saved_hint)
    else:
        results_file = None
    if results_file and not Path(results_file).exists():
        results_file = None
    data = {
        name: getattr(result, name)
        for name in (
            "total_files",
            "successful",
            "failed",
            "skipped",
            "start_time",
            "end_time",
            "duration_seconds",
            "output_files",
            "errors",
        )
    }
    data["metadata_csv"] = result.metadata.get("metadata_csv")
    data["output_dir"] = output_dir
    data["results_file"] = results_file
    return data


def _report(
    result,
    output_dir: str | None,
    quiet: bool,
    saved_hint: str | None = None,
    *,
    failure: str | None = None,
) -> None:
    """Report a BatchResult: the summary for people, or the ``--json`` result.

    ``failure``: the run counts as failed (e.g. validation found bad files). In
    JSON mode that is a failure envelope with the result in ``error.details``;
    for people the summary prints as usual and the caller exits non-zero.
    """
    from bioamla.cli.console import print_success, print_warning
    from bioamla.cli.output import emit_result, fail, json_mode

    data = _batch_json(result, output_dir, saved_hint)
    if failure and json_mode():
        fail(failure, "InvalidInputError", details=data)
    if emit_result(data):
        return
    if quiet:
        return
    summary = (
        f"Processed {result.total_files} files: "
        f"{result.successful} successful, {result.failed} failed"
    )
    if result.failed > 0:
        print_warning(summary)
    else:
        print_success(summary)
    if data["results_file"]:
        print_success(f"Results saved to {data['results_file']}")


def _write_metadata_csv(result, context, quiet: bool) -> None:
    """Write the CSV-mode metadata CSV and record its path in the result."""
    from bioamla.batch import write_csv
    from bioamla.cli.console import print_success

    out_csv = write_csv(context)
    result.metadata["metadata_csv"] = str(out_csv)
    if not quiet:
        print_success(f"Updated metadata CSV written to: {out_csv}")


def _stty_sane() -> None:
    """Flush output and reset the terminal to a sane state (parallel cleanup)."""
    import subprocess
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    if sys.stdout.isatty():
        try:
            subprocess.run(["stty", "sane"], check=False, stderr=subprocess.DEVNULL, timeout=1)
        except Exception:
            pass


# =============================================================================
# CSV-metadata helpers for the commands not yet run through operations
# =============================================================================


def _csv_context(config: BatchConfig) -> CSVBatchContext:
    """Load the CSV metadata context for ``config.input_file``."""
    from bioamla.batch import load_csv

    output_dir = config.output_dir or None
    return load_csv(config.input_file, output_dir)


def _csv_written(result, quiet: bool) -> None:
    """Say where the CSV-mode metadata CSV went."""
    from bioamla.cli.console import print_success

    if not quiet and result.metadata.get("metadata_csv"):
        print_success(f"Updated metadata CSV written to: {result.metadata['metadata_csv']}")


def _run_csv_merge(
    config: BatchConfig,
    analyze_row: Callable[[Path], dict],
    *,
    context: CSVBatchContext | None = None,
    clear: tuple[str, ...] = (),
) -> object:
    """CSV mode for analysis commands: :func:`bioamla.batch.run_csv_merge`."""
    from bioamla.batch import run_csv_merge

    result = run_csv_merge(
        context or _csv_context(config),
        analyze_row,
        clear=clear,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        operation=_operation(),
    )
    _csv_written(result, config.quiet)
    return result


def _run_csv_process(config: BatchConfig, process_row: Callable[[Path], object]) -> object:
    """CSV mode for commands whose results go elsewhere: :func:`bioamla.batch.run_csv_process`."""
    from bioamla.batch import run_csv_process

    return run_csv_process(
        _csv_context(config),
        process_row,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        operation=_operation(),
    )


def _run_csv_predict_segments(
    config: BatchConfig, predict_row: Callable[[Path], list[dict]]
) -> object:
    """CSV mode for per-segment predictions: :func:`bioamla.batch.run_csv_predict_segments`."""
    from bioamla.batch import run_csv_predict_segments

    result = run_csv_predict_segments(
        _csv_context(config),
        predict_row,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        operation=_operation(),
    )
    _csv_written(result, config.quiet)
    return result


def _run_batch_operation(
    name: str,
    *,
    input_dir: str | None,
    input_file: str | None,
    output_dir: str | None,
    max_workers: int,
    recursive: bool,
    quiet: bool,
    header: str | None = None,
    progress: str | None = None,
    **params,
) -> object:
    """Run the batch operation ``name`` for a command: print ``header``, show a
    progress bar titled ``progress`` (directory mode), and report where a CSV-mode
    metadata CSV went."""
    from bioamla.cli.console import ProgressBar, print_header
    from bioamla.cli.operations import run_operation

    if header and not quiet:
        print_header(header)
    fields = {
        "input_dir": input_dir,
        "input_file": input_file,
        "output_dir": output_dir,
        "recursive": recursive,
        "max_workers": max_workers,
        **params,
    }
    if progress and input_dir:
        bar = ProgressBar(description=progress, disable=quiet)
        with bar:
            result = run_operation(name, on_progress=bar.on_progress, **fields)
    else:
        result = run_operation(name, **fields)
    _csv_written(result, quiet)
    return result


@click.group()
def batch() -> None:
    """Batch processing operations for multiple files."""
    pass


# =============================================================================
# Audio batch commands
# =============================================================================


@batch.group()
def audio() -> None:
    """Batch audio processing operations."""
    pass


@audio.command("info")
@batch_input_options
@batch_output_options
@max_workers_option
@recursive_option
@quiet_option
def audio_info(input_dir, input_file, output_dir, max_workers, recursive, quiet) -> None:
    """Extract audio file metadata (duration, sample rate, channels, format, etc.)."""
    try:
        result = _run_batch_operation(
            "batch.audio.info", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet,
            header="Extracting audio metadata...", progress="Reading metadata",
        )  # fmt: skip
        _report(result, output_dir, quiet, None if input_file else "audio_info.csv")
    finally:
        _stty_sane()


@audio.command("validate")
@batch_input_options
@batch_output_options
@max_workers_option
@recursive_option
@quiet_option
def audio_validate(input_dir, input_file, output_dir, max_workers, recursive, quiet) -> None:
    """Check that WAV files are well-formed PCM or float (header only; no FFmpeg, no decoding).

    Writes validation_report.csv (file_name, valid, error) when --output-dir is given.
    Non-WAV files are reported as skipped. Exits 1 if any file is invalid, so this
    works as a pre-flight step in scripts.
    """
    try:
        result = _run_batch_operation(
            "batch.audio.validate", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet, header="Validating WAV files..."
        )  # fmt: skip
        invalid = result.metadata.get("invalid_files", [])
        _report(
            result,
            output_dir,
            quiet,
            saved_hint=None if input_file else "validation_report.csv",
            failure=f"{len(invalid)} invalid file(s)" if invalid else None,
        )
        if invalid:
            raise click.exceptions.Exit(1)
    finally:
        _stty_sane()


@audio.command("metadata")
@batch_input_options
@batch_output_options
@max_workers_option
@recursive_option
@quiet_option
def audio_metadata(input_dir, input_file, output_dir, max_workers, recursive, quiet) -> None:
    """Extract recorder metadata (GUANO, AudioMoth, WAMD, bext) from WAV files.

    Writes recorder_metadata.csv with recorded_at, latitude, longitude, device and
    sensor columns when --output-dir is given; with --input-file the columns are
    merged into the manifest. Only WAV files carry this metadata; others are skipped.
    """
    try:
        result = _run_batch_operation(
            "batch.audio.metadata", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet, header="Extracting recorder metadata..."
        )  # fmt: skip
        _report(result, output_dir, quiet, None if input_file else "recorder_metadata.csv")
    finally:
        _stty_sane()


@audio.command("features")
@batch_input_options
@batch_output_options
@click.option(
    "--format",
    "-f",
    "out_format",
    default="csv",
    show_default=True,
    type=click.Choice(["csv", "parquet"]),
    help="Format of each per-frame table",
)
@frame_feature_options
@max_workers_option
@recursive_option
@quiet_option
def audio_features(
    input_dir,
    input_file,
    output_dir,
    out_format,
    trajectory,
    gate_db,
    no_denoise,
    n_fft,
    hop,
    n_mels,
    n_mfcc,
    f_min,
    f_max,
    max_workers,
    recursive,
    quiet,
) -> None:
    """Write a per-frame feature table (<name>.features.csv) for every file.

    Directory mode mirrors the input tree under --output-dir and writes an
    index, frame_features.csv. CSV mode writes each table under --output-dir
    (mirroring the CSV's folder) or next to its audio, and adds the
    features_file (relative to the CSV where possible) and n_frames columns.
    """
    from bioamla.cli.options import frame_feature_config

    try:
        analysis = frame_feature_config(no_denoise, n_fft, hop, n_mels, n_mfcc, f_min, f_max)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    try:
        result = _run_batch_operation(
            "batch.audio.features", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet,
            header="Computing frame features...", progress="Computing features",
            config=analysis, trajectory=trajectory, gate_db=gate_db, file_format=out_format,
        )  # fmt: skip
        _report(result, output_dir, quiet, None if input_file else "frame_features.csv")
    finally:
        _stty_sane()


@audio.command("convert")
@batch_input_options
@batch_output_options
@click.option("--sample-rate", "-r", default=None, type=int, help="Target sample rate")
@click.option("--channels", "-c", default=None, type=int, help="Target number of channels")
@click.option(
    "--format",
    "-f",
    "out_format",
    default="wav",
    type=click.Choice(["wav", "mp3", "flac", "ogg"]),
    help="Output format",
)
@click.option(
    "--delete-original",
    is_flag=True,
    default=False,
    help="Delete original files after successful conversion",
)
@max_workers_option
@recursive_option
@quiet_option
def audio_convert(
    input_dir,
    input_file,
    output_dir,
    sample_rate,
    channels,
    out_format,
    delete_original,
    max_workers,
    recursive,
    quiet,
) -> None:
    """Batch convert audio files to a target format (optionally resample/re-channel)."""
    try:
        result = _run_batch_operation(
            "batch.audio.convert", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet, header="Converting audio files...",
            target_format=out_format, sample_rate=sample_rate, channels=channels,
            delete_original=delete_original,
        )  # fmt: skip
        _report(result, output_dir, quiet)
    finally:
        _stty_sane()


@audio.command("resample")
@batch_input_options
@batch_output_options
@click.option("--sample-rate", "-r", required=True, type=int, help="Target sample rate in Hz")
@max_workers_option
@recursive_option
@quiet_option
def audio_resample(
    input_dir, input_file, output_dir, sample_rate, max_workers, recursive, quiet
) -> None:
    """Batch resample audio files to a target sample rate."""
    result = _run_batch_operation(
        "batch.audio.resample", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet, header="Resampling audio files...",
        sample_rate=sample_rate,
    )  # fmt: skip
    _report(result, output_dir, quiet)


@audio.command("normalize")
@batch_input_options
@batch_output_options
@click.option("--target-db", "-d", default=-20.0, type=float, help="Target loudness in dB")
@click.option("--peak", is_flag=True, help="Use peak normalization instead of RMS")
@max_workers_option
@recursive_option
@quiet_option
def audio_normalize(
    input_dir, input_file, output_dir, target_db, peak, max_workers, recursive, quiet
) -> None:
    """Batch normalize audio levels."""
    result = _run_batch_operation(
        "batch.audio.normalize", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet, header="Normalizing audio files...",
        target_db=target_db, peak=peak,
    )  # fmt: skip
    _report(result, output_dir, quiet)


@audio.command("trim")
@batch_input_options
@batch_output_options
@click.option("--start", "-s", default=None, type=float, help="Start time in seconds")
@click.option("--end", "-e", default=None, type=float, help="End time in seconds")
@click.option(
    "--trim-silence",
    "trim_silence_flag",
    is_flag=True,
    help="Trim silence from start/end instead of time range",
)
@click.option(
    "--silence-threshold-db",
    default=-40.0,
    type=float,
    help="Silence threshold in dB (with --trim-silence)",
)
@max_workers_option
@recursive_option
@quiet_option
def audio_trim(
    input_dir,
    input_file,
    output_dir,
    start,
    end,
    trim_silence_flag,
    silence_threshold_db,
    max_workers,
    recursive,
    quiet,
) -> None:
    """Batch trim audio files by time range or remove silence."""
    result = _run_batch_operation(
        "batch.audio.trim", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet, header="Trimming audio files...",
        start=start, end=end, trim_silence=trim_silence_flag,
        silence_threshold_db=silence_threshold_db,
    )  # fmt: skip
    _report(result, output_dir, quiet)


@audio.command("filter")
@batch_input_options
@batch_output_options
@click.option("--lowpass", default=None, type=float, help="Lowpass cutoff frequency in Hz")
@click.option("--highpass", default=None, type=float, help="Highpass cutoff frequency in Hz")
@click.option("--bandpass-low", default=None, type=float, help="Bandpass low frequency in Hz")
@click.option("--bandpass-high", default=None, type=float, help="Bandpass high frequency in Hz")
@click.option("--order", default=5, type=int, help="Filter order (default: 5)")
@max_workers_option
@recursive_option
@quiet_option
def audio_filter(
    input_dir,
    input_file,
    output_dir,
    lowpass,
    highpass,
    bandpass_low,
    bandpass_high,
    order,
    max_workers,
    recursive,
    quiet,
) -> None:
    """Batch apply frequency filters to audio files."""
    header = None  # no header for an invalid combination: the operation reports it
    if bandpass_low is not None and bandpass_high is not None:
        header = f"Applying bandpass filter ({bandpass_low}-{bandpass_high} Hz)..."
    elif bandpass_low is None and bandpass_high is None:
        if lowpass is not None:
            header = f"Applying lowpass filter ({lowpass} Hz)..."
        elif highpass is not None:
            header = f"Applying highpass filter ({highpass} Hz)..."
    result = _run_batch_operation(
        "batch.audio.filter", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet, header=header,
        lowpass=lowpass, highpass=highpass, bandpass_low=bandpass_low,
        bandpass_high=bandpass_high, order=order,
    )  # fmt: skip
    _report(result, output_dir, quiet)


@audio.command("denoise")
@batch_input_options
@batch_output_options
@click.option("--strength", default=1.0, type=float, help="Noise reduction strength (0-2)")
@max_workers_option
@recursive_option
@quiet_option
def audio_denoise(
    input_dir, input_file, output_dir, strength, max_workers, recursive, quiet
) -> None:
    """Batch apply spectral noise reduction to audio files."""
    result = _run_batch_operation(
        "batch.audio.denoise", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet,
        header=f"Applying spectral noise reduction (strength={strength})...", strength=strength,
    )  # fmt: skip
    _report(result, output_dir, quiet)


@audio.command("segment")
@batch_input_options
@batch_output_options
@click.option(
    "--duration-seconds",
    "-d",
    "duration",
    required=True,
    type=float,
    help="Segment duration in seconds",
)
@click.option(
    "--overlap", "-o", default=0.0, type=float, help="Overlap between segments in seconds"
)
@max_workers_option
@recursive_option
@quiet_option
def audio_segment(
    input_dir, input_file, output_dir, duration, overlap, max_workers, recursive, quiet
) -> None:
    """Batch segment audio files into fixed-duration chunks."""
    try:
        result = _run_batch_operation(
            "batch.audio.segment", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet, header="Segmenting audio files...",
            progress="Segmenting", duration=duration, overlap=overlap,
        )  # fmt: skip
        _report(result, output_dir, quiet)
    finally:
        _stty_sane()


@audio.command("visualize")
@batch_input_options
@batch_output_options
@click.option(
    "--plot-type",
    "-t",
    default="mel",
    type=click.Choice(["mel", "stft", "mfcc", "waveform"]),
    help="Visualization type",
)
@click.option("--legend/--no-legend", default=True, help="Show axes, title, and colorbar")
@max_workers_option
@recursive_option
@quiet_option
def audio_visualize(
    input_dir, input_file, output_dir, plot_type, legend, max_workers, recursive, quiet
) -> None:
    """Batch generate audio visualizations."""
    from bioamla.cli.console import print_warning
    from bioamla.cli.logging_setup import configure_cli_logging

    configure_cli_logging()
    if not legend:
        print_warning("--no-legend is not supported in batch mode yet and is ignored (see #246)")
    try:
        result = _run_batch_operation(
            "batch.audio.visualize", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet,
            header=f"Generating {plot_type} visualizations...",
            plot_type=plot_type, verbose=not quiet,
        )  # fmt: skip
        _report(result, output_dir, quiet)
    finally:
        _stty_sane()


# =============================================================================
# Detection batch commands
# =============================================================================


@batch.group()
def detect() -> None:
    """Batch detection operations."""
    pass


def _run_detect(config: BatchConfig, method: str, output_dir, quiet, recursive, **params) -> None:
    """Shared dispatch for detection commands (directory or CSV mode)."""
    from bioamla.detect import batch_detect_dir

    if config.input_file:
        from bioamla.batch import merge_analysis_results
        from bioamla.detect import batch_detect_files

        context = _csv_context(config)
        # The detections file goes beside the output CSV, as in directory mode.
        out_dir = Path(context.output_dir) if context.output_dir else context.csv_path.parent
        result = batch_detect_files(
            [row.file_path for row in context.rows],
            out_dir,
            method,
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            **params,
        )
        counts = result.metadata["file_counts"]
        for row in context.rows:
            # Blank for a file that failed, so no earlier run's count lingers.
            merge_analysis_results(row, {"num_detections": counts.get(str(row.file_path), "")})
        _write_metadata_csv(result, context, quiet)
        _report(result, config.output_dir or None, quiet, saved_hint=f"detections_{method}.json")
        return

    from bioamla.cli.console import ProgressBar

    in_dir = _require_input_dir(config)
    out_dir = output_dir or in_dir
    bar = ProgressBar(description=f"Detecting ({method})", disable=quiet)
    with bar:
        result = batch_detect_dir(
            in_dir,
            out_dir,
            method=method,
            recursive=recursive,
            max_workers=config.max_workers,
            on_progress=bar.on_progress,
            **params,
        )
    _report(result, out_dir, quiet, saved_hint=f"detections_{method}.json")


@detect.command("energy")
@batch_input_options
@batch_output_options
@click.option("--low-freq", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option("--threshold-db", default=-20.0, type=float, help="Detection threshold (dB)")
@click.option("--min-duration", default=0.05, type=float, help="Minimum detection duration (s)")
@max_workers_option
@recursive_option
@quiet_option
def detect_energy(
    input_dir,
    input_file,
    output_dir,
    low_freq,
    high_freq,
    threshold_db,
    min_duration,
    max_workers,
    recursive,
    quiet,
) -> None:
    """Batch detect sounds using band-limited energy detection."""
    from bioamla.cli.console import print_header

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            print_header("Detecting energy...")
        _run_detect(
            config,
            "energy",
            output_dir,
            quiet,
            recursive,
            low_freq=low_freq,
            high_freq=high_freq,
            threshold_db=threshold_db,
            min_duration=min_duration,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


@detect.command("ribbit")
@batch_input_options
@batch_output_options
@click.option("--pulse-rate", default=10.0, type=float, help="Expected pulse rate (Hz)")
@click.option("--pulse-tolerance", default=0.2, type=float, help="Pulse rate tolerance")
@click.option("--low-freq", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option(
    "--window-seconds",
    "window_duration",
    default=2.0,
    type=float,
    help="Analysis window in seconds",
)
@click.option("--min-score", default=0.3, type=float, help="Minimum detection score")
@max_workers_option
@recursive_option
@quiet_option
def detect_ribbit(
    input_dir,
    input_file,
    output_dir,
    pulse_rate,
    pulse_tolerance,
    low_freq,
    high_freq,
    window_duration,
    min_score,
    max_workers,
    recursive,
    quiet,
) -> None:
    """Batch detect periodic calls using RIBBIT algorithm."""
    from bioamla.cli.console import print_header

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            print_header("Detecting RIBBIT calls...")
        _run_detect(
            config,
            "ribbit",
            output_dir,
            quiet,
            recursive,
            pulse_rate_hz=pulse_rate,
            pulse_rate_tolerance=pulse_tolerance,
            low_freq=low_freq,
            high_freq=high_freq,
            window_duration=window_duration,
            min_score=min_score,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


@detect.command("peaks")
@batch_input_options
@batch_output_options
@click.option("--snr-threshold", default=2.0, type=float, help="Signal-to-noise ratio threshold")
@click.option("--min-peak-distance", default=0.01, type=float, help="Minimum peak distance (s)")
@click.option("--low-freq", default=None, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=None, type=float, help="High frequency bound (Hz)")
@max_workers_option
@recursive_option
@quiet_option
def detect_peaks(
    input_dir,
    input_file,
    output_dir,
    snr_threshold,
    min_peak_distance,
    low_freq,
    high_freq,
    max_workers,
    recursive,
    quiet,
) -> None:
    """Batch detect peaks using Continuous Wavelet Transform."""
    from bioamla.cli.console import print_header

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            print_header("Detecting peaks...")
        _run_detect(
            config,
            "peaks",
            output_dir,
            quiet,
            recursive,
            snr_threshold=snr_threshold,
            min_peak_distance=min_peak_distance,
            low_freq=low_freq,
            high_freq=high_freq,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


@detect.command("accelerating")
@batch_input_options
@batch_output_options
@click.option("--min-pulses", default=5, type=int, help="Minimum pulses to detect pattern")
@click.option("--accel-threshold", default=1.5, type=float, help="Acceleration threshold")
@click.option("--decel-threshold", default=None, type=float, help="Deceleration threshold")
@click.option("--low-freq", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option(
    "--window-seconds",
    "window_duration",
    default=3.0,
    type=float,
    help="Analysis window in seconds",
)
@max_workers_option
@recursive_option
@quiet_option
def detect_accelerating(
    input_dir,
    input_file,
    output_dir,
    min_pulses,
    accel_threshold,
    decel_threshold,
    low_freq,
    high_freq,
    window_duration,
    max_workers,
    recursive,
    quiet,
) -> None:
    """Batch detect accelerating or decelerating call patterns."""
    from bioamla.cli.console import print_header

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            print_header("Detecting accelerating patterns...")
        _run_detect(
            config,
            "accelerating",
            output_dir,
            quiet,
            recursive,
            min_pulses=min_pulses,
            acceleration_threshold=accel_threshold,
            deceleration_threshold=decel_threshold,
            low_freq=low_freq,
            high_freq=high_freq,
            window_duration=window_duration,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


# =============================================================================
# Indices batch commands
# =============================================================================


@batch.command("index")
@batch_input_options
@batch_output_options
@click.option(
    "--indices",
    default="aci,adi,aei,bio,ndsi",
    help="Comma-separated indices (add h_spectral,h_temporal for entropy)",
)
@max_workers_option
@recursive_option
@quiet_option
def batch_index(input_dir, input_file, output_dir, indices, max_workers, recursive, quiet) -> None:
    """Batch calculate acoustic indices for audio files."""
    from bioamla.cli.console import ProgressBar, print_header
    from bioamla.cli.logging_setup import configure_cli_logging
    from bioamla.indices import batch_index_files, check_index_names
    from bioamla.indices.compute import _indices_row

    configure_cli_logging()
    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        selected = [name.strip() for name in indices.split(",") if name.strip()]
        check_index_names(selected)
        if not quiet:
            print_header("Calculating acoustic indices...")

        if config.input_file:

            def _analyze(path: Path) -> dict:
                result = _indices_row(path, selected)
                # Drop the bookkeeping columns before merging into the CSV row.
                for key in ("filepath", "success"):
                    result.pop(key, None)
                return result

            result = _run_csv_merge(config, _analyze)
            _report(result, config.output_dir or None, quiet)
            return

        bar = ProgressBar(description="Calculating indices", disable=quiet)
        with bar:
            result = batch_index_files(
                _require_input_dir(config),
                output_dir or None,
                indices=selected,
                recursive=recursive,
                max_workers=config.max_workers,
                on_progress=bar.on_progress,
            )
        _report(result, output_dir, quiet, "indices.csv")
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    finally:
        _stty_sane()


# =============================================================================
# Model inference batch commands
# =============================================================================


@batch.group()
def models() -> None:
    """Batch model operations (AST-only)."""
    pass


@models.command("predict")
@batch_input_options
@batch_output_options
@click.option(
    "--model",
    "--model-path",
    "-m",
    "model",
    required=True,
    help="Model path (HuggingFace ID or local path)",
)
@click.option("--top-k", default=5, type=int, help="Number of top predictions to return")
@click.option("--min-confidence", default=0.0, type=float, help="Minimum confidence threshold")
@click.option(
    "--segment-seconds",
    "segment_duration",
    default=0,
    type=int,
    help="Split each file into N-second segments and classify each (0 = whole file)",
)
@click.option("--overlap", default=0, type=int, help="Overlap between segments (seconds)")
@max_workers_option
@recursive_option
@quiet_option
def models_predict(
    input_dir,
    input_file,
    output_dir,
    model,
    top_k,
    min_confidence,
    segment_duration,
    overlap,
    max_workers,
    recursive,
    quiet,
) -> None:
    """Batch run AST model predictions on audio files (whole file or in segments).

    With ``--segment-duration`` each file is split into fixed-length (optionally
    overlapping) segments and classified per segment: directory mode writes a flat
    ``predictions.csv`` (one row per segment) to ``--output-dir``; CSV-metadata mode
    expands each input row into one row per segment.
    """
    import csv as _csv
    import json

    from bioamla.cli.console import print_header
    from bioamla.ml import batch_predict_files, batch_predict_segments

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            print_header(f"Running AST predictions with model {model}...")

        if segment_duration > 0:
            if config.input_file:
                from bioamla.ml.inference import ASTInference

                inference = ASTInference(model_path=model)

                def _predict_segments(path: Path) -> list[dict]:
                    results = inference.predict_segments(
                        str(path), clip_length=segment_duration, overlap=overlap
                    )
                    return [
                        {
                            "start_time": r.start_time,
                            "end_time": r.end_time,
                            "prediction": r.predicted_label,
                            "confidence": r.confidence,
                        }
                        for r in results
                        if r.confidence >= min_confidence
                    ]

                result = _run_csv_predict_segments(config, _predict_segments)
                _report(result, output_dir, quiet)
                return

            in_dir = _require_input_dir(config)
            result = batch_predict_segments(
                in_dir,
                model_path=model,
                segment_duration=segment_duration,
                overlap=overlap,
                min_confidence=min_confidence,
                recursive=recursive,
                max_workers=max_workers,
            )
            segments = result.metadata.get("segments", [])
            if output_dir and segments:
                out_path = Path(output_dir) / "predictions.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", newline="", encoding="utf-8") as f:
                    writer = _csv.DictWriter(
                        f,
                        fieldnames=[
                            "filepath",
                            "start_time",
                            "end_time",
                            "predicted_label",
                            "confidence",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(segments)
            _report(result, output_dir, quiet, saved_hint="predictions.csv")
            return

        if config.input_file:
            from bioamla.ml.inference import ASTInference

            inference = ASTInference(model_path=model)
            predictions: list = []

            def _process(path: Path) -> None:
                pred = inference.predict_topk(str(path), top_k=top_k, min_confidence=min_confidence)
                predictions.append(
                    {
                        "filepath": str(path),
                        "predicted_label": pred.predicted_label,
                        "confidence": pred.confidence,
                        "top_k_labels": pred.top_k_labels,
                        "top_k_scores": pred.top_k_scores,
                    }
                )

            result = _run_csv_process(config, _process)
            if output_dir and predictions:
                out_path = Path(output_dir) / "predictions.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
            _report(result, output_dir, quiet, saved_hint="predictions.json")
            return

        in_dir = _require_input_dir(config)
        result = batch_predict_files(
            in_dir,
            model_path=model,
            top_k=top_k,
            min_confidence=min_confidence,
            recursive=recursive,
            max_workers=max_workers,
        )

        predictions = result.metadata.get("predictions", [])
        if output_dir and predictions:
            out_path = Path(output_dir) / "predictions.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

        _report(result, output_dir, quiet, saved_hint="predictions.json")
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


@models.command("embed")
@batch_input_options
@batch_output_options
@click.option(
    "--model",
    "--model-path",
    "-m",
    "model",
    required=True,
    help="Model path (HuggingFace ID or local path)",
)
@max_workers_option
@recursive_option
@quiet_option
def models_embed(input_dir, input_file, output_dir, model, max_workers, recursive, quiet) -> None:
    """Batch extract embeddings from audio files."""
    from bioamla.cli.console import print_header
    from bioamla.ml import batch_embed_files

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not output_dir:
            raise InvalidInputError("--output-dir is required for this command.")
        if not quiet:
            print_header(f"Extracting AST embeddings with model {model}...")

        if config.input_file:
            import numpy as _np

            from bioamla.batch import output_stems
            from bioamla.ml.embedding import EmbeddingConfig, EmbeddingExtractor

            out_dir = Path(output_dir)
            context = _csv_context(config)  # read (and check) the CSV before loading the model
            extractor = EmbeddingExtractor(config=EmbeddingConfig(model_path=model))
            listed = [str(row.file_path) for row in context.rows]
            # Mirror the CSV's folder under out_dir; inputs that share a name get
            # distinct files (a.wav + a.flac, or rec.wav in two folders).
            stems = output_stems(listed, context.csv_dir)

            def _analyze(path: Path) -> dict:
                stem = stems[str(path)]
                target = out_dir / stem.parent / f"{stem.name}_embeddings.npy"
                target.parent.mkdir(parents=True, exist_ok=True)
                emb = extractor.extract(str(path))
                _np.save(str(target), emb.embeddings)
                # A Path: written relative to wherever the metadata CSV is saved.
                return {"embedding_file": target.resolve()}

            result = _run_csv_merge(config, _analyze, context=context, clear=("embedding_file",))
            _report(result, output_dir, quiet)
            return

        _require_input_dir(config)
        result = batch_embed_files(
            input_dir,
            output_dir,
            model_path=model,
            recursive=recursive,
            max_workers=max_workers,
        )
        _report(result, output_dir, quiet)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


# =============================================================================
# Clustering batch command
# =============================================================================


@batch.command("cluster")
@batch_input_options
@batch_output_options
@click.option(
    "--method",
    default="hdbscan",
    type=click.Choice(["hdbscan", "kmeans", "dbscan", "agglomerative"]),
    help="Clustering method",
)
@click.option(
    "--n-clusters", default=None, type=int, help="Number of clusters (k-means/agglomerative)"
)
@click.option("--min-cluster-size", default=5, type=int, help="Minimum cluster size (HDBSCAN)")
@click.option("--min-samples", default=3, type=int, help="Minimum samples per cluster")
@max_workers_option
@recursive_option
@quiet_option
def cluster_batch(
    input_dir,
    input_file,
    output_dir,
    method,
    n_clusters,
    min_cluster_size,
    min_samples,
    max_workers,
    recursive,
    quiet,
) -> None:
    """Batch cluster embedding files from a directory or a metadata CSV."""
    from bioamla.cli.console import ProgressBar, print_header
    from bioamla.cluster import cluster_batch_files

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not output_dir:
            raise InvalidInputError("--output-dir is required for this command.")
        if not quiet:
            print_header(f"Clustering embeddings using {method}...")

        if config.input_file:
            from bioamla.batch import merge_analysis_results, resolve_file_path
            from bioamla.cluster.batch import cluster_embedding_files

            context = _csv_context(config)
            # A CSV from `batch models embed` names each row's embedding; a blank
            # one (embedding failed) is skipped. Without that column, file_name
            # is the embedding.
            if "embedding_file" in context.fieldnames:
                row_paths = [
                    resolve_file_path(e, context.csv_dir)
                    if (e := row.metadata_fields.get("embedding_file"))
                    else None
                    for row in context.rows
                ]
            else:
                row_paths = [row.file_path for row in context.rows]
            embedding_paths = [str(p) for p in row_paths if p is not None]
            if not embedding_paths:
                raise InvalidInputError(f"No row in {config.input_file} has an embedding_file")
            bar = ProgressBar(description="Loading embeddings", disable=quiet)
            with bar:
                result = cluster_embedding_files(
                    embedding_paths,
                    output_dir,
                    method=method,
                    n_clusters=n_clusters,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    on_progress=bar.on_progress,
                )
            by_file = {a["filepath"]: a for a in result.metadata.get("file_assignments", [])}
            for row, path in zip(context.rows, row_paths, strict=True):
                # Blank for a file that wasn't clustered, so no earlier run's value lingers.
                assignment = by_file.get(str(path)) if path is not None else None
                merge_analysis_results(
                    row,
                    {
                        "cluster": assignment["cluster"] if assignment else "",
                        "cluster_labels": ";".join(str(x) for x in assignment["cluster_labels"])
                        if assignment
                        else "",
                    },
                )
            _write_metadata_csv(result, context, quiet)
            _report(result, output_dir, quiet, saved_hint="cluster_assignments.json")
            return

        bar = ProgressBar(description="Loading embeddings", disable=quiet)
        with bar:
            result = cluster_batch_files(
                input_dir,
                output_dir,
                method=method,
                n_clusters=n_clusters,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                recursive=recursive,
                on_progress=bar.on_progress,
            )
        _report(result, output_dir, quiet, saved_hint="cluster_assignments.json")
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

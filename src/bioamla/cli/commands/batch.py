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

Each command runs its ``batch.*`` operation (see :mod:`bioamla.operations`), which
calls the library's batch function for its domain; the command prints a header,
shows progress and reports the :class:`~bioamla.batch.BatchResult`.
"""

from __future__ import annotations

from pathlib import Path

import click

from bioamla.cli.batch_options import batch_input_options, batch_output_options
from bioamla.cli.options import (
    frame_feature_options,
    max_workers_option,
    quiet_option,
    recursive_option,
)
from bioamla.exceptions import BioamlaError

# =============================================================================
# Shared helpers
# =============================================================================


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
# Running a batch operation
# =============================================================================


def _csv_written(result, quiet: bool) -> None:
    """Say where the CSV-mode metadata CSV went."""
    from bioamla.cli.console import print_success

    if not quiet and result.metadata.get("metadata_csv"):
        print_success(f"Updated metadata CSV written to: {result.metadata['metadata_csv']}")


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
    progress bar titled ``progress``, and report where a CSV-mode
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
    if progress:
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
    result = _run_batch_operation(
        "batch.detect.energy", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet,
        header="Detecting energy...", progress="Detecting (energy)",
        low_freq=low_freq, high_freq=high_freq, threshold_db=threshold_db, min_duration=min_duration,
    )  # fmt: skip
    _report(result, output_dir or input_dir, quiet)


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
    result = _run_batch_operation(
        "batch.detect.ribbit", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet,
        header="Detecting RIBBIT calls...", progress="Detecting (ribbit)",
        pulse_rate=pulse_rate, pulse_tolerance=pulse_tolerance, low_freq=low_freq, high_freq=high_freq, window_duration=window_duration, min_score=min_score,
    )  # fmt: skip
    _report(result, output_dir or input_dir, quiet)


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
    result = _run_batch_operation(
        "batch.detect.peaks", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet,
        header="Detecting peaks...", progress="Detecting (peaks)",
        snr_threshold=snr_threshold, min_peak_distance=min_peak_distance, low_freq=low_freq, high_freq=high_freq,
    )  # fmt: skip
    _report(result, output_dir or input_dir, quiet)


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
    result = _run_batch_operation(
        "batch.detect.accelerating", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet,
        header="Detecting accelerating patterns...", progress="Detecting (accelerating)",
        min_pulses=min_pulses, accel_threshold=accel_threshold, decel_threshold=decel_threshold, low_freq=low_freq, high_freq=high_freq, window_duration=window_duration,
    )  # fmt: skip
    _report(result, output_dir or input_dir, quiet)


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
    from bioamla.cli.logging_setup import configure_cli_logging

    configure_cli_logging()
    selected = [name.strip() for name in indices.split(",") if name.strip()]
    try:
        result = _run_batch_operation(
            "batch.index", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet,
            header="Calculating acoustic indices...", progress="Calculating indices",
            indices=selected,
        )  # fmt: skip
        _report(result, output_dir, quiet, None if input_file else "indices.csv")
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

    With ``--segment-seconds`` each file is split into fixed-length (optionally
    overlapping) segments and classified per segment: directory mode writes a flat
    ``predictions.csv`` (one row per segment) to ``--output-dir``; CSV-metadata mode
    expands each input row into one row per segment.
    """
    result = _run_batch_operation(
        "batch.models.predict", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet,
        header=f"Running AST predictions with model {model}...",
        model=model, top_k=top_k, min_confidence=min_confidence,
        segment_duration=segment_duration, overlap=overlap,
    )  # fmt: skip
    _report(result, output_dir, quiet)


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
    result = _run_batch_operation(
        "batch.models.embed", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet,
        header=f"Extracting AST embeddings with model {model}...",
        model=model,
    )  # fmt: skip
    _report(result, output_dir, quiet)


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
    result = _run_batch_operation(
        "batch.cluster", input_dir=input_dir, input_file=input_file, output_dir=output_dir, max_workers=max_workers, recursive=recursive, quiet=quiet,
        header=f"Clustering embeddings using {method}...",
        progress="Loading embeddings",
        method=method, n_clusters=n_clusters, min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )  # fmt: skip
    _report(result, output_dir, quiet)

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

import functools
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np

from bioamla.batch import (
    BatchConfig,
    CSVBatchContext,
    expand_row_for_segments,
    load_csv,
    merge_analysis_results,
    run_csv_batch,
    update_row_path,
    write_csv,
)
from bioamla.cli.batch_options import batch_input_options, batch_output_options
from bioamla.exceptions import BioamlaError, InvalidInputError

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
    return BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir or "",
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )


def _require_input_dir(config: BatchConfig) -> str:
    """Return the input directory, or raise if it was not provided."""
    if not config.input_dir:
        raise InvalidInputError("--input-dir is required for this command.")
    return config.input_dir


def _report(result, output_dir: str | None, quiet: bool, saved_hint: str | None = None) -> None:
    """Print a standard BatchResult summary."""
    if quiet:
        return
    click.echo(
        f"Processed {result.total_files} files: "
        f"{result.successful} successful, {result.failed} failed"
    )
    if output_dir and saved_hint:
        click.echo(f"Results saved to {output_dir}/{saved_hint}")
    for error in result.errors:
        click.echo(f"  Error: {error}")


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
# CSV-metadata dispatch helpers (shared across commands)
# =============================================================================


def _csv_context(config: BatchConfig) -> CSVBatchContext:
    """Load the CSV metadata context for ``config.input_file``."""
    output_dir = config.output_dir or None
    return load_csv(config.input_file, output_dir)


def _run_csv_merge(
    config: BatchConfig,
    analyze_row: Callable[[Path], dict],
) -> object:
    """CSV mode for analysis commands (audio info, indices).

    Runs ``analyze_row`` over each listed file, merges the returned column dict
    into the row, and writes the updated CSV (column-union, file_name first).
    """
    context = _csv_context(config)

    def _process(row) -> str:
        results = analyze_row(row.file_path)
        merge_analysis_results(row, results)
        return str(row.file_path)

    result = run_csv_batch(
        context,
        _process,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        quiet=config.quiet,
    )
    out_csv = write_csv(context)
    if not config.quiet:
        click.echo(f"Updated metadata CSV written to: {out_csv}")
    return result


def _run_csv_transform(
    config: BatchConfig,
    transform_row: Callable[[Path, Path], Path],
    *,
    new_extension: str | None = None,
) -> object:
    """CSV mode for file-changing commands (resample/normalize/trim/filter/
    denoise/convert).

    Computes the output path via the CSV context (in-place vs output_dir),
    invokes ``transform_row(input_path, output_path)``, and updates the row's
    ``file_name`` to the new path before writing the CSV.
    """
    from bioamla.batch import resolve_output_path

    context = _csv_context(config)

    def _process(row) -> str:
        out_path = resolve_output_path(row.file_path, context, new_extension=new_extension)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_path = transform_row(row.file_path, out_path)
        update_row_path(row, Path(final_path), context)
        return str(final_path)

    result = run_csv_batch(
        context,
        _process,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        quiet=config.quiet,
    )
    out_csv = write_csv(context)
    if not config.quiet:
        click.echo(f"Updated metadata CSV written to: {out_csv}")
    return result


def _run_csv_process(
    config: BatchConfig,
    process_row: Callable[[Path], object],
) -> object:
    """CSV mode for commands that process files but do not change the CSV rows
    (detect, models predict/embed).

    Runs ``process_row`` per file; the CSV is written back unchanged (rows
    preserved, paths re-resolved relative to the output CSV location).
    """
    context = _csv_context(config)

    def _process(row) -> str:
        process_row(row.file_path)
        return str(row.file_path)

    result = run_csv_batch(
        context,
        _process,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        quiet=config.quiet,
    )
    write_csv(context)
    return result


def _run_csv_segment(
    config: BatchConfig,
    segment_row: Callable[[Path, Path], list],
) -> object:
    """CSV mode for segmentation: expands one input row into many output rows.

    ``segment_row(input_path, output_dir)`` returns a list of
    :class:`bioamla.batch.SegmentInfo`. Each parent row is replaced by one row
    per segment (carrying ``parent_file``/``segment_id``/``start_time``/
    ``end_time``/``duration``), then the expanded CSV is written.
    """
    from bioamla.batch import resolve_output_path

    context = _csv_context(config)
    segment_mapping: dict = {}

    def _process(row) -> str:
        # Segments are written next to the resolved output location for the file.
        out_path = resolve_output_path(row.file_path, context)
        seg_dir = out_path.parent
        seg_dir.mkdir(parents=True, exist_ok=True)
        segments = segment_row(row.file_path, seg_dir)
        if segments:
            segment_mapping[row.file_path] = segments
        return str(row.file_path)

    result = run_csv_batch(
        context,
        _process,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        quiet=config.quiet,
    )

    # Expand rows BEFORE writing the CSV.
    new_rows = []
    for row in context.rows:
        if row.file_path in segment_mapping:
            new_rows.extend(expand_row_for_segments(row, segment_mapping[row.file_path], context))
        else:
            new_rows.append(row)
    context.rows = new_rows

    for field in ("parent_file", "segment_id", "start_time", "end_time", "duration"):
        if field not in context.fieldnames:
            context.fieldnames.append(field)

    out_csv = write_csv(context)
    if not config.quiet:
        click.echo(f"Updated metadata CSV written to: {out_csv}")
    return result


def _run_csv_predict_segments(
    config: BatchConfig,
    predict_row: Callable[[Path], list[dict]],
) -> object:
    """CSV mode for segmented prediction: expand one input row into many.

    ``predict_row(input_path)`` returns one dict per segment (keys
    ``start_time``/``end_time``/``prediction``/``confidence``). Each parent row is
    replaced by one row per segment — keeping the source ``file_name`` and adding
    ``segment_id``/``start_time``/``end_time``/``prediction``/``confidence`` — then
    the expanded CSV is written. Unlike :func:`_run_csv_segment`, no segment files
    are written; rows describe sub-spans of the original file.
    """
    from bioamla.batch import MetadataRow

    context = _csv_context(config)
    segment_mapping: dict = {}

    def _process(row) -> str:
        segments = predict_row(row.file_path)
        if segments:
            segment_mapping[row.file_path] = segments
        return str(row.file_path)

    result = run_csv_batch(
        context,
        _process,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        quiet=config.quiet,
    )

    new_rows = []
    for row in context.rows:
        segments = segment_mapping.get(row.file_path)
        if not segments:
            new_rows.append(row)
            continue
        for seg_id, seg in enumerate(segments):
            fields = row.metadata_fields.copy()
            fields.update(
                {
                    "segment_id": seg_id,
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "prediction": seg["prediction"],
                    "confidence": seg["confidence"],
                }
            )
            new_rows.append(
                MetadataRow(
                    file_name=row.file_name,
                    file_path=row.file_path,
                    metadata_fields=fields,
                    output_path=row.output_path,
                )
            )
    context.rows = new_rows

    for fld in ("segment_id", "start_time", "end_time", "prediction", "confidence"):
        if fld not in context.fieldnames:
            context.fieldnames.append(fld)

    out_csv = write_csv(context)
    if not config.quiet:
        click.echo(f"Updated metadata CSV written to: {out_csv}")
    return result


# =============================================================================
# Module-level audio processors (picklable for parallel execution)
# =============================================================================


def _proc_normalize_rms(audio: np.ndarray, sr: int, *, target_db: float) -> np.ndarray:
    from bioamla.audio.processing import normalize_loudness

    return normalize_loudness(audio, sr, target_db=target_db)


def _proc_normalize_peak(audio: np.ndarray, sr: int) -> np.ndarray:
    from bioamla.audio.processing import peak_normalize

    return peak_normalize(audio)


def _proc_trim_range(
    audio: np.ndarray, sr: int, *, start: float | None, end: float | None
) -> np.ndarray:
    from bioamla.audio.processing import trim_audio

    return trim_audio(audio, sr, start_time=start, end_time=end)


def _proc_trim_silence(audio: np.ndarray, sr: int, *, threshold_db: float) -> np.ndarray:
    from bioamla.audio.processing import trim_silence

    return trim_silence(audio, sr, threshold_db=threshold_db)


def _proc_denoise(audio: np.ndarray, sr: int, *, strength: float) -> np.ndarray:
    from bioamla.audio.processing import spectral_denoise

    return spectral_denoise(audio, sr, noise_reduce_factor=strength)


def _proc_filter(
    audio: np.ndarray,
    sr: int,
    *,
    lowpass: float | None,
    highpass: float | None,
    bandpass: tuple | None,
    order: int,
) -> np.ndarray:
    from bioamla.audio.processing import bandpass_filter, highpass_filter, lowpass_filter

    if bandpass is not None:
        return bandpass_filter(audio, sr, bandpass[0], bandpass[1], order=order)
    if lowpass is not None:
        return lowpass_filter(audio, sr, lowpass, order=order)
    return highpass_filter(audio, sr, highpass, order=order)


def _passthrough(audio: np.ndarray, sr: int) -> np.ndarray:
    """Identity processor used by resample."""
    return audio


def _run_audio_transform(config: BatchConfig, processor, *, sample_rate=None) -> object:
    """Run an audio per-file transform in directory OR CSV mode."""
    from bioamla.audio.batch import batch_transform_files
    from bioamla.audio.io import process_file

    if config.input_file:

        def _transform_row(in_path: Path, out_path: Path) -> Path:
            # Per-file transforms emit .wav; the CSV output path keeps the
            # original name unless an extension change is requested elsewhere.
            wav_out = out_path.with_suffix(".wav")
            saved = process_file(str(in_path), str(wav_out), processor, sample_rate)
            return Path(saved)

        return _run_csv_transform(config, _transform_row, new_extension=".wav")

    input_dir = _require_input_dir(config)
    if not config.output_dir:
        raise InvalidInputError("--output-dir is required for this command.")
    return batch_transform_files(
        input_dir,
        config.output_dir,
        processor_fn=processor,
        sample_rate=sample_rate,
        recursive=config.recursive,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
    )


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
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_info(input_dir, input_file, output_dir, max_workers, recursive, quiet) -> None:
    """Extract audio file metadata (duration, sample rate, channels, format, etc.)."""
    import csv

    from bioamla.audio.discovery import list_audio_files
    from bioamla.audio.info import get_audio_info
    from bioamla.batch import run_batch

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            click.echo("Extracting audio metadata...")

        def _analyze(path: Path) -> dict:
            info = get_audio_info(str(path))
            return {
                "duration": info.duration,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "samples": info.samples,
                "format": info.format,
            }

        if config.input_file:
            result = _run_csv_merge(config, _analyze)
            _report(result, None, quiet)
            return

        files = list_audio_files(_require_input_dir(config), recursive=recursive)
        rows: list = []

        def _info(path):
            data = {"file_name": str(path)}
            data.update(_analyze(path))
            rows.append(data)
            return str(path)

        result = run_batch(files, _info, continue_on_error=True)

        if output_dir and rows:
            out_path = Path(output_dir) / "audio_info.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        _report(result, output_dir, quiet, saved_hint="audio_info.csv")
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
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
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
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
    from bioamla.audio.batch import batch_convert_files
    from bioamla.audio.convert import convert_audio_file

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            click.echo("Converting audio files...")

        if config.input_file:

            def _convert_row(in_path: Path, out_path: Path) -> Path:
                saved = convert_audio_file(
                    in_path,
                    out_path,
                    target_format=out_format,
                    target_sample_rate=sample_rate,
                    target_channels=channels,
                    delete_original=delete_original,
                )
                return Path(saved)

            result = _run_csv_transform(config, _convert_row, new_extension=f".{out_format}")
            _report(result, None, quiet)
            return

        _require_input_dir(config)
        if not output_dir:
            raise InvalidInputError("--output-dir is required for this command.")

        result = batch_convert_files(
            input_dir,
            output_dir,
            target_format=out_format,
            sample_rate=sample_rate,
            channels=channels,
            delete_original=delete_original,
            recursive=recursive,
            max_workers=max_workers,
            continue_on_error=config.continue_on_error,
        )
        _report(result, output_dir, quiet)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    finally:
        _stty_sane()


@audio.command("resample")
@batch_input_options
@batch_output_options
@click.option("--sample-rate", "-r", required=True, type=int, help="Target sample rate in Hz")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_resample(
    input_dir, input_file, output_dir, sample_rate, max_workers, recursive, quiet
) -> None:
    """Batch resample audio files to a target sample rate."""
    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            click.echo("Resampling audio files...")
        result = _run_audio_transform(config, _passthrough, sample_rate=sample_rate)
        _report(result, output_dir if not config.input_file else None, quiet)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


@audio.command("normalize")
@batch_input_options
@batch_output_options
@click.option("--target-db", "-d", default=-20.0, type=float, help="Target loudness in dB")
@click.option("--peak", is_flag=True, help="Use peak normalization instead of RMS")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_normalize(
    input_dir, input_file, output_dir, target_db, peak, max_workers, recursive, quiet
) -> None:
    """Batch normalize audio levels."""
    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            click.echo("Normalizing audio files...")
        processor = (
            _proc_normalize_peak
            if peak
            else functools.partial(_proc_normalize_rms, target_db=target_db)
        )
        result = _run_audio_transform(config, processor)
        _report(result, output_dir if not config.input_file else None, quiet)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


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
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
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
    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            click.echo("Trimming audio files...")
        if trim_silence_flag:
            processor = functools.partial(_proc_trim_silence, threshold_db=silence_threshold_db)
        else:
            processor = functools.partial(_proc_trim_range, start=start, end=end)
        result = _run_audio_transform(config, processor)
        _report(result, output_dir if not config.input_file else None, quiet)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


@audio.command("filter")
@batch_input_options
@batch_output_options
@click.option("--lowpass", default=None, type=float, help="Lowpass cutoff frequency in Hz")
@click.option("--highpass", default=None, type=float, help="Highpass cutoff frequency in Hz")
@click.option("--bandpass-low", default=None, type=float, help="Bandpass low frequency in Hz")
@click.option("--bandpass-high", default=None, type=float, help="Bandpass high frequency in Hz")
@click.option("--order", default=5, type=int, help="Filter order (default: 5)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
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
    try:
        if not any([lowpass, highpass, bandpass_low]):
            raise InvalidInputError(
                "Must specify --lowpass, --highpass, or --bandpass-low/--bandpass-high"
            )
        if (bandpass_low is not None) != (bandpass_high is not None):
            raise InvalidInputError(
                "Both --bandpass-low and --bandpass-high must be specified together"
            )

        bandpass = (bandpass_low, bandpass_high) if bandpass_low is not None else None
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            if bandpass:
                click.echo(f"Applying bandpass filter ({bandpass[0]}-{bandpass[1]} Hz)...")
            elif lowpass:
                click.echo(f"Applying lowpass filter ({lowpass} Hz)...")
            else:
                click.echo(f"Applying highpass filter ({highpass} Hz)...")

        processor = functools.partial(
            _proc_filter, lowpass=lowpass, highpass=highpass, bandpass=bandpass, order=order
        )
        result = _run_audio_transform(config, processor)
        _report(result, output_dir if not config.input_file else None, quiet)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


@audio.command("denoise")
@batch_input_options
@batch_output_options
@click.option("--strength", default=1.0, type=float, help="Noise reduction strength (0-2)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_denoise(
    input_dir, input_file, output_dir, strength, max_workers, recursive, quiet
) -> None:
    """Batch apply spectral noise reduction to audio files."""
    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            click.echo(f"Applying spectral noise reduction (strength={strength})...")
        processor = functools.partial(_proc_denoise, strength=strength)
        result = _run_audio_transform(config, processor)
        _report(result, output_dir if not config.input_file else None, quiet)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


@audio.command("segment")
@batch_input_options
@batch_output_options
@click.option("--duration", "-d", required=True, type=float, help="Segment duration in seconds")
@click.option(
    "--overlap", "-o", default=0.0, type=float, help="Overlap between segments in seconds"
)
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_segment(
    input_dir, input_file, output_dir, duration, overlap, max_workers, recursive, quiet
) -> None:
    """Batch segment audio files into fixed-duration chunks."""
    from bioamla.audio.batch import segment_audio_file
    from bioamla.audio.discovery import list_audio_files
    from bioamla.batch import run_batch

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if duration <= 0:
            raise InvalidInputError("--duration must be positive.")
        if not quiet:
            click.echo("Segmenting audio files...")

        if config.input_file:

            def _segment_row(in_path: Path, seg_dir: Path) -> list:
                return segment_audio_file(
                    str(in_path), str(seg_dir), duration=duration, overlap=overlap
                )

            result = _run_csv_segment(config, _segment_row)
            _report(result, None, quiet)
            return

        in_dir = _require_input_dir(config)
        if not output_dir:
            raise InvalidInputError("--output-dir is required for this command.")

        in_path = Path(in_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        files = [Path(f) for f in list_audio_files(in_dir, recursive=recursive)]

        def _segment(path: Path) -> str:
            try:
                rel = path.relative_to(in_path)
            except ValueError:
                rel = Path(path.name)
            seg_dir = out_path / rel.parent
            segment_audio_file(str(path), str(seg_dir), duration=duration, overlap=overlap)
            return str(path)

        result = run_batch(files, _segment, continue_on_error=True)
        _report(result, output_dir, quiet)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


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
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_visualize(
    input_dir, input_file, output_dir, plot_type, legend, max_workers, recursive, quiet
) -> None:
    """Batch generate audio visualizations."""
    from bioamla.viz import batch_generate_spectrograms

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        in_dir = _require_input_dir(config)
        if not output_dir:
            raise InvalidInputError("--output-dir is required for this command.")
        if not quiet:
            click.echo(f"Generating {plot_type} visualizations...")

        stats = batch_generate_spectrograms(
            in_dir,
            output_dir,
            viz_type=plot_type,
            recursive=recursive,
            verbose=not quiet,
        )
        if not quiet:
            click.echo(
                f"Processed {stats['files_processed']} files, {stats['files_failed']} failed"
            )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
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
        from bioamla.detect.batch import _build_detector

        detector = _build_detector(method, params)

        def _process(path: Path) -> None:
            detector.detect_from_file(path)

        result = _run_csv_process(config, _process)
        _report(result, None, quiet)
        return

    in_dir = _require_input_dir(config)
    out_dir = output_dir or in_dir
    result = batch_detect_dir(
        in_dir,
        out_dir,
        method=method,
        recursive=recursive,
        max_workers=config.max_workers,
        **params,
    )
    _report(result, output_dir, quiet, saved_hint=f"detections_{method}.json")


@detect.command("energy")
@batch_input_options
@batch_output_options
@click.option("--low-freq", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option("--threshold-db", default=-20.0, type=float, help="Detection threshold (dB)")
@click.option("--min-duration", default=0.05, type=float, help="Minimum detection duration (s)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
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
    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            click.echo("Detecting energy...")
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
@click.option("--window-duration", default=2.0, type=float, help="Analysis window duration (s)")
@click.option("--min-score", default=0.3, type=float, help="Minimum detection score")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
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
    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            click.echo("Detecting RIBBIT calls...")
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
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
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
    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            click.echo("Detecting peaks...")
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
@click.option("--window-duration", default=3.0, type=float, help="Analysis window duration (s)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
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
    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            click.echo("Detecting accelerating patterns...")
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


@batch.group()
def indices() -> None:
    """Batch acoustic indices operations."""
    pass


@indices.command("calculate")
@batch_input_options
@batch_output_options
@click.option(
    "--indices",
    default="aci,adi,aei,bio,ndsi,h_spectral,h_temporal",
    help="Comma-separated indices",
)
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def indices_calculate(
    input_dir, input_file, output_dir, indices, max_workers, recursive, quiet
) -> None:
    """Batch calculate acoustic indices for audio files."""
    import csv

    from bioamla.audio.discovery import list_audio_files
    from bioamla.indices import batch_compute_indices, compute_indices_from_file

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            click.echo("Calculating acoustic indices...")

        if config.input_file:

            def _analyze(path: Path) -> dict:
                result = compute_indices_from_file(path).to_dict()
                # Drop redundant filename/path-ish keys before merge.
                result.pop("filepath", None)
                result.pop("filename", None)
                return result

            result = _run_csv_merge(config, _analyze)
            _report(result, None, quiet)
            return

        filepaths = list_audio_files(_require_input_dir(config), recursive=recursive)
        results = batch_compute_indices(filepaths, verbose=not quiet)

        if output_dir and results:
            out_path = Path(output_dir) / "indices.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames: list = []
            for row in results:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        if not quiet:
            successful = sum(1 for r in results if r.get("success"))
            failed = len(results) - successful
            click.echo(f"Processed {len(results)} files: {successful} successful, {failed} failed")
            if output_dir:
                click.echo(f"Results saved to {output_dir}/indices.csv")
            for r in results:
                if not r.get("success"):
                    click.echo(f"  Error: {r.get('filepath')}: {r.get('error')}")
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
    "--segment-duration",
    default=0,
    type=int,
    help="Split each file into N-second segments and classify each (0 = whole file)",
)
@click.option("--overlap", default=0, type=int, help="Overlap between segments (seconds)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
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

    from bioamla.ml import batch_predict_files, batch_predict_segments

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not quiet:
            click.echo(f"Running AST predictions with model {model}...")

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
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def models_embed(input_dir, input_file, output_dir, model, max_workers, recursive, quiet) -> None:
    """Batch extract embeddings from audio files."""
    from bioamla.ml import batch_embed_files

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not output_dir:
            raise InvalidInputError("--output-dir is required for this command.")
        if not quiet:
            click.echo(f"Extracting AST embeddings with model {model}...")

        if config.input_file:
            import numpy as _np

            from bioamla.ml.embedding import EmbeddingConfig, EmbeddingExtractor

            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            extractor = EmbeddingExtractor(config=EmbeddingConfig(model_path=model))

            def _process(path: Path) -> None:
                emb = extractor.extract(str(path))
                _np.save(str(out_dir / f"{path.stem}_embeddings.npy"), emb.embeddings)

            result = _run_csv_process(config, _process)
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
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
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
    from bioamla.cluster import cluster_batch_files

    try:
        config = _build_config(input_dir, input_file, output_dir, recursive, max_workers, quiet)
        if not output_dir:
            raise InvalidInputError("--output-dir is required for this command.")
        if not quiet:
            click.echo(f"Clustering embeddings using {method}...")

        if config.input_file:
            from bioamla.cluster.batch import cluster_embedding_files

            context = _csv_context(config)
            embedding_paths = [str(row.file_path) for row in context.rows]
            result = cluster_embedding_files(
                embedding_paths,
                output_dir,
                method=method,
                n_clusters=n_clusters,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            write_csv(context)
            _report(result, output_dir, quiet, saved_hint="cluster_assignments.json")
            return

        result = cluster_batch_files(
            input_dir,
            output_dir,
            method=method,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            recursive=recursive,
        )
        _report(result, output_dir, quiet, saved_hint="cluster_assignments.json")
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

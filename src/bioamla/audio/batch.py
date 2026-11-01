"""
Audio Batch Processing
======================

Batch wrappers for the audio domain built on :func:`bioamla.batch.run_batch`.
These let cut-over wire the batch CLI to the audio domain later without
depending on the old service layer.

Each wrapper discovers audio files under a directory, applies a per-file
transform, and returns a :class:`bioamla.batch.BatchResult`.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bioamla.audio.convert import convert_audio_file
from bioamla.audio.discovery import list_audio_files
from bioamla.audio.io import load_audio, process_file, save_audio
from bioamla.batch import (
    BatchResult,
    SegmentInfo,
    output_stems,
    run_batch,
    write_rows_csv,
)
from bioamla.batch import config_csv_context as _csv_context
from bioamla.batch import require_input_dir as _input_dir
from bioamla.batch import require_output_dir as _output_dir
from bioamla.common.constants import (
    DEFAULT_CMAP,
    DEFAULT_DPI,
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_N_MFCC,
)
from bioamla.exceptions import InvalidInputError, NotFoundError
from bioamla.protocols import OnProgress

if TYPE_CHECKING:
    from bioamla.audio.features import FrameFeatureConfig
    from bioamla.batch import BatchConfig


def batch_transform_files(
    input_dir: str,
    output_dir: str,
    processor_fn: Callable[[np.ndarray, int], np.ndarray],
    *,
    sample_rate: int | None = None,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
    operation: str = "transform",
    keep_channels: bool = False,
) -> BatchResult:
    """
    Apply a per-file transform to every audio file in a directory.

    Args:
        input_dir: Directory containing the input audio files.
        output_dir: Directory to write the processed ``.wav`` files to.
        processor_fn: Callable taking ``(audio, sample_rate)`` and returning the
            processed audio array. With ``max_workers > 1`` it runs in worker
            threads, so it must be thread-safe.
        sample_rate: Optional target sample rate for the outputs.
        recursive: Search subdirectories.
        max_workers: Number of worker threads (1 = sequential).
        continue_on_error: Collect per-file errors and keep going if True.
        on_progress: Optional ``(completed, total)`` progress callback.
        operation: Name used in the batch log records (e.g. the command).
        keep_channels: Keep each file's channels, applying ``processor_fn`` to each
            (default: files are mixed down to mono first).

    Returns:
        A :class:`bioamla.batch.BatchResult` summarizing the run.

    Raises:
        NotFoundError: If the input directory does not exist.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    files = [str(f) for f in list_audio_files(str(in_dir), recursive=recursive)]
    stems = output_stems(files, in_dir)

    def _process_one(audio_path: str) -> str:
        stem = stems[audio_path]
        out_path = out_dir / stem.parent / f"{stem.name}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        return process_file(
            audio_path, str(out_path), processor_fn, sample_rate, keep_channels=keep_channels
        )

    return run_batch(
        files,
        _process_one,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation=operation,
    )


def batch_resample_files(
    input_dir: str,
    output_dir: str,
    target_sample_rate: int,
    *,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
) -> BatchResult:
    """
    Resample every audio file in a directory to ``target_sample_rate``.

    See :func:`batch_transform_files` for shared argument semantics.
    """
    return batch_transform_files(
        input_dir,
        output_dir,
        processor_fn=lambda audio, sr: audio,
        sample_rate=target_sample_rate,
        keep_channels=True,
        recursive=recursive,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
    )


def batch_convert_files(
    input_dir: str,
    output_dir: str,
    *,
    target_format: str = "wav",
    sample_rate: int | None = None,
    channels: int | None = None,
    bit_depth: int | None = None,
    delete_original: bool = False,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
) -> BatchResult:
    """
    Convert every audio file in a directory to ``target_format``.

    Preserves the relative directory structure under ``output_dir`` and changes
    each file's extension to match the target format. Optionally re-channels,
    resamples, and deletes the originals.

    See :func:`bioamla.audio.convert.convert_audio_file` for the per-file
    semantics.

    Raises:
        NotFoundError: If the input directory does not exist.
        InvalidInputError: If ``target_format`` or ``channels`` is unsupported, or
            ``bit_depth`` doesn't apply to the format (before anything is written).
    """
    from bioamla.audio.convert import check_convert_options

    check_convert_options(target_format, channels, bit_depth)
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    files = [str(f) for f in list_audio_files(str(in_dir), recursive=recursive)]
    stems = output_stems(files, in_dir)

    def _process_one(audio_path: str) -> str:
        stem = stems[audio_path]
        out_path = out_dir / stem.parent / f"{stem.name}.{target_format}"
        return convert_audio_file(
            Path(audio_path),
            out_path,
            target_format=target_format,
            target_sample_rate=sample_rate,
            target_channels=channels,
            target_bit_depth=bit_depth,
            delete_original=delete_original,
        )

    return run_batch(
        files,
        _process_one,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation="convert",
    )


def segment_audio_file(
    input_path: str,
    output_dir: str,
    *,
    duration: float,
    overlap: float = 0.0,
    prefix: str | None = None,
    format: str = "wav",
    include_partial: bool = False,
) -> list[SegmentInfo]:
    """
    Split one audio file into fixed-duration segments written under ``output_dir``.

    Segments are named ``<prefix>_seg<index:04d>.<format>`` (``rec_seg0000.wav``).

    Args:
        input_path: Source audio file.
        output_dir: Directory the segment files are written to.
        duration: Segment duration in seconds.
        overlap: Overlap between consecutive segments in seconds.
        prefix: Filename prefix for segments (defaults to the input stem).
        format: Segment file format ("wav", "mp3", "flac", "ogg").
        include_partial: Keep a last segment shorter than ``duration`` (by default
            every segment is exactly ``duration`` long and a shorter tail is left out).

    Returns:
        A list of :class:`bioamla.batch.SegmentInfo`, one per written segment.

    Raises:
        InvalidInputError: If ``duration`` is not positive, ``overlap`` is not
            smaller than ``duration``, ``prefix`` isn't a plain name (so clips stay
            inside ``output_dir``), or the file is shorter than ``duration`` without
            ``include_partial`` (it would give no clip at all).

    If writing fails partway, the segments already written are removed.
    """
    if duration <= 0:
        raise InvalidInputError("--duration must be positive.")
    if prefix is not None and (prefix in ("", ".", "..") or any(c in prefix for c in "/\\")):
        raise InvalidInputError(f"prefix must be a plain file-name prefix, not {prefix!r}")

    src = Path(input_path)
    out_dir = Path(output_dir)
    stem = prefix if prefix is not None else src.stem

    audio, sr = load_audio(str(src))
    seg_samples = int(duration * sr)
    step = int((duration - overlap) * sr)
    if step <= 0:
        raise InvalidInputError("overlap must be smaller than duration.")
    n = audio.shape[-1] if audio.ndim > 1 else len(audio)
    if n == 0:
        raise InvalidInputError(f"{src.name} has no audio to cut")
    if n < seg_samples and not include_partial:
        raise InvalidInputError(
            f"{src.name} is {n / sr:.3g} s long, shorter than one {duration:g} s clip "
            "(keep it as a shorter clip with --keep-short / include_partial)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)  # only once there is something to write

    segments: list[SegmentInfo] = []
    count = 0
    start_idx = 0
    try:
        while start_idx < n:
            if not include_partial and start_idx + seg_samples > n:
                break
            end_idx = min(start_idx + seg_samples, n)
            chunk = audio[..., start_idx:end_idx] if audio.ndim > 1 else audio[start_idx:end_idx]
            seg_out = out_dir / f"{stem}_seg{count:04d}.{format}"
            save_audio(str(seg_out), chunk, sr)
            segments.append(
                SegmentInfo(
                    segment_path=seg_out,
                    segment_id=count,
                    start_time=start_idx / sr,
                    end_time=end_idx / sr,
                    duration=(end_idx - start_idx) / sr,
                )
            )
            count += 1
            if end_idx >= n:  # the clip reached the end: one shorter last clip at most
                break
            start_idx += step
    except BaseException:
        # Don't leave a partial set behind for a later run to mix with: remove the
        # finished segments and the one being written (also on Ctrl-C).
        for seg in segments:
            seg.segment_path.unlink(missing_ok=True)
        if start_idx < n:
            (out_dir / f"{stem}_seg{count:04d}.{format}").unlink(missing_ok=True)
        raise

    return segments


def batch_info_files(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
) -> BatchResult:
    """
    Read the metadata of every audio file in a directory.

    Args:
        input_dir: Directory containing audio files.
        output_dir: If given, ``audio_info.csv`` is written here, one row per
            file that could be read, in discovery order.
        recursive: Search subdirectories.
        max_workers: Number of worker threads (1 = sequential).
        continue_on_error: Collect per-file errors and keep going if True.
        on_progress: Optional ``(completed, total)`` progress callback.

    Returns:
        A :class:`bioamla.batch.BatchResult`. Its ``metadata["rows"]`` holds
        the per-file rows (``file_name``, ``duration``, ``sample_rate``,
        ``channels``, ``samples``, ``format``); ``output_files`` holds the CSV
        path when one was written.

    Raises:
        NotFoundError: If the input directory does not exist.
    """
    from bioamla.audio.info import get_audio_info

    in_dir = Path(input_dir)
    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")

    files = [str(f) for f in list_audio_files(str(in_dir), recursive=recursive)]
    by_path: dict[str, dict] = {}

    def _info(path: str) -> None:
        info = get_audio_info(path)
        by_path[path] = {
            "file_name": path,
            "duration": info.duration,
            "sample_rate": info.sample_rate,
            "channels": info.channels,
            "samples": info.samples,
            "format": info.format,
        }

    result = run_batch(
        files,
        _info,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation="info",
    )
    rows = [by_path[f] for f in files if f in by_path]
    result.metadata["rows"] = rows
    if output_dir is not None and rows:
        result.output_files = [str(write_rows_csv(Path(output_dir) / "audio_info.csv", rows))]
    return result


def batch_segment_files(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    duration: float,
    overlap: float = 0.0,
    format: str = "wav",
    include_partial: bool = False,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
) -> BatchResult:
    """
    Split every audio file in a directory into fixed-duration segments.

    Segments mirror the input tree under ``output_dir`` and are named
    ``<stem>_seg0000.<format>`` and so on (see :func:`segment_audio_file`).

    Args:
        input_dir: Directory containing audio files.
        output_dir: Directory the segments are written under.
        duration: Segment duration in seconds.
        overlap: Overlap between consecutive segments in seconds.
        format: Segment file format ("wav", "mp3", "flac", "ogg").
        include_partial: Keep each file's last segment when it is shorter than
            ``duration``.
        recursive: Search subdirectories.
        max_workers: Number of worker threads (1 = sequential).
        continue_on_error: Collect per-file errors and keep going if True.
        on_progress: Optional ``(completed, total)`` progress callback.

    Returns:
        A :class:`bioamla.batch.BatchResult` whose ``output_files`` lists every
        segment written, in input order.

    Raises:
        NotFoundError: If the input directory does not exist.
        InvalidInputError: If ``duration`` is not positive or ``overlap`` is not
            smaller than ``duration``.
    """
    if duration <= 0:
        raise InvalidInputError("duration must be positive.")
    if overlap >= duration:
        raise InvalidInputError("overlap must be smaller than duration.")

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    files = [str(f) for f in list_audio_files(str(in_dir), recursive=recursive)]
    stems = output_stems(files, in_dir)
    by_path: dict[str, list[SegmentInfo]] = {}

    def _segment(path: str) -> None:
        stem = stems[path]
        by_path[path] = segment_audio_file(
            path,
            str(out_dir / stem.parent),
            duration=duration,
            overlap=overlap,
            prefix=stem.name,
            format=format,
            include_partial=include_partial,
        )

    result = run_batch(
        files,
        _segment,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation="segment",
    )
    result.output_files = [str(seg.segment_path) for f in files for seg in by_path.get(f, [])]
    return result


def frame_feature_table_paths(
    files: list[str], root: str | Path, output_dir: str | Path | None, file_format: str
) -> dict[str, Path]:
    """Where each file's per-frame table goes: ``<stem>.features.<file_format>``.

    With ``output_dir`` the tables mirror ``root``'s tree under it; without, each
    table sits next to its audio. Stems come from :func:`bioamla.batch.output_stems`,
    so files that would share a name (``a.wav`` and ``a.flac``, or two ``rec.wav``
    from outside ``root``) get distinct tables.
    """
    stems = output_stems(files, root)
    paths: dict[str, Path] = {}
    for f in files:
        stem = stems[f]
        name = f"{stem.name}.features.{file_format}"
        base = Path(f).parent if output_dir is None else Path(output_dir) / stem.parent
        paths[f] = base / name
    return paths


def batch_frame_features_files(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    config: "FrameFeatureConfig | None" = None,
    trajectory: bool = False,
    gate_db: float = 38.0,
    file_format: str = "csv",
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
) -> BatchResult:
    """
    Write a per-frame feature table for every audio file in a directory.

    Tables mirror the input tree under ``output_dir`` as
    ``<stem>.features.csv`` (or ``.parquet``), named by
    :func:`frame_feature_table_paths`; see :func:`frame_features_file`.

    Args:
        input_dir: Directory containing audio files.
        output_dir: Directory to write the tables to.
        config: Analysis settings for :func:`bioamla.audio.frame_features`.
        trajectory: Add ``pc1``..``pc3`` trajectory columns.
        gate_db: Level gate for the trajectory.
        file_format: ``"csv"`` or ``"parquet"``.
        recursive: Search subdirectories.
        max_workers: Number of worker threads (1 = sequential).
        continue_on_error: Collect per-file errors and keep going if True.
        on_progress: Optional ``(completed, total)`` progress callback.

    Returns:
        A :class:`bioamla.batch.BatchResult`. ``output_files`` holds the tables
        written and the index ``frame_features.csv`` (``file_name``,
        ``features_file``, ``n_frames`` and ``duration`` per file, in discovery
        order), whose rows are also in ``metadata["rows"]``.

    Raises:
        NotFoundError: If the input directory does not exist.
        InvalidInputError: If ``file_format`` is not csv or parquet.
    """
    from bioamla.audio.features import frame_features_file

    if file_format not in ("csv", "parquet"):
        raise InvalidInputError(f"file_format must be 'csv' or 'parquet', got {file_format!r}")
    in_dir = Path(input_dir)
    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")
    out_dir = Path(output_dir)
    files = [str(f) for f in list_audio_files(str(in_dir), recursive=recursive)]
    tables = frame_feature_table_paths(files, in_dir, out_dir, file_format)
    by_path: dict[str, dict] = {}

    def _process_one(audio_path: str) -> str:
        out_path = tables[audio_path]
        result = frame_features_file(
            audio_path, out_path, config=config, trajectory=trajectory, gate_db=gate_db
        )
        by_path[audio_path] = {
            "file_name": audio_path,
            "features_file": result.output_path,
            "n_frames": result.features.n_frames,
            "duration": result.features.duration,
        }
        return str(out_path)

    result = run_batch(
        files,
        _process_one,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation="features",
    )
    rows = [by_path[f] for f in files if f in by_path]
    result.metadata["rows"] = rows
    if rows:
        index = write_rows_csv(out_dir / "frame_features.csv", rows)
        result.output_files = [*result.output_files, str(index)]
    return result


# =============================================================================
# Per-file processors for batch_transform (module level, so they pickle)
# =============================================================================


def _normalize_rms(audio: np.ndarray, sr: int, *, target_db: float) -> np.ndarray:
    from bioamla.audio.processing import normalize_loudness

    return normalize_loudness(audio, sr, target_db=target_db)


def _normalize_peak(audio: np.ndarray, sr: int) -> np.ndarray:
    from bioamla.audio.processing import peak_normalize

    return peak_normalize(audio)


def _trim_range(
    audio: np.ndarray, sr: int, *, start: float | None, end: float | None
) -> np.ndarray:
    from bioamla.audio.processing import trim_audio

    return trim_audio(audio, sr, start_time=start, end_time=end)


def _trim_silence(audio: np.ndarray, sr: int, *, threshold_db: float) -> np.ndarray:
    from bioamla.audio.processing import trim_silence

    return trim_silence(audio, sr, threshold_db=threshold_db)


def _denoise(audio: np.ndarray, sr: int, *, strength: float) -> np.ndarray:
    from bioamla.audio.processing import spectral_denoise

    return spectral_denoise(audio, sr, noise_reduce_factor=strength)


def _filter(
    audio: np.ndarray,
    sr: int,
    *,
    lowpass: float | None,
    highpass: float | None,
    bandpass: tuple[float, float] | None,
    order: int,
) -> np.ndarray:
    from bioamla.audio.processing import bandpass_filter, highpass_filter, lowpass_filter

    if bandpass is not None:
        return bandpass_filter(audio, sr, bandpass[0], bandpass[1], order=order)
    if lowpass is not None:
        return lowpass_filter(audio, sr, lowpass, order=order)
    return highpass_filter(audio, sr, highpass, order=order)


def _passthrough(audio: np.ndarray, sr: int) -> np.ndarray:
    return audio


def normalize_processor(*, target_db: float = -20.0, peak: bool = False) -> Callable:
    """A :func:`batch_transform` processor: RMS loudness to ``target_db``, or (with
    ``peak``) peak normalization."""
    import functools

    return _normalize_peak if peak else functools.partial(_normalize_rms, target_db=target_db)


def trim_processor(
    *,
    start: float | None = None,
    end: float | None = None,
    silence_threshold_db: float | None = None,
) -> Callable:
    """A :func:`batch_transform` processor: keep ``start``..``end`` (s), or, with
    ``silence_threshold_db``, trim leading and trailing silence below it."""
    import functools

    if silence_threshold_db is not None:
        return functools.partial(_trim_silence, threshold_db=silence_threshold_db)
    return functools.partial(_trim_range, start=start, end=end)


def denoise_processor(*, strength: float = 1.0) -> Callable:
    """A :func:`batch_transform` processor: spectral noise reduction (0–2)."""
    import functools

    return functools.partial(_denoise, strength=strength)


def filter_processor(
    *,
    lowpass: float | None = None,
    highpass: float | None = None,
    bandpass_low: float | None = None,
    bandpass_high: float | None = None,
    order: int = 5,
) -> Callable:
    """A :func:`batch_transform` processor: one Butterworth filter, band-pass before
    low-pass before high-pass (as :func:`~bioamla.audio.filter_audio_file`).

    Raises:
        InvalidInputError: If no filter is given, or only one band-pass edge.
    """
    import functools

    if (bandpass_low is not None) != (bandpass_high is not None):
        raise InvalidInputError("Both bandpass_low and bandpass_high must be specified together")
    if lowpass is None and highpass is None and bandpass_low is None:
        raise InvalidInputError("Must specify lowpass, highpass, or bandpass_low and bandpass_high")
    bandpass = (bandpass_low, bandpass_high) if bandpass_low is not None else None
    return functools.partial(
        _filter, lowpass=lowpass, highpass=highpass, bandpass=bandpass, order=order
    )


# =============================================================================
# Batch commands, directory or CSV mode (from a BatchConfig)
# =============================================================================


def batch_transform(
    config: "BatchConfig",
    processor: Callable[[np.ndarray, int], np.ndarray],
    *,
    sample_rate: int | None = None,
    on_progress: OnProgress | None = None,
    operation: str = "batch audio transform",
    keep_channels: bool = False,
) -> BatchResult:
    """Apply ``processor`` to every file of a batch and save the results as WAV.

    Directory mode writes into ``config.output_dir`` (required), mirroring the input
    tree; CSV mode writes in place or under ``config.output_dir`` and updates each
    row's ``file_name``.

    Args:
        config: The batch input and output (see :class:`~bioamla.batch.BatchConfig`).
        processor: ``processor(audio, sample_rate) -> audio``, e.g. from
            :func:`normalize_processor`; it must pickle for ``max_workers > 1``.
        sample_rate: Resample the outputs to this rate.
        on_progress: ``(done, total)`` callback.
        operation: Name used in the batch log records.
        keep_channels: Keep each file's channels, applying ``processor`` to each
            (default: files are mixed down to mono first).

    Raises:
        InvalidInputError: If directory mode has no ``output_dir``.
    """
    if config.input_file:
        from bioamla.audio.io import process_file
        from bioamla.batch import run_csv_transform

        def _transform(in_path: Path, out_path: Path) -> Path:
            wav_out = out_path.with_suffix(".wav")
            return Path(process_file(str(in_path), str(wav_out), processor, sample_rate,
                                     keep_channels=keep_channels))  # fmt: skip

        return run_csv_transform(
            _csv_context(config),
            _transform,
            new_extension=".wav",
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            on_progress=on_progress,
            operation=operation,
        )
    input_dir = _input_dir(config)
    return batch_transform_files(
        input_dir,
        _output_dir(config),
        processor_fn=processor,
        sample_rate=sample_rate,
        recursive=config.recursive,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        on_progress=on_progress,
        operation=operation,
        keep_channels=keep_channels,
    )


def batch_resample(
    config: "BatchConfig",
    sample_rate: int,
    *,
    on_progress: OnProgress | None = None,
    operation: str = "batch audio resample",
) -> BatchResult:
    """Resample every file of a batch to ``sample_rate``, keeping its channels (see
    :func:`batch_transform`)."""
    return batch_transform(
        config, _passthrough, sample_rate=sample_rate, on_progress=on_progress,
        operation=operation, keep_channels=True,
    )  # fmt: skip


def batch_visualize(
    config: "BatchConfig",
    *,
    viz_type: str = "mel",
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    n_mfcc: int = DEFAULT_N_MFCC,
    cmap: str = DEFAULT_CMAP,
    dpi: int = DEFAULT_DPI,
    legend: bool = True,
    verbose: bool = True,
    on_progress: OnProgress | None = None,
) -> BatchResult:
    """Draw a spectrogram, MFCC or waveform image of every file in a folder, with axes,
    title and colorbar unless ``legend`` is ``False``. The drawing settings default to
    :func:`bioamla.viz.generate_spectrogram`'s, so each picture matches ``audio
    visualize``'s of the same file.

    Directory mode only; images mirror the input tree under ``config.output_dir``
    (required). See :func:`bioamla.viz.batch_generate_spectrograms`.

    Raises:
        InvalidInputError: If ``config`` is CSV mode or has no ``output_dir``.
    """
    from bioamla.viz import batch_generate_spectrograms

    stats = batch_generate_spectrograms(
        _input_dir(config),
        _output_dir(config),
        viz_type=viz_type,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        cmap=cmap,
        dpi=dpi,
        show_colorbar=legend,
        recursive=config.recursive,
        verbose=verbose,
        on_progress=on_progress,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
    )
    return BatchResult(
        total_files=stats["files_processed"] + stats["files_failed"],
        successful=stats["files_processed"],
        failed=stats["files_failed"],
        output_files=stats.get("output_files", []),
        errors=stats.get("errors", []),
    )


def _info_columns(path: Path) -> dict:
    from bioamla.audio.info import get_audio_info

    info = get_audio_info(str(path))
    return {
        "duration": info.duration,
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "samples": info.samples,
        "format": info.format,
    }


def batch_audio_info(
    config: "BatchConfig",
    *,
    on_progress: OnProgress | None = None,
    operation: str = "batch audio info",
) -> BatchResult:
    """Read every file's header (duration, sample rate, channels, samples, format).

    Directory mode writes ``audio_info.csv`` into ``config.output_dir`` if given (see
    :func:`batch_info_files`); CSV mode adds the columns to the metadata CSV.
    """
    if config.input_file:
        from bioamla.batch import run_csv_merge

        return run_csv_merge(
            _csv_context(config),
            _info_columns,
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            on_progress=on_progress,
            operation=operation,
        )
    return batch_info_files(
        _input_dir(config),
        str(config.output_dir) if config.output_dir else None,
        recursive=config.recursive,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        on_progress=on_progress,
    )


def _check_wav(path: Path) -> dict:
    from bioamla.audio._riffy import is_wav_path
    from bioamla.audio.info import validate_wav
    from bioamla.exceptions import AudioLoadError

    if not is_wav_path(path):
        return {"valid": "skipped", "error": "not a WAV file"}
    try:
        validate_wav(str(path))
    except AudioLoadError as e:
        return {"valid": "false", "error": str(e)}
    return {"valid": "true", "error": ""}


def batch_validate_wavs(
    config: "BatchConfig",
    *,
    on_progress: OnProgress | None = None,
    operation: str = "batch audio validate",
) -> BatchResult:
    """Check that every WAV is well-formed PCM or float (header only).

    Non-WAV files are skipped. Directory mode counts an invalid file as failed and
    writes ``validation_report.csv`` (``file_name``, ``valid``, ``error``) into
    ``config.output_dir`` if given; CSV mode adds ``valid`` and ``error`` columns.

    Returns:
        The run's result; ``metadata["invalid_files"]`` lists the invalid files in
        either mode.
    """
    from bioamla.batch import run_batch, run_csv_merge
    from bioamla.exceptions import AudioLoadError

    invalid: list[str] = []
    if config.input_file:

        def _check_row(path: Path) -> dict:
            row = _check_wav(path)
            if row["valid"] == "false":
                invalid.append(str(path))
            return row

        result = run_csv_merge(
            _csv_context(config),
            _check_row,
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            on_progress=on_progress,
            operation=operation,
        )
        result.metadata["invalid_files"] = sorted(invalid)
        return result

    files = list_audio_files(_input_dir(config), recursive=config.recursive)
    rows: list[dict] = []

    def _validate(path: Path) -> str:
        row = {"file_name": str(path), **_check_wav(path)}
        rows.append(row)
        if row["valid"] == "false":
            invalid.append(str(path))
            raise AudioLoadError(f"{path}: {row['error']}")
        return str(path)

    result = run_batch(
        files,
        _validate,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        on_progress=on_progress,
        operation=operation,
    )
    rows.sort(key=lambda r: r["file_name"])
    result.skipped = sum(1 for r in rows if r["valid"] == "skipped")
    result.metadata["invalid_files"] = sorted(invalid)
    if config.output_dir:
        report = Path(config.output_dir) / "validation_report.csv"
        write_rows_csv(report, rows, ["file_name", "valid", "error"])
        result.metadata["output_file"] = str(report)
    return result


def batch_recording_metadata(
    config: "BatchConfig",
    *,
    on_progress: OnProgress | None = None,
    operation: str = "batch audio metadata",
) -> BatchResult:
    """Read every WAV's recorder metadata (GUANO, AudioMoth, WAMD, bext).

    Non-WAV files are skipped. Directory mode writes ``recorder_metadata.csv`` into
    ``config.output_dir`` if given and anything was read; CSV mode adds the columns
    to the metadata CSV.
    """
    from bioamla.audio._riffy import is_wav_path
    from bioamla.audio.metadata import RECORDING_METADATA_FIELDS, get_recording_metadata
    from bioamla.batch import run_batch, run_csv_merge

    if config.input_file:

        def _row_meta(path: Path) -> dict:
            if not is_wav_path(path):
                return {}
            return get_recording_metadata(str(path)).to_dict()

        return run_csv_merge(
            _csv_context(config),
            _row_meta,
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            on_progress=on_progress,
            operation=operation,
        )

    all_files = list_audio_files(_input_dir(config), recursive=config.recursive)
    files = [f for f in all_files if is_wav_path(f)]
    rows: list[dict] = []

    def _meta(path: Path) -> str:
        rows.append({"file_name": str(path), **get_recording_metadata(str(path)).to_dict()})
        return str(path)

    result = run_batch(
        files,
        _meta,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        on_progress=on_progress,
        operation=operation,
    )
    result.skipped = len(all_files) - len(files)
    rows.sort(key=lambda r: r["file_name"])
    if config.output_dir and rows:
        out = Path(config.output_dir) / "recorder_metadata.csv"
        write_rows_csv(out, rows, ["file_name", *RECORDING_METADATA_FIELDS])
        result.metadata["output_file"] = str(out)
    return result


def batch_frame_features(
    config: "BatchConfig",
    *,
    analysis: "FrameFeatureConfig | None" = None,
    trajectory: bool = False,
    gate_db: float = 38.0,
    file_format: str = "csv",
    on_progress: OnProgress | None = None,
    operation: str = "batch audio features",
) -> BatchResult:
    """Write a per-frame feature table for every file.

    Directory mode mirrors the input tree under ``config.output_dir`` (required) and
    writes an index, ``frame_features.csv`` (see :func:`batch_frame_features_files`).
    CSV mode writes each table under ``config.output_dir`` (mirroring the CSV's
    folder) or next to its audio, and adds ``features_file`` and ``n_frames``.
    """
    from bioamla.audio.features import FrameFeatureConfig, frame_features_file

    analysis = analysis or FrameFeatureConfig()
    if config.input_file:
        from bioamla.batch import run_csv_merge

        context = _csv_context(config)
        listed = [str(row.file_path) for row in context.rows]
        tables = frame_feature_table_paths(listed, context.csv_dir, context.output_dir, file_format)

        def _analyze(path: Path) -> dict:
            result = frame_features_file(
                path, tables[str(path)], config=analysis, trajectory=trajectory, gate_db=gate_db
            )
            # A Path: written relative to wherever the metadata CSV is saved.
            return {
                "features_file": Path(result.output_path).resolve(),
                "n_frames": result.features.n_frames,
            }

        return run_csv_merge(
            context,
            _analyze,
            clear=("features_file", "n_frames"),
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            on_progress=on_progress,
            operation=operation,
        )
    input_dir = _input_dir(config)
    return batch_frame_features_files(
        input_dir,
        _output_dir(config),
        config=analysis,
        trajectory=trajectory,
        gate_db=gate_db,
        file_format=file_format,
        recursive=config.recursive,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        on_progress=on_progress,
    )


def batch_convert(
    config: "BatchConfig",
    *,
    target_format: str = "wav",
    sample_rate: int | None = None,
    channels: int | None = None,
    bit_depth: int | None = None,
    delete_original: bool = False,
    on_progress: OnProgress | None = None,
    operation: str = "batch audio convert",
) -> BatchResult:
    """Convert every file to ``target_format`` (see :func:`convert_audio_file`).

    Directory mode writes into ``config.output_dir`` (required); CSV mode writes in
    place or under ``config.output_dir`` and updates each row's ``file_name``.

    Raises:
        InvalidInputError: If ``target_format`` or ``channels`` is unsupported, or
            ``bit_depth`` doesn't apply to the format (before anything is written).
    """
    from bioamla.audio.convert import check_convert_options

    check_convert_options(target_format, channels, bit_depth)
    if config.input_file:
        from bioamla.batch import run_csv_transform

        def _convert(in_path: Path, out_path: Path) -> Path:
            return Path(
                convert_audio_file(
                    in_path,
                    out_path,
                    target_format=target_format,
                    target_sample_rate=sample_rate,
                    target_channels=channels,
                    target_bit_depth=bit_depth,
                    delete_original=delete_original,
                )
            )

        return run_csv_transform(
            _csv_context(config),
            _convert,
            new_extension=f".{target_format}",
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            on_progress=on_progress,
            operation=operation,
        )
    input_dir = _input_dir(config)
    return batch_convert_files(
        input_dir,
        _output_dir(config),
        target_format=target_format,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
        delete_original=delete_original,
        recursive=config.recursive,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        on_progress=on_progress,
    )


def batch_segment(
    config: "BatchConfig",
    *,
    duration: float,
    overlap: float = 0.0,
    format: str = "wav",
    include_partial: bool = False,
    on_progress: OnProgress | None = None,
    operation: str = "batch audio segment",
) -> BatchResult:
    """Cut every file into ``duration``-second clips (see :func:`segment_audio_file`).

    Directory mode mirrors the input tree under ``config.output_dir`` (required); CSV
    mode writes the clips next to each file's output and replaces each row by one
    row per clip.

    Raises:
        InvalidInputError: If ``duration`` isn't positive, or directory mode has no
            ``output_dir``.
    """
    if duration <= 0:
        raise InvalidInputError(f"duration must be positive (got {duration})")
    if config.input_file:
        from bioamla.batch import run_csv_segment

        def _segment(in_path: Path, seg_dir: Path) -> list:
            return segment_audio_file(
                str(in_path), str(seg_dir), duration=duration, overlap=overlap,
                format=format, include_partial=include_partial,
            )  # fmt: skip

        return run_csv_segment(
            _csv_context(config),
            _segment,
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            on_progress=on_progress,
            operation=operation,
        )
    input_dir = _input_dir(config)
    return batch_segment_files(
        input_dir,
        _output_dir(config),
        duration=duration,
        overlap=overlap,
        format=format,
        include_partial=include_partial,
        recursive=config.recursive,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        on_progress=on_progress,
    )


__all__ = [
    "batch_transform_files",
    "batch_resample_files",
    "batch_convert_files",
    "batch_info_files",
    "batch_segment_files",
    "segment_audio_file",
    "batch_frame_features_files",
    "frame_feature_table_paths",
    # Batch commands, directory or CSV mode
    "batch_transform",
    "batch_resample",
    "batch_visualize",
    "batch_audio_info",
    "batch_validate_wavs",
    "batch_recording_metadata",
    "batch_frame_features",
    "batch_convert",
    "batch_segment",
    "normalize_processor",
    "trim_processor",
    "denoise_processor",
    "filter_processor",
]

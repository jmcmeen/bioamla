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

import numpy as np

from bioamla.audio.convert import convert_audio_file
from bioamla.audio.discovery import list_audio_files
from bioamla.audio.io import load_audio, save_audio
from bioamla.audio.processing import resample_audio
from bioamla.batch import BatchResult, SegmentInfo, output_stems, run_batch, write_rows_csv
from bioamla.exceptions import InvalidInputError, NotFoundError


def batch_transform_files(
    input_dir: str,
    output_dir: str,
    processor_fn: Callable[[np.ndarray, int], np.ndarray],
    *,
    sample_rate: int | None = None,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
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

        audio, sr = load_audio(audio_path)
        processed = processor_fn(audio, sr)
        if sample_rate is not None and sample_rate != sr:
            processed = resample_audio(processed, sr, sample_rate)
            sr = sample_rate
        return save_audio(str(out_path), processed, sr)

    return run_batch(
        files,
        _process_one,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
    )


def batch_resample_files(
    input_dir: str,
    output_dir: str,
    target_sample_rate: int,
    *,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
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
    delete_original: bool = False,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
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
        out_path = out_dir / stem.parent / f"{stem.name}.{target_format}"
        return convert_audio_file(
            Path(audio_path),
            out_path,
            target_format=target_format,
            target_sample_rate=sample_rate,
            target_channels=channels,
            delete_original=delete_original,
        )

    return run_batch(
        files,
        _process_one,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
    )


def segment_audio_file(
    input_path: str,
    output_dir: str,
    *,
    duration: float,
    overlap: float = 0.0,
    prefix: str | None = None,
) -> list[SegmentInfo]:
    """
    Split one audio file into fixed-duration segments written under ``output_dir``.

    Args:
        input_path: Source audio file.
        output_dir: Directory the segment ``.wav`` files are written to.
        duration: Segment duration in seconds.
        overlap: Overlap between consecutive segments in seconds.
        prefix: Filename prefix for segments (defaults to the input stem).

    Returns:
        A list of :class:`bioamla.batch.SegmentInfo`, one per written segment.

    Raises:
        InvalidInputError: If ``duration`` is not positive or ``overlap`` is not
            smaller than ``duration``.

    If writing fails partway, the segments already written are removed.
    """
    if duration <= 0:
        raise InvalidInputError("--duration must be positive.")

    src = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix if prefix is not None else src.stem

    audio, sr = load_audio(str(src))
    seg_samples = int(duration * sr)
    step = int((duration - overlap) * sr)
    if step <= 0:
        raise InvalidInputError("overlap must be smaller than duration.")

    n = audio.shape[-1] if audio.ndim > 1 else len(audio)
    segments: list[SegmentInfo] = []
    count = 0
    start_idx = 0
    try:
        while start_idx < n:
            end_idx = min(start_idx + seg_samples, n)
            chunk = audio[..., start_idx:end_idx] if audio.ndim > 1 else audio[start_idx:end_idx]
            seg_out = out_dir / f"{stem}_seg{count:04d}.wav"
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
            start_idx += step
    except Exception:
        # Don't leave a partial set behind for a later run to mix with.
        for seg in segments:
            seg.segment_path.unlink(missing_ok=True)
        raise

    return segments


def batch_info_files(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
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
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """
    Split every audio file in a directory into fixed-duration segments.

    Segments mirror the input tree under ``output_dir`` and are named
    ``<stem>_seg0000.wav`` and so on (see :func:`segment_audio_file`).

    Args:
        input_dir: Directory containing audio files.
        output_dir: Directory the segments are written under.
        duration: Segment duration in seconds.
        overlap: Overlap between consecutive segments in seconds.
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
        )

    result = run_batch(
        files,
        _segment,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
    )
    result.output_files = [str(seg.segment_path) for f in files for seg in by_path.get(f, [])]
    return result


__all__ = [
    "batch_transform_files",
    "batch_resample_files",
    "batch_convert_files",
    "batch_info_files",
    "batch_segment_files",
    "segment_audio_file",
]

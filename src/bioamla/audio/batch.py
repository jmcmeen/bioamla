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
from bioamla.batch import BatchResult, SegmentInfo, run_batch
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
            processed audio array. Must be picklable for parallel execution.
        sample_rate: Optional target sample rate for the outputs.
        recursive: Search subdirectories.
        max_workers: Number of worker processes (1 = sequential).
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
    files = list_audio_files(str(in_dir), recursive=recursive)

    def _process_one(audio_path: Path) -> str:
        try:
            rel_path = audio_path.relative_to(in_dir)
        except ValueError:
            rel_path = Path(audio_path.name)

        out_path = out_dir / rel_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        audio, sr = load_audio(str(audio_path))
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
    files = list_audio_files(str(in_dir), recursive=recursive)

    def _process_one(audio_path: Path) -> str:
        try:
            rel_path = audio_path.relative_to(in_dir)
        except ValueError:
            rel_path = Path(audio_path.name)
        out_path = out_dir / rel_path.with_suffix(f".{target_format}")
        return convert_audio_file(
            audio_path,
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

    return segments


__all__ = [
    "batch_transform_files",
    "batch_resample_files",
    "batch_convert_files",
    "segment_audio_file",
]

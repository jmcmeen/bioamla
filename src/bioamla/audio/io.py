"""
Audio File I/O
==============

Raising free functions for audio file I/O, ported from the old
``AudioFileService``. These functions return plain data and raise on error:

- :func:`load_audio_data`  — load a file into :class:`AudioData`
- :func:`save_audio_data`  — write :class:`AudioData` to disk
- :func:`save_audio_data_as` — write with optional resampling/format conversion
- :func:`create_temp_audio_file` — write to a temp file (caller owns cleanup)

Heavy/optional backends (pydub/ffmpeg) are imported lazily inside the functions.
"""

import logging
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from bioamla.audio.data import AudioData
from bioamla.common.files import prepare_output_path, require_exists
from bioamla.exceptions import AudioLoadError, AudioSaveError, DependencyError
from bioamla.protocols import OnProgress

logger = logging.getLogger(__name__)


def load_audio_data(
    filepath: str | Path,
    *,
    sample_rate: int | None = None,
    start: float | None = None,
    duration: float | None = None,
) -> AudioData:
    """
    Load an audio file, or a window of it, into an :class:`AudioData`.

    Passing ``start`` and/or ``duration`` reads only that part of the file,
    so a few seconds of a multi-hour recording never decode the rest. For
    walking through a whole long file, see :func:`stream_audio_data`.

    Args:
        filepath: Path to the audio file.
        sample_rate: If given, resample the loaded audio to this rate.
        start: Window start in seconds (default: the beginning).
        duration: Window length in seconds (default: to the end).

    Returns:
        The loaded :class:`AudioData`. A window records its offset in
        ``metadata["start_time"]``; it is shorter than ``duration`` if it runs
        past the end of the file.

    Raises:
        NotFoundError: If the file does not exist.
        InvalidInputError: If ``start`` is negative or ``duration`` not positive.
        AudioLoadError: If decoding fails.
    """
    from bioamla.audio._pydub import load_audio, load_audio_window

    path = require_exists(filepath)
    windowed = start is not None or duration is not None
    if windowed:
        _check_window(start or 0.0, duration)

    try:
        if windowed:
            audio, sr = load_audio_window(str(path), start or 0.0, duration)
        else:
            audio, sr = load_audio(str(path))
    except DependencyError:
        raise
    except Exception as e:
        raise AudioLoadError(f"Failed to open audio file: {e}") from e

    metadata: dict = {"start_time": start or 0.0} if windowed else {}
    return _to_audio_data(audio, sr, path, sample_rate, metadata)


def stream_audio_data(
    filepath: str | Path,
    *,
    block_duration: float,
    overlap: float = 0.0,
    sample_rate: int | None = None,
) -> Iterator[AudioData]:
    """
    Stream an audio file as consecutive fixed-length blocks.

    Only one block is in memory at a time, so this works on recordings far
    larger than RAM. WAV, FLAC and OGG are read straight from disk; other
    formats decode one block at a time through ffmpeg.

    Arguments and the file's existence are checked when this is called, before
    the first block is read.

    Args:
        filepath: Path to the audio file.
        block_duration: Block length in seconds.
        overlap: Seconds each block shares with the previous one (for analyses
            that need context across block edges).
        sample_rate: If given, resample each block to this rate. Blocks are
            resampled independently, so use some ``overlap`` if edge effects
            matter.

    Returns:
        An iterator of :class:`AudioData` blocks, each with its offset into the
        file in ``metadata["start_time"]``. The last block may be shorter.

    Raises:
        NotFoundError: If the file does not exist.
        InvalidInputError: If ``block_duration`` is not positive or ``overlap``
            is negative or not smaller than ``block_duration`` (on the call), or
            either spans less than one sample at the file's rate (on the first
            block).
        DependencyError: If the format needs ffmpeg and it isn't installed.
        AudioLoadError: If decoding fails.
    """
    from bioamla.exceptions import InvalidInputError

    if block_duration <= 0:
        raise InvalidInputError(f"block_duration must be positive, got {block_duration}")
    if overlap < 0 or overlap >= block_duration:
        raise InvalidInputError(
            f"overlap must be at least 0 and less than block_duration, got {overlap}"
        )
    path = require_exists(filepath)
    return _stream_blocks(path, block_duration, overlap, sample_rate)


def _stream_blocks(
    path: Path, block_duration: float, overlap: float, sample_rate: int | None
) -> Iterator[AudioData]:
    from bioamla.audio._pydub import iter_audio_blocks
    from bioamla.exceptions import BioamlaError

    blocks = iter_audio_blocks(str(path), block_duration, overlap)
    while True:
        try:
            audio, sr, block_start = next(blocks)
        except StopIteration:
            return
        except BioamlaError:
            raise
        except Exception as e:
            raise AudioLoadError(f"Failed to stream audio file: {e}") from e
        yield _to_audio_data(audio, sr, path, sample_rate, {"start_time": block_start})


def _check_window(start: float, duration: float | None) -> None:
    from bioamla.exceptions import InvalidInputError

    if start < 0:
        raise InvalidInputError(f"start must not be negative, got {start}")
    if duration is not None and duration <= 0:
        raise InvalidInputError(f"duration must be positive, got {duration}")


def _to_audio_data(
    audio: np.ndarray, sr: int, path: Path, sample_rate: int | None, metadata: dict
) -> AudioData:
    """Wrap decoded samples in an :class:`AudioData`, resampling if asked."""
    # Ensure 1D for mono
    if audio.ndim == 1:
        channels = 1
    else:
        channels = audio.shape[1] if audio.shape[1] <= 2 else 1
        if channels == 1:
            audio = audio.flatten()

    audio_data = AudioData(
        samples=audio,
        sample_rate=sr,
        channels=channels,
        source_path=str(path.resolve()),
        is_modified=False,
        metadata={"original_duration": len(audio) / sr, **metadata},
    )

    if sample_rate is not None and sample_rate != audio_data.sample_rate:
        try:
            from bioamla.audio.processing import resample_audio

            resampled = resample_audio(audio_data.samples, audio_data.sample_rate, sample_rate)
        except Exception as e:
            raise AudioLoadError(f"Resampling failed: {e}") from e
        audio_data = AudioData(
            samples=resampled,
            sample_rate=sample_rate,
            channels=audio_data.channels,
            source_path=audio_data.source_path,
            is_modified=True,
            metadata=audio_data.metadata.copy(),
        )

    return audio_data


def save_audio_data(
    audio: AudioData, output_path: str | Path, *, format: str | None = None
) -> Path:
    """
    Save :class:`AudioData` to a file.

    Args:
        audio: Audio data to save.
        output_path: Destination file path.
        format: Audio format (auto-detected from extension if not specified).

    Returns:
        The output :class:`~pathlib.Path`.

    Raises:
        AudioSaveError: If encoding/writing fails.
    """
    from bioamla.audio._pydub import save_audio

    output = prepare_output_path(output_path)

    try:
        save_audio(
            str(output),
            audio.samples,
            audio.sample_rate,
            format=format,
        )
    except Exception as e:
        raise AudioSaveError(f"Failed to save audio: {e}") from e

    return output


def save_audio_data_as(
    audio: AudioData,
    output_path: str | Path,
    *,
    target_sample_rate: int | None = None,
    format: str | None = None,
) -> Path:
    """
    Save :class:`AudioData` to a new file, optionally resampling.

    Args:
        audio: Audio data to save.
        output_path: Destination file path.
        target_sample_rate: Resample to this rate before saving (optional).
        format: Audio format (auto-detected from extension if not specified).

    Returns:
        The output :class:`~pathlib.Path`.

    Raises:
        AudioSaveError: If resampling or encoding/writing fails.
    """
    data_to_save = audio
    if target_sample_rate and target_sample_rate != audio.sample_rate:
        try:
            from bioamla.audio.processing import resample_audio

            resampled = resample_audio(
                audio.samples,
                audio.sample_rate,
                target_sample_rate,
            )
        except Exception as e:
            raise AudioSaveError(f"Resampling failed: {e}") from e

        data_to_save = AudioData(
            samples=resampled,
            sample_rate=target_sample_rate,
            channels=audio.channels,
            source_path=audio.source_path,
            is_modified=True,
            metadata=audio.metadata.copy(),
        )

    return save_audio_data(data_to_save, output_path, format=format)


def create_temp_audio_file(audio: AudioData, suffix: str = ".wav") -> Path:
    """
    Write :class:`AudioData` to a temporary file and return its path.

    The temporary file is created with ``delete=False``; the caller owns
    cleanup of the returned path.

    Args:
        audio: Audio data to write.
        suffix: File extension for the temporary file.

    Returns:
        Path to the created temporary file.

    Raises:
        AudioSaveError: If encoding/writing fails.
    """
    from bioamla.audio._pydub import save_audio

    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        save_audio(str(temp_path), audio.samples, audio.sample_rate)
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise AudioSaveError(f"Failed to create temp file: {e}") from e

    return temp_path


# =============================================================================
# Array-level I/O (folded from core/signal.py)
# =============================================================================


def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file as a mono ``numpy`` array (channels averaged).

    Args:
        filepath: Path to the audio file.

    Returns:
        Tuple of (audio array, sample rate). The array is mono float32.

    Raises:
        AudioLoadError: If the file is missing or decoding fails.
    """
    from bioamla.audio._pydub import load_audio as _load

    return _loaded(_load, filepath)


def load_audio_channels(filepath: str | Path) -> tuple[np.ndarray, int]:
    """
    Load an audio file keeping every channel (unlike :func:`load_audio`, which
    downmixes to mono).

    Args:
        filepath: Path to the audio file.

    Returns:
        Tuple of (audio array, sample rate): float32, 1-D for a mono file and
        ``(n_samples, n_channels)`` otherwise.

    Raises:
        AudioLoadError: If the file is missing or decoding fails.
    """
    from bioamla.audio._pydub import load_audio_channels as _load

    return _loaded(_load, filepath)


def _loaded(load: Callable[[str], tuple[np.ndarray, int]], filepath: str | Path):
    try:
        audio, sr = load(str(filepath))
    except AudioLoadError:
        raise  # already says what failed
    except Exception as e:  # a missing file too: reported as a load failure
        raise AudioLoadError(f"Failed to load audio file: {e}") from e
    return audio.astype(np.float32, copy=False), sr


def save_audio(
    filepath: str, audio: np.ndarray, sample_rate: int, format: str | None = None
) -> str:
    """
    Save a ``numpy`` audio array to a file.

    Args:
        filepath: Destination file path.
        audio: Audio data as a numpy array.
        sample_rate: Sample rate in Hz.
        format: Output format (auto-detected from the extension if not given).

    Returns:
        Path to the saved file (as a string).

    Raises:
        AudioSaveError: If encoding/writing fails.
    """
    from bioamla.audio._pydub import save_audio as _pydub_save_audio

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _pydub_save_audio(str(path), audio, sample_rate, format=format)
    except Exception as e:
        raise AudioSaveError(f"Failed to save audio: {e}") from e
    return str(path)


def load_waveform_tensor(filepath: str) -> tuple:
    """
    Load an audio file as a ``torch`` waveform tensor.

    Thin re-export of the canonical torch-IO implementation in
    :mod:`bioamla.audio.torchaudio`, kept here for the public ``bioamla.audio``
    surface.

    Args:
        filepath: Path to the audio file.

    Returns:
        Tuple of (waveform tensor, sample rate).

    Raises:
        AudioLoadError: If decoding fails.
    """
    from bioamla.audio.torchaudio import load_waveform_tensor as _load_waveform_tensor

    return _load_waveform_tensor(filepath)


def process_file(
    input_path: str,
    output_path: str,
    processor_fn: Callable[[np.ndarray, int], np.ndarray],
    sample_rate: int | None = None,
) -> str:
    """
    Load, process, and save a single audio file.

    Args:
        input_path: Path to the input file.
        output_path: Path to the output file.
        processor_fn: Callable taking ``(audio, sample_rate)`` and returning
            the processed audio array.
        sample_rate: Optional target sample rate for the output.

    Returns:
        Path to the output file.

    Raises:
        AudioLoadError: If the input cannot be loaded.
        AudioSaveError: If the output cannot be saved.
    """
    from bioamla.audio.processing import resample_audio

    audio, sr = load_audio(input_path)

    # Process
    processed = processor_fn(audio, sr)

    # Resample if needed
    if sample_rate is not None and sample_rate != sr:
        processed = resample_audio(processed, sr, sample_rate)
        sr = sample_rate

    return save_audio(output_path, processed, sr)


def batch_process(
    input_dir: str,
    output_dir: str,
    processor_fn: Callable[[np.ndarray, int], np.ndarray],
    sample_rate: int | None = None,
    recursive: bool = True,
    verbose: bool = True,
    *,
    max_workers: int = 1,
    on_progress: OnProgress | None = None,
) -> dict:
    """
    Process all audio files in a directory.

    Per-file failures are caught and logged (graceful batch behaviour).

    Args:
        input_dir: Path to the input directory.
        output_dir: Path to the output directory.
        processor_fn: Callable taking ``(audio, sample_rate)`` and returning the
            processed audio array.
        sample_rate: Optional target sample rate for the output.
        recursive: Search subdirectories.
        verbose: Log progress at INFO level.
        max_workers: Worker threads; 1 runs sequentially.
        on_progress: Optional ``(completed, total)`` progress callback.

    Returns:
        Statistics dict with ``files_processed``, ``files_failed``, ``output_dir``.

    Raises:
        NotFoundError: If the input directory does not exist.
    """
    from bioamla.audio.discovery import get_audio_files
    from bioamla.batch import output_stems, run_batch
    from bioamla.exceptions import NotFoundError

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = get_audio_files(str(in_dir), recursive=recursive)

    if not audio_files:
        if verbose:
            logger.info(f"No audio files found in {in_dir}")
        return {"files_processed": 0, "files_failed": 0, "output_dir": str(out_dir)}

    if verbose:
        logger.info(f"Found {len(audio_files)} audio files to process")

    files = [str(f) for f in audio_files]
    stems = output_stems(files, in_dir)

    def _process(audio_path: str) -> Path:
        stem = stems[audio_path]
        out_path = out_dir / stem.parent / f"{stem.name}.wav"
        process_file(audio_path, str(out_path), processor_fn, sample_rate)
        if verbose:
            logger.info(f"  Processed: {out_path}")
        return out_path

    result = run_batch(
        files, _process, max_workers=max_workers, on_progress=on_progress, operation="process"
    )

    return {
        "files_processed": result.successful,
        "files_failed": result.failed,
        "output_dir": str(out_dir),
    }


__all__ = [
    "load_audio_data",
    "save_audio_data",
    "save_audio_data_as",
    "create_temp_audio_file",
    "load_audio",
    "load_audio_channels",
    "save_audio",
    "load_waveform_tensor",
    "process_file",
    "batch_process",
]

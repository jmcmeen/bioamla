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

import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bioamla.audio.data import AudioData
from bioamla.common.files import prepare_output_path, require_exists
from bioamla.exceptions import AudioLoadError, AudioSaveError, DependencyError


def load_audio_data(filepath: str | Path, *, sample_rate: int | None = None) -> AudioData:
    """
    Load an audio file into an :class:`AudioData`.

    Args:
        filepath: Path to the audio file.
        sample_rate: If given, resample the loaded audio to this rate.

    Returns:
        The loaded :class:`AudioData`.

    Raises:
        NotFoundError: If the file does not exist.
        AudioLoadError: If decoding fails.
    """
    from bioamla.adapters.pydub import load_audio

    path = require_exists(filepath)

    try:
        audio, sr = load_audio(str(path))
    except Exception as e:
        raise AudioLoadError(f"Failed to open audio file: {e}") from e

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
        metadata={"original_duration": len(audio) / sr},
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
    from bioamla.adapters.pydub import save_audio

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
    from bioamla.adapters.pydub import save_audio

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
    Load an audio file as a mono ``numpy`` array.

    Args:
        filepath: Path to the audio file.

    Returns:
        Tuple of (audio array, sample rate). The array is mono float32.

    Raises:
        AudioLoadError: If decoding fails.
    """
    from bioamla.adapters.pydub import load_audio as _pydub_load_audio

    try:
        audio, sr = _pydub_load_audio(filepath)
    except Exception as e:
        raise AudioLoadError(f"Failed to load audio file: {e}") from e

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)

    return audio.astype(np.float32), sr


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
    from bioamla.adapters.pydub import save_audio as _pydub_save_audio

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _pydub_save_audio(str(path), audio, sample_rate, format=format)
    except Exception as e:
        raise AudioSaveError(f"Failed to save audio: {e}") from e
    return str(path)


def load_waveform_tensor(filepath: str):  # noqa: ANN201 - torch optional
    """
    Load an audio file as a ``torch`` waveform tensor.

    Args:
        filepath: Path to the audio file.

    Returns:
        Tuple of (waveform tensor, sample rate).

    Raises:
        DependencyError: If ``torchaudio`` is not installed.
        AudioLoadError: If decoding fails.
    """
    try:
        import torchaudio
    except ImportError as err:
        raise DependencyError(
            "loading waveform tensors requires torchaudio — install bioamla[ml]"
        ) from err

    try:
        waveform, sample_rate = torchaudio.load(filepath)
    except Exception as e:
        raise AudioLoadError(f"Failed to load audio file: {e}") from e
    return waveform, sample_rate


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
        verbose: Print progress.

    Returns:
        Statistics dict with ``files_processed``, ``files_failed``, ``output_dir``.

    Raises:
        NotFoundError: If the input directory does not exist.
    """
    from bioamla.audio.discovery import get_audio_files
    from bioamla.exceptions import NotFoundError

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = get_audio_files(str(in_dir), recursive=recursive)

    if not audio_files:
        if verbose:
            print(f"No audio files found in {in_dir}")
        return {"files_processed": 0, "files_failed": 0, "output_dir": str(out_dir)}

    if verbose:
        print(f"Found {len(audio_files)} audio files to process")

    files_processed = 0
    files_failed = 0

    for audio_path in audio_files:
        audio_path = Path(audio_path)

        try:
            rel_path = audio_path.relative_to(in_dir)
        except ValueError:
            rel_path = Path(audio_path.name)

        out_path = out_dir / rel_path.with_suffix(".wav")

        try:
            process_file(str(audio_path), str(out_path), processor_fn, sample_rate)
            files_processed += 1
            if verbose:
                print(f"  Processed: {out_path}")
        except Exception as e:
            files_failed += 1
            if verbose:
                print(f"  Failed: {audio_path} - {e}")

    if verbose:
        print(f"Processed {files_processed} files, {files_failed} failed")

    return {
        "files_processed": files_processed,
        "files_failed": files_failed,
        "output_dir": str(out_dir),
    }


__all__ = [
    "load_audio_data",
    "save_audio_data",
    "save_audio_data_as",
    "create_temp_audio_file",
    "load_audio",
    "save_audio",
    "load_waveform_tensor",
    "process_file",
    "batch_process",
]

"""
Audio Format Conversion
=======================

Format / channel / sample-rate conversion. :func:`convert_audio_file` loads an audio
file with its channels, optionally re-channels and resamples it, saves it in the
target format, and optionally deletes the original. A stereo (or multi-channel)
file stays that way unless ``target_channels`` says otherwise.

Raises :class:`~bioamla.exceptions.*` on failure (via the underlying I/O
helpers); ``numpy`` is the only hard dependency at import time.
"""

from pathlib import Path

import numpy as np

from bioamla.audio.io import load_audio_channels, save_audio
from bioamla.audio.processing import resample_audio
from bioamla.exceptions import InvalidInputError

__all__ = ["convert_audio_file"]

SUPPORTED_FORMATS = {"wav", "mp3", "flac", "ogg"}


def _rechannel(audio: np.ndarray, current_channels: int, target_channels: int) -> np.ndarray:
    """Convert ``audio`` to mono or stereo.

    Mono arrays are 1D; others are 2D with shape ``(n_samples, n_channels)`` (the
    layout :func:`bioamla.audio.io.save_audio` expects). Going to stereo from more
    than two channels downmixes them first.
    """
    if target_channels == current_channels:
        return audio

    if target_channels == 1:
        # Down-mix to mono.
        if audio.ndim == 2:
            return audio.mean(axis=1).astype(np.float32)
        return audio
    if target_channels == 2:
        # Up-mix mono to stereo by duplicating the channel.
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        return np.column_stack([mono, mono]).astype(np.float32)

    raise InvalidInputError(f"Unsupported target channel count: {target_channels}")


def convert_audio_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_format: str | None = "wav",
    target_sample_rate: int | None = None,
    target_channels: int | None = None,
    delete_original: bool = False,
) -> str:
    """Convert an audio file to a target format, optionally re-channel/resample.

    Args:
        input_path: Source audio file.
        output_path: Destination file path (its suffix should match
            ``target_format``).
        target_format: Output container/codec ("wav", "mp3", "flac", "ogg"), or
            ``None`` to choose it from ``output_path``'s extension.
        target_sample_rate: Resample to this rate before saving (optional).
        target_channels: Output channel count (1 = mono, 2 = stereo); by default
            the file keeps its own channels.
        delete_original: Delete the source file after a successful conversion
            (only when the output path differs from the input path).

    Returns:
        The output path as a string.

    Raises:
        InvalidInputError: If ``target_format``/``target_channels`` is unsupported, or
            the file has more than 2 channels for MP3 output without ``target_channels``.
        AudioLoadError / AudioSaveError: On I/O failure.
    """
    fmt = None if target_format is None else target_format.lower().lstrip(".")
    if fmt is not None and fmt not in SUPPORTED_FORMATS:
        raise InvalidInputError(
            f"Unsupported target format: {target_format}. "
            f"Choose one of {sorted(SUPPORTED_FORMATS)}."
        )

    if target_channels not in (None, 1, 2):
        raise InvalidInputError(f"Unsupported target channel count: {target_channels} (1 or 2)")
    in_path = Path(input_path)
    out_path = Path(output_path)

    audio, sr = load_audio_channels(in_path)
    current_channels = 1 if audio.ndim == 1 else audio.shape[1]
    out_fmt = fmt or out_path.suffix.lower().lstrip(".")
    if target_channels is None and current_channels > 2 and out_fmt == "mp3":
        raise InvalidInputError(
            f"{in_path.name} has {current_channels} channels and MP3 holds at most 2; "
            "choose 1 or 2 channels, or another format"
        )

    # Fewer channels first, so resampling works on what's kept.
    if target_channels is not None and target_channels < current_channels:
        audio = _rechannel(audio, current_channels, target_channels)
        current_channels = target_channels
    if target_sample_rate is not None and target_sample_rate != sr:
        # resample_audio works along the last axis; channels are columns here.
        audio = resample_audio(audio.T, sr, target_sample_rate).T
        sr = target_sample_rate
    if target_channels is not None:
        audio = _rechannel(audio, current_channels, target_channels)

    saved = save_audio(str(out_path), audio, sr, format=fmt)

    if delete_original and out_path.resolve() != in_path.resolve():
        in_path.unlink(missing_ok=True)

    return saved

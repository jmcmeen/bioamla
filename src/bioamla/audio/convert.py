"""
Audio Format Conversion
=======================

Format / channel / sample-rate conversion folded from the old
``BatchAudioTransformService`` convert path. :func:`convert_audio_file` loads an
audio file (via the pydub backend), optionally re-channels and resamples it,
saves it in the target format, and optionally deletes the original.

Raises :class:`~bioamla.exceptions.*` on failure (via the underlying I/O
helpers); ``numpy`` is the only hard dependency at import time.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from bioamla.audio.io import load_audio, save_audio
from bioamla.audio.processing import resample_audio
from bioamla.exceptions import InvalidInputError

__all__ = ["convert_audio_file"]

SUPPORTED_FORMATS = {"wav", "mp3", "flac", "ogg"}


def _rechannel(audio: np.ndarray, current_channels: int, target_channels: int) -> np.ndarray:
    """Convert ``audio`` between mono and stereo.

    Mono arrays are 1D; stereo arrays are 2D with shape ``(n_samples, 2)`` (the
    layout :func:`bioamla.audio.io.save_audio` expects).
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
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    target_format: str = "wav",
    target_sample_rate: Optional[int] = None,
    target_channels: Optional[int] = None,
    delete_original: bool = False,
) -> str:
    """Convert an audio file to a target format, optionally re-channel/resample.

    Args:
        input_path: Source audio file.
        output_path: Destination file path (its suffix should match
            ``target_format``).
        target_format: Output container/codec ("wav", "mp3", "flac", "ogg").
        target_sample_rate: Resample to this rate before saving (optional).
        target_channels: Output channel count (1 = mono, 2 = stereo); optional.
        delete_original: Delete the source file after a successful conversion
            (only when the output path differs from the input path).

    Returns:
        The output path as a string.

    Raises:
        InvalidInputError: If ``target_format``/``target_channels`` is unsupported.
        AudioLoadError / AudioSaveError: On I/O failure.
    """
    fmt = target_format.lower().lstrip(".")
    if fmt not in SUPPORTED_FORMATS:
        raise InvalidInputError(
            f"Unsupported target format: {target_format}. "
            f"Choose one of {sorted(SUPPORTED_FORMATS)}."
        )

    in_path = Path(input_path)
    out_path = Path(output_path)

    # load_audio returns mono float32; treat as 1 channel.
    audio, sr = load_audio(str(in_path))
    current_channels = 1 if audio.ndim == 1 else audio.shape[1]

    if target_sample_rate is not None and target_sample_rate != sr:
        audio = resample_audio(audio, sr, target_sample_rate)
        sr = target_sample_rate

    if target_channels is not None:
        audio = _rechannel(audio, current_channels, target_channels)

    saved = save_audio(str(out_path), audio, sr, format=fmt)

    if delete_original and out_path.resolve() != in_path.resolve():
        in_path.unlink(missing_ok=True)

    return saved

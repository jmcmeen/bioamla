"""
TorchAudio waveform helpers (ML extra).

Helper functions for working with audio as ``torch`` waveform tensors:
loading, resampling, splitting into segments, and loading from raw bytes.
These are used by the ML domain (AST training/inference) and the dataset
augmentation pipeline.

``torch`` / ``torchaudio`` are optional dependencies (the ``[ml]`` extra) and
are imported lazily inside each function so this module is importable on a slim
install; calling a function without them installed raises
:class:`~bioamla.exceptions.DependencyError`.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np

from bioamla.common.constants import DefaultConfig
from bioamla.common.files import get_files_by_extension
from bioamla.exceptions import DependencyError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


def _import_torchaudio() -> Any:
    """Import and return the ``torchaudio`` module, or raise DependencyError."""
    try:
        import torchaudio
    except ImportError as e:
        raise DependencyError(
            "torchaudio waveform helpers require torchaudio — install bioamla[ml]"
        ) from e
    return torchaudio


def _import_torch() -> Any:
    """Import and return the ``torch`` module, or raise DependencyError."""
    try:
        import torch
    except ImportError as e:
        raise DependencyError(
            "torchaudio waveform helpers require torch — install bioamla[ml]"
        ) from e
    return torch


def get_wav_info(filepath: str) -> Any:
    """Get metadata about a WAV file (sample rate, frames, channels, etc.)."""
    torchaudio = _import_torchaudio()
    return torchaudio.info(filepath)


def get_wav_files(directory: str) -> list:
    """Get a list of all WAV files in a directory."""
    return get_files_by_extension(directory, [".wav"])


def get_wavefile_shape(wavefile_path: str) -> torch.Size:
    """Get the shape (channels, samples) of an audio waveform."""
    torchaudio = _import_torchaudio()
    waveform, _ = torchaudio.load(wavefile_path)
    return waveform.shape


def get_wavefile_sample_rate(wavefile_path: str) -> int:
    """Get the sample rate of an audio file."""
    torchaudio = _import_torchaudio()
    _, sample_rate = torchaudio.load(wavefile_path)
    return sample_rate


def load_waveform_tensor(filepath: str) -> tuple[torch.Tensor, int]:
    """
    Load an audio file as a waveform tensor.

    Args:
        filepath: Path to the audio file.

    Returns:
        Tuple of (waveform tensor, sample rate).

    Raises:
        DependencyError: If ``torchaudio`` is not installed.
    """
    torchaudio = _import_torchaudio()
    waveform, sample_rate = torchaudio.load(filepath)
    return (waveform, sample_rate)


def split_waveform_tensor(
    waveform_tensor: torch.Tensor,
    freq: int,
    segment_duration: int,
    segment_overlap: int,
) -> list[tuple[torch.Tensor, int, int]]:
    """
    Split a waveform tensor into overlapping fixed-length segments.

    Args:
        waveform_tensor: Input waveform tensor.
        freq: Sample rate of the audio.
        segment_duration: Duration of each segment in seconds.
        segment_overlap: Overlap between consecutive segments in seconds.

    Returns:
        List of (segment_tensor, start_sample, end_sample) tuples.
    """
    segment_size = int(segment_duration * freq)
    step_size = int((segment_duration - segment_overlap) * freq)

    segments = []
    start = 0
    while start + segment_size <= waveform_tensor.shape[1]:
        segment = waveform_tensor[:, start : start + segment_size]
        segments.append((segment, start, start + segment_size))
        start += step_size
    return segments


def resample_waveform_tensor(
    waveform_tensor: torch.Tensor, orig_freq: int, new_freq: int
) -> torch.Tensor:
    """
    Resample a waveform tensor to a different sample rate.

    Args:
        waveform_tensor: Input waveform tensor.
        orig_freq: Original sample rate in Hz.
        new_freq: Target sample rate in Hz.

    Returns:
        Resampled waveform tensor.

    Raises:
        DependencyError: If ``torchaudio`` is not installed.
    """
    torchaudio = _import_torchaudio()
    resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
    return resampler(waveform_tensor)


def load_audio_from_bytes(
    audio_bytes: bytes, target_sr: int = DefaultConfig.SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    """
    Load audio from raw bytes, convert to mono, resample, and return as a 1D array.

    Args:
        audio_bytes: Raw audio file data as bytes.
        target_sr: Target sample rate for output.

    Returns:
        Tuple of (audio_array, sample_rate).

    Raises:
        DependencyError: If ``torch`` / ``torchaudio`` are not installed.
        ValueError: If the audio bytes cannot be processed.
    """
    torch = _import_torch()
    torchaudio = _import_torchaudio()
    try:
        audio_io = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_io)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)

        audio_array = waveform.squeeze().numpy()
        return audio_array, target_sr
    except Exception as e:
        raise ValueError("Could not process audio bytes") from e


__all__ = [
    "get_wav_info",
    "get_wav_files",
    "get_wavefile_shape",
    "get_wavefile_sample_rate",
    "load_waveform_tensor",
    "split_waveform_tensor",
    "resample_waveform_tensor",
    "load_audio_from_bytes",
]

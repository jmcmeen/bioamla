"""
Audio File Information & Analysis
================================

File-level metadata and complete-analysis helpers folded from ``core/audio.py``.
These functions take a file path, read the file (lazily via the pydub backend),
and return small dataclasses. They raise :class:`NotFoundError` when the path
does not exist and :class:`AudioLoadError` when decoding fails.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bioamla.audio.analysis import (
    AmplitudeStats,
    FrequencyStats,
    SilenceInfo,
    detect_silence,
    get_amplitude_stats,
    get_frequency_stats,
)
from bioamla.exceptions import AudioLoadError, NotFoundError

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AudioInfo:
    """
    Basic audio file information.

    Attributes:
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels.
        samples: Total number of samples.
        bit_depth: Bit depth (if available).
        format: Audio format (e.g., 'WAV', 'FLAC').
        subtype: Audio subtype (e.g., 'PCM_16').
    """

    duration: float
    sample_rate: int
    channels: int
    samples: int
    bit_depth: int | None = None
    format: str | None = None
    subtype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "samples": self.samples,
            "bit_depth": self.bit_depth,
            "format": self.format,
            "subtype": self.subtype,
        }


@dataclass
class AudioAnalysis:
    """
    Complete audio analysis results.

    Attributes:
        file_path: Path to the analyzed audio file.
        info: Basic audio information.
        amplitude: Amplitude statistics.
        frequency: Frequency statistics.
        silence: Silence detection results.
    """

    file_path: str
    info: AudioInfo
    amplitude: AmplitudeStats
    frequency: FrequencyStats
    silence: SilenceInfo

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "info": self.info.to_dict(),
            "amplitude": self.amplitude.to_dict(),
            "frequency": self.frequency.to_dict(),
            "silence": self.silence.to_dict(),
        }


# =============================================================================
# Basic Audio Information
# =============================================================================


def get_audio_info(filepath: str) -> AudioInfo:
    """
    Get basic information about an audio file.

    Extracts metadata without loading the entire audio into memory.

    Args:
        filepath: Path to the audio file.

    Returns:
        :class:`AudioInfo` with duration, sample rate, channels, etc.

    Raises:
        NotFoundError: If the file does not exist.
        AudioLoadError: If metadata cannot be extracted.
    """
    from bioamla.audio._pydub import get_audio_info as _pydub_get_audio_info

    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Audio file not found: {filepath}")

    try:
        info_dict = _pydub_get_audio_info(str(filepath))
    except Exception as e:
        raise AudioLoadError(f"Failed to read audio info: {e}") from e

    return AudioInfo(
        duration=info_dict["duration"],
        sample_rate=info_dict["sample_rate"],
        channels=info_dict["channels"],
        samples=info_dict["samples"],
        bit_depth=info_dict.get("bit_depth"),
        format=info_dict["format"],
        subtype=info_dict.get("subtype"),
    )


def get_duration(filepath: str) -> float:
    """Get the duration of an audio file in seconds."""
    return get_audio_info(filepath).duration


def get_sample_rate(filepath: str) -> int:
    """Get the sample rate of an audio file in Hz."""
    return get_audio_info(filepath).sample_rate


def get_channels(filepath: str) -> int:
    """Get the number of channels in an audio file."""
    return get_audio_info(filepath).channels


# =============================================================================
# Complete Analysis
# =============================================================================


def analyze_audio(
    filepath: str,
    silence_threshold_db: float = -40,
) -> AudioAnalysis:
    """
    Perform complete audio analysis on a file.

    Combines metadata, amplitude, frequency, and silence analysis into a single
    comprehensive result.

    Args:
        filepath: Path to the audio file.
        silence_threshold_db: Threshold for silence detection in dBFS.

    Returns:
        :class:`AudioAnalysis` with all analysis results.

    Raises:
        NotFoundError: If the file does not exist.
        AudioLoadError: If the audio cannot be loaded.
    """
    from bioamla.audio._pydub import load_audio

    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Audio file not found: {filepath}")

    # Get basic info
    info = get_audio_info(filepath)

    # Load audio
    try:
        audio, sr = load_audio(filepath)
    except Exception as e:
        raise AudioLoadError(f"Failed to load audio file: {e}") from e

    # Analyze
    amplitude = get_amplitude_stats(audio)
    frequency = get_frequency_stats(audio, sr)
    silence = detect_silence(audio, sr, threshold_db=silence_threshold_db)

    return AudioAnalysis(
        file_path=str(path),
        info=info,
        amplitude=amplitude,
        frequency=frequency,
        silence=silence,
    )


def analyze_audio_batch(
    filepaths: list[str],
    silence_threshold_db: float = -40,
    verbose: bool = True,
) -> list[AudioAnalysis]:
    """
    Analyze multiple audio files.

    Failures are logged and skipped (graceful batch behaviour) rather than
    aborting the whole run.

    Args:
        filepaths: List of paths to audio files.
        silence_threshold_db: Threshold for silence detection in dBFS.
        verbose: Print progress.

    Returns:
        List of :class:`AudioAnalysis` objects for the files that succeeded.
    """
    results = []

    for i, filepath in enumerate(filepaths):
        try:
            analysis = analyze_audio(filepath, silence_threshold_db)
            results.append(analysis)
            if verbose:
                print(f"[{i + 1}/{len(filepaths)}] Analyzed: {filepath}")
        except Exception as e:
            logger.warning(f"Error analyzing {filepath}: {e}")
            if verbose:
                print(f"[{i + 1}/{len(filepaths)}] Error: {filepath} - {e}")

    return results


def summarize_analysis(analyses: list[AudioAnalysis]) -> dict[str, Any]:
    """
    Summarize analysis results from multiple files.

    Args:
        analyses: List of :class:`AudioAnalysis` objects.

    Returns:
        Dictionary with summary statistics.
    """
    if not analyses:
        return {
            "total_files": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "avg_rms_db": 0.0,
            "avg_peak_db": 0.0,
            "avg_peak_frequency": 0.0,
            "silent_file_count": 0,
        }

    durations = [a.info.duration for a in analyses]
    rms_dbs = [a.amplitude.rms_db for a in analyses if not np.isinf(a.amplitude.rms_db)]
    peak_dbs = [a.amplitude.peak_db for a in analyses if not np.isinf(a.amplitude.peak_db)]
    peak_freqs = [a.frequency.peak_frequency for a in analyses]
    silent_count = sum(1 for a in analyses if a.silence.is_silent)

    return {
        "total_files": len(analyses),
        "total_duration": sum(durations),
        "avg_duration": np.mean(durations),
        "min_duration": min(durations),
        "max_duration": max(durations),
        "avg_rms_db": np.mean(rms_dbs) if rms_dbs else -np.inf,
        "avg_peak_db": np.mean(peak_dbs) if peak_dbs else -np.inf,
        "avg_peak_frequency": np.mean(peak_freqs),
        "min_peak_frequency": min(peak_freqs),
        "max_peak_frequency": max(peak_freqs),
        "silent_file_count": silent_count,
        "silent_file_ratio": silent_count / len(analyses),
    }


__all__ = [
    "AudioInfo",
    "AudioAnalysis",
    "get_audio_info",
    "get_duration",
    "get_sample_rate",
    "get_channels",
    "analyze_audio",
    "analyze_audio_batch",
    "summarize_analysis",
]

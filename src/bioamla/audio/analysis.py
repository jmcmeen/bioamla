"""
Audio Amplitude / Frequency / Silence Analysis
==============================================

Pure ``numpy``-based analysis of in-memory audio samples, folded from
``core/audio.py``. These functions accept raw ``numpy`` arrays (and a sample
rate where needed) and return small dataclasses describing the signal.

They are deterministic numeric routines: they raise :class:`ValueError` only on
genuinely bad parameters and otherwise do not raise domain exceptions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AmplitudeStats:
    """
    Amplitude statistics for audio.

    Attributes:
        rms: Root mean square amplitude (0.0 to 1.0)
        rms_db: RMS level in dB (relative to full scale)
        peak: Peak absolute amplitude (0.0 to 1.0)
        peak_db: Peak level in dBFS
        crest_factor: Peak to RMS ratio (in dB)
        dynamic_range: Difference between peak and RMS in dB
    """

    rms: float
    rms_db: float
    peak: float
    peak_db: float
    crest_factor: float
    dynamic_range: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rms": self.rms,
            "rms_db": self.rms_db,
            "peak": self.peak,
            "peak_db": self.peak_db,
            "crest_factor": self.crest_factor,
            "dynamic_range": self.dynamic_range,
        }


@dataclass
class FrequencyStats:
    """
    Frequency statistics for audio.

    Attributes:
        peak_frequency: Frequency with highest magnitude in Hz
        peak_magnitude: Magnitude at peak frequency
        mean_frequency: Weighted mean frequency in Hz
        min_frequency: Lowest significant frequency in Hz
        max_frequency: Highest significant frequency in Hz
        bandwidth: Frequency bandwidth (max - min) in Hz
        spectral_centroid: Spectral centroid (center of mass) in Hz
        spectral_rolloff: Frequency below which 85% of energy is contained
    """

    peak_frequency: float
    peak_magnitude: float
    mean_frequency: float
    min_frequency: float
    max_frequency: float
    bandwidth: float
    spectral_centroid: float
    spectral_rolloff: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "peak_frequency": self.peak_frequency,
            "peak_magnitude": self.peak_magnitude,
            "mean_frequency": self.mean_frequency,
            "min_frequency": self.min_frequency,
            "max_frequency": self.max_frequency,
            "bandwidth": self.bandwidth,
            "spectral_centroid": self.spectral_centroid,
            "spectral_rolloff": self.spectral_rolloff,
        }


@dataclass
class SilenceInfo:
    """
    Silence detection results.

    Attributes:
        is_silent: True if audio is considered silent
        silence_ratio: Ratio of silent samples to total samples (0.0-1.0)
        sound_ratio: Ratio of non-silent samples to total samples (0.0-1.0)
        silent_segments: List of (start_time, end_time) tuples for silent regions
        sound_segments: List of (start_time, end_time) tuples for non-silent regions
        threshold_used: The amplitude threshold used for detection
    """

    is_silent: bool
    silence_ratio: float
    sound_ratio: float
    silent_segments: List[Tuple[float, float]] = field(default_factory=list)
    sound_segments: List[Tuple[float, float]] = field(default_factory=list)
    threshold_used: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_silent": self.is_silent,
            "silence_ratio": self.silence_ratio,
            "sound_ratio": self.sound_ratio,
            "silent_segments": self.silent_segments,
            "sound_segments": self.sound_segments,
            "threshold_used": self.threshold_used,
        }


# =============================================================================
# Amplitude Analysis
# =============================================================================


def calculate_rms(audio: np.ndarray) -> float:
    """
    Calculate the Root Mean Square (RMS) amplitude of audio.

    Args:
        audio: Audio data as numpy array.

    Returns:
        RMS amplitude (0.0 to 1.0 for normalized audio).
    """
    if audio.ndim > 1:
        audio = audio.flatten()

    return float(np.sqrt(np.mean(audio**2)))


def calculate_dbfs(amplitude: float, reference: float = 1.0) -> float:
    """
    Convert linear amplitude to decibels relative to full scale (dBFS).

    Args:
        amplitude: Linear amplitude value.
        reference: Reference amplitude for full scale (default: 1.0).

    Returns:
        Amplitude in dBFS (always negative or zero for normalized audio).
    """
    if amplitude <= 0:
        return -np.inf

    return float(20 * np.log10(amplitude / reference))


def calculate_peak(audio: np.ndarray) -> float:
    """
    Calculate the peak absolute amplitude of audio.

    Args:
        audio: Audio data as numpy array.

    Returns:
        Peak absolute amplitude (0.0 to 1.0+ for normalized audio).
    """
    if audio.ndim > 1:
        audio = audio.flatten()

    return float(np.max(np.abs(audio)))


def get_amplitude_stats(audio: np.ndarray) -> AmplitudeStats:
    """
    Calculate comprehensive amplitude statistics for audio.

    Args:
        audio: Audio data as numpy array.

    Returns:
        :class:`AmplitudeStats` with RMS, peak, dBFS values, etc.
    """
    rms = calculate_rms(audio)
    peak = calculate_peak(audio)

    rms_db = calculate_dbfs(rms)
    peak_db = calculate_dbfs(peak)

    # Crest factor: ratio of peak to RMS
    if rms > 0:
        crest_factor = peak_db - rms_db
    else:
        crest_factor = 0.0

    # Dynamic range
    dynamic_range = abs(rms_db - peak_db)

    return AmplitudeStats(
        rms=rms,
        rms_db=rms_db,
        peak=peak,
        peak_db=peak_db,
        crest_factor=crest_factor,
        dynamic_range=dynamic_range,
    )


# =============================================================================
# Frequency Analysis
# =============================================================================


def get_peak_frequency(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Find the frequency with the highest magnitude.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT window size.
        min_freq: Minimum frequency to consider (optional).
        max_freq: Maximum frequency to consider (optional).

    Returns:
        Tuple of (peak_frequency_hz, magnitude).
    """
    # Convert to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=-1) if audio.shape[-1] == 2 else audio.flatten()

    # Compute FFT
    fft = np.fft.rfft(audio, n=n_fft)
    magnitude = np.abs(fft)
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

    # Apply frequency range filter
    mask = np.ones(len(frequencies), dtype=bool)
    if min_freq is not None:
        mask &= frequencies >= min_freq
    if max_freq is not None:
        mask &= frequencies <= max_freq

    if not np.any(mask):
        return 0.0, 0.0

    # Find peak
    masked_magnitude = magnitude.copy()
    masked_magnitude[~mask] = 0

    peak_idx = np.argmax(masked_magnitude)
    peak_freq = frequencies[peak_idx]
    peak_mag = magnitude[peak_idx]

    return float(peak_freq), float(peak_mag)


def get_frequency_stats(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    threshold_db: float = -60,
) -> FrequencyStats:
    """
    Calculate comprehensive frequency statistics for audio.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT window size.
        threshold_db: Threshold in dB below peak for significant frequencies.

    Returns:
        :class:`FrequencyStats` with peak, mean, min, max frequencies, etc.
    """
    # Convert to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=-1) if audio.shape[-1] == 2 else audio.flatten()

    # Compute FFT
    fft = np.fft.rfft(audio, n=n_fft)
    magnitude = np.abs(fft)
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

    # Skip DC component
    magnitude = magnitude[1:]
    frequencies = frequencies[1:]

    if len(magnitude) == 0 or np.max(magnitude) == 0:
        return FrequencyStats(
            peak_frequency=0.0,
            peak_magnitude=0.0,
            mean_frequency=0.0,
            min_frequency=0.0,
            max_frequency=0.0,
            bandwidth=0.0,
            spectral_centroid=0.0,
            spectral_rolloff=0.0,
        )

    # Peak frequency
    peak_idx = np.argmax(magnitude)
    peak_freq = frequencies[peak_idx]
    peak_mag = magnitude[peak_idx]

    # Convert to dB for threshold
    magnitude_db = 20 * np.log10(magnitude / np.max(magnitude) + 1e-10)

    # Find significant frequencies (above threshold)
    significant = magnitude_db >= threshold_db
    if np.any(significant):
        sig_freqs = frequencies[significant]
        min_freq = float(np.min(sig_freqs))
        max_freq = float(np.max(sig_freqs))
    else:
        min_freq = 0.0
        max_freq = sample_rate / 2

    # Spectral centroid (weighted mean frequency)
    total_magnitude = np.sum(magnitude)
    if total_magnitude > 0:
        spectral_centroid = float(np.sum(frequencies * magnitude) / total_magnitude)
    else:
        spectral_centroid = 0.0

    # Mean frequency (weighted by power)
    power = magnitude**2
    total_power = np.sum(power)
    if total_power > 0:
        mean_freq = float(np.sum(frequencies * power) / total_power)
    else:
        mean_freq = 0.0

    # Spectral rolloff (frequency below which 85% of energy is contained)
    cumsum = np.cumsum(power)
    rolloff_threshold = 0.85 * total_power
    rolloff_idx = np.searchsorted(cumsum, rolloff_threshold)
    if rolloff_idx < len(frequencies):
        spectral_rolloff = float(frequencies[rolloff_idx])
    else:
        spectral_rolloff = float(frequencies[-1])

    return FrequencyStats(
        peak_frequency=float(peak_freq),
        peak_magnitude=float(peak_mag),
        mean_frequency=mean_freq,
        min_frequency=min_freq,
        max_frequency=max_freq,
        bandwidth=max_freq - min_freq,
        spectral_centroid=spectral_centroid,
        spectral_rolloff=spectral_rolloff,
    )


# =============================================================================
# Silence Detection
# =============================================================================


def detect_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40,
    min_silence_duration: float = 0.1,
    min_sound_duration: float = 0.1,
) -> SilenceInfo:
    """
    Detect silent and non-silent regions in audio.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        threshold_db: Threshold in dBFS below which is considered silence.
        min_silence_duration: Minimum duration in seconds for a silent region.
        min_sound_duration: Minimum duration in seconds for a sound region.

    Returns:
        :class:`SilenceInfo` with silence ratio and segment information.
    """
    # Convert to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=-1) if audio.shape[-1] == 2 else audio.flatten()

    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold_db / 20)

    # Calculate RMS in frames
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)  # 10ms hop

    num_frames = 1 + (len(audio) - frame_length) // hop_length
    if num_frames <= 0:
        # Audio too short
        rms = calculate_rms(audio)
        silent = rms < threshold_linear
        return SilenceInfo(
            is_silent=silent,
            silence_ratio=1.0 if silent else 0.0,
            sound_ratio=0.0 if silent else 1.0,
            silent_segments=[(0.0, len(audio) / sample_rate)] if silent else [],
            sound_segments=[] if silent else [(0.0, len(audio) / sample_rate)],
            threshold_used=threshold_db,
        )

    # Calculate RMS for each frame
    frame_rms = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        frame_rms[i] = np.sqrt(np.mean(frame**2))

    # Determine which frames are silent
    is_silent_frame = frame_rms < threshold_linear

    # Find contiguous regions
    silent_segments = []
    sound_segments = []

    min_silence_frames = int(min_silence_duration * sample_rate / hop_length)
    min_sound_frames = int(min_sound_duration * sample_rate / hop_length)

    # Find runs of silent/non-silent frames
    current_silent = is_silent_frame[0]
    start_frame = 0

    for i in range(1, len(is_silent_frame)):
        if is_silent_frame[i] != current_silent:
            end_frame = i
            start_time = start_frame * hop_length / sample_rate
            end_time = min(end_frame * hop_length / sample_rate, len(audio) / sample_rate)

            if current_silent:
                if end_frame - start_frame >= min_silence_frames:
                    silent_segments.append((start_time, end_time))
            else:
                if end_frame - start_frame >= min_sound_frames:
                    sound_segments.append((start_time, end_time))

            current_silent = is_silent_frame[i]
            start_frame = i

    # Handle last segment
    end_time = len(audio) / sample_rate
    start_time = start_frame * hop_length / sample_rate
    if current_silent:
        if len(is_silent_frame) - start_frame >= min_silence_frames:
            silent_segments.append((start_time, end_time))
    else:
        if len(is_silent_frame) - start_frame >= min_sound_frames:
            sound_segments.append((start_time, end_time))

    # Calculate ratios
    total_duration = len(audio) / sample_rate
    silence_duration = sum(end - start for start, end in silent_segments)
    sound_duration = sum(end - start for start, end in sound_segments)

    silence_ratio = silence_duration / total_duration if total_duration > 0 else 0.0
    sound_ratio = sound_duration / total_duration if total_duration > 0 else 0.0

    # Overall is_silent determination
    overall_rms = calculate_rms(audio)
    silent = overall_rms < threshold_linear or silence_ratio > 0.9

    return SilenceInfo(
        is_silent=silent,
        silence_ratio=silence_ratio,
        sound_ratio=sound_ratio,
        silent_segments=silent_segments,
        sound_segments=sound_segments,
        threshold_used=threshold_db,
    )


def is_silent(
    audio: np.ndarray,
    threshold_db: float = -40,
) -> bool:
    """
    Quick check if audio is mostly silent.

    Args:
        audio: Audio data as numpy array.
        threshold_db: Threshold in dBFS below which is considered silence.

    Returns:
        True if the audio is considered silent.
    """
    rms = calculate_rms(audio)
    threshold_linear = 10 ** (threshold_db / 20)
    return rms < threshold_linear


__all__ = [
    "AmplitudeStats",
    "FrequencyStats",
    "SilenceInfo",
    "calculate_rms",
    "calculate_dbfs",
    "calculate_peak",
    "get_amplitude_stats",
    "get_peak_frequency",
    "get_frequency_stats",
    "detect_silence",
    "is_silent",
]

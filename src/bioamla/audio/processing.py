"""
Audio Signal Processing
=======================

Signal-processing transforms folded from ``core/signal.py``: filtering,
denoising, segmentation, onset detection, normalization, resampling, and
trimming. Functions operate on raw ``numpy`` arrays plus a sample rate.

These are deterministic ``numpy``/``scipy``/``librosa`` routines (all base
dependencies). They raise :class:`~bioamla.exceptions.ProcessingError` on bad
parameters and otherwise return processed arrays.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

from bioamla.exceptions import ProcessingError

# ``librosa`` is imported inside the functions that use it (not at module top)
# so that ``import bioamla`` / CLI startup stays fast — librosa is heavy and this
# module sits on the eager CLI import chain (invariant: keep startup snappy).

# =============================================================================
# Filter Functions
# =============================================================================


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    low_freq: float,
    high_freq: float,
    order: int = 5,
) -> np.ndarray:
    """
    Apply a bandpass filter to audio.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        low_freq: Low cutoff frequency in Hz.
        high_freq: High cutoff frequency in Hz.
        order: Filter order (default: 5).

    Returns:
        Filtered audio as numpy array.

    Raises:
        ProcessingError: If the low cutoff is not below the high cutoff.
    """
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Clamp to valid range
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))

    if low >= high:
        raise ProcessingError(
            f"Low frequency ({low_freq}) must be less than high frequency ({high_freq})"
        )

    b, a = scipy_signal.butter(order, [low, high], btype="band")
    filtered = scipy_signal.filtfilt(b, a, audio)

    return filtered.astype(np.float32)


def lowpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_freq: float,
    order: int = 5,
) -> np.ndarray:
    """
    Apply a lowpass filter to audio.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        cutoff_freq: Cutoff frequency in Hz.
        order: Filter order (default: 5).

    Returns:
        Filtered audio as numpy array.
    """
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    normalized_cutoff = max(0.001, min(normalized_cutoff, 0.999))

    b, a = scipy_signal.butter(order, normalized_cutoff, btype="low")
    filtered = scipy_signal.filtfilt(b, a, audio)

    return filtered.astype(np.float32)


def highpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_freq: float,
    order: int = 5,
) -> np.ndarray:
    """
    Apply a highpass filter to audio.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        cutoff_freq: Cutoff frequency in Hz.
        order: Filter order (default: 5).

    Returns:
        Filtered audio as numpy array.
    """
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    normalized_cutoff = max(0.001, min(normalized_cutoff, 0.999))

    b, a = scipy_signal.butter(order, normalized_cutoff, btype="high")
    filtered = scipy_signal.filtfilt(b, a, audio)

    return filtered.astype(np.float32)


# =============================================================================
# Denoise Functions
# =============================================================================


def spectral_denoise(
    audio: np.ndarray,
    sample_rate: int,
    noise_reduce_factor: float = 1.0,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Apply spectral noise reduction using spectral gating.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        noise_reduce_factor: How aggressively to reduce noise (0-2, default: 1.0).
        n_fft: FFT window size.
        hop_length: Hop length for STFT.

    Returns:
        Denoised audio as numpy array.
    """
    import librosa

    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise floor from quietest frames
    frame_energy = np.mean(magnitude, axis=0)
    noise_frames = frame_energy < np.percentile(frame_energy, 10)

    if np.sum(noise_frames) > 0:
        noise_profile = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
    else:
        noise_profile = np.min(magnitude, axis=1, keepdims=True)

    # Spectral subtraction
    magnitude_denoised = magnitude - noise_reduce_factor * noise_profile
    magnitude_denoised = np.maximum(magnitude_denoised, 0)

    # Reconstruct
    stft_denoised = magnitude_denoised * np.exp(1j * phase)
    audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length)

    return audio_denoised.astype(np.float32)


# =============================================================================
# Segment Functions
# =============================================================================


@dataclass
class AudioSegment:
    """Represents a segment of audio."""

    start_time: float
    end_time: float
    start_sample: int
    end_sample: int


def segment_on_silence(
    audio: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = -40,
    min_silence_duration: float = 0.3,
    min_segment_duration: float = 0.5,
) -> list[AudioSegment]:
    """
    Split audio on silence.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        silence_threshold_db: Threshold in dB below which is considered silence.
        min_silence_duration: Minimum silence duration in seconds to split on.
        min_segment_duration: Minimum segment duration in seconds to keep.

    Returns:
        List of :class:`AudioSegment` objects.
    """
    # Convert to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    import librosa

    # Get intervals of non-silence
    intervals = librosa.effects.split(
        audio,
        top_db=-silence_threshold_db,
        frame_length=2048,
        hop_length=512,
    )

    # Convert to segments
    segments = []
    min_samples = int(min_segment_duration * sample_rate)

    for start_sample, end_sample in intervals:
        if end_sample - start_sample >= min_samples:
            segments.append(
                AudioSegment(
                    start_time=start_sample / sample_rate,
                    end_time=end_sample / sample_rate,
                    start_sample=start_sample,
                    end_sample=end_sample,
                )
            )

    return segments


def split_audio_on_silence(
    audio: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = -40,
    min_silence_duration: float = 0.3,
    min_segment_duration: float = 0.5,
) -> list[tuple[np.ndarray, float, float]]:
    """
    Split audio on silence and return audio chunks.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        silence_threshold_db: Threshold in dB below which is considered silence.
        min_silence_duration: Minimum silence duration in seconds to split on.
        min_segment_duration: Minimum segment duration in seconds to keep.

    Returns:
        List of tuples (audio_chunk, start_time, end_time).
    """
    segments = segment_on_silence(
        audio, sample_rate, silence_threshold_db, min_silence_duration, min_segment_duration
    )

    chunks = []
    for seg in segments:
        chunk = audio[seg.start_sample : seg.end_sample]
        chunks.append((chunk, seg.start_time, seg.end_time))

    return chunks


# =============================================================================
# Event Detection Functions
# =============================================================================


@dataclass
class AudioEvent:
    """Represents a detected audio event."""

    time: float
    strength: float


def detect_onsets(
    audio: np.ndarray,
    sample_rate: int,
    method: str = "energy",
    threshold: float = 0.1,
) -> list[AudioEvent]:
    """
    Detect onset events in audio.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        method: Detection method ('energy', 'spectral', 'complex').
        threshold: Detection threshold (0-1).

    Returns:
        List of :class:`AudioEvent` objects.
    """
    # Convert to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    import librosa

    # Compute onset strength
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)

    # Detect onsets
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sample_rate,
        onset_envelope=onset_env,
        backtrack=True,
    )

    # Convert to times and get strengths
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)

    events = []
    for i, time in enumerate(onset_times):
        if i < len(onset_env):
            strength = float(onset_env[onset_frames[i]] if onset_frames[i] < len(onset_env) else 0)
        else:
            strength = 0.0
        events.append(AudioEvent(time=float(time), strength=strength))

    return events


# =============================================================================
# Normalization Functions
# =============================================================================


def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int,
    target_db: float = -20,
) -> np.ndarray:
    """
    Normalize audio to target loudness level.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        target_db: Target loudness in dB (default: -20).

    Returns:
        Normalized audio as numpy array.
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio**2))

    if rms == 0:
        return audio

    # Convert target dB to linear
    target_linear = 10 ** (target_db / 20)

    # Calculate gain
    gain = target_linear / rms

    # Apply gain
    normalized = audio * gain

    # Clip to prevent clipping
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized.astype(np.float32)


def peak_normalize(
    audio: np.ndarray,
    target_peak: float = 0.95,
) -> np.ndarray:
    """
    Normalize audio to target peak level.

    Args:
        audio: Audio data as numpy array.
        target_peak: Target peak level (0-1, default: 0.95).

    Returns:
        Normalized audio as numpy array.
    """
    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    gain = target_peak / peak
    normalized = audio * gain

    return normalized.astype(np.float32)


# =============================================================================
# Resample Functions
# =============================================================================


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to a different sample rate.

    Args:
        audio: Audio data as numpy array.
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz.

    Returns:
        Resampled audio as numpy array.
    """
    if orig_sr == target_sr:
        return audio

    import librosa

    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    return resampled.astype(np.float32)


# =============================================================================
# Trim Functions
# =============================================================================


def trim_audio(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float | None = None,
    end_time: float | None = None,
) -> np.ndarray:
    """
    Trim audio to specified time range.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        start_time: Start time in seconds (None = beginning).
        end_time: End time in seconds (None = end).

    Returns:
        Trimmed audio as numpy array.

    Raises:
        ProcessingError: If the resulting trim range is empty.
    """
    start_sample = 0 if start_time is None else int(start_time * sample_rate)
    end_sample = len(audio) if end_time is None else int(end_time * sample_rate)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)

    if start_sample >= end_sample:
        raise ProcessingError(f"Invalid trim range: start={start_time}s, end={end_time}s")

    return audio[start_sample:end_sample]


# =============================================================================
# Editing Transforms
# =============================================================================
#
# Deterministic, single-file edits — peers of the filter/normalize ops above.
# These are *editing* operations (apply once with explicit parameters), distinct
# from the randomized pre-training augmentation layer in ``bioamla.datasets``
# (``create_augmentation_pipeline``), which composes these kinds of effects with
# random parameters to synthesize training data.


def pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    n_steps: float,
) -> np.ndarray:
    """
    Shift the pitch of audio up or down without changing its duration.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        n_steps: Number of (fractional) semitones to shift; positive raises pitch.

    Returns:
        Pitch-shifted audio as numpy array.
    """
    import librosa

    shifted = librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=n_steps)
    return shifted.astype(np.float32)


def time_stretch(
    audio: np.ndarray,
    rate: float,
) -> np.ndarray:
    """
    Time-stretch audio without changing its pitch.

    Args:
        audio: Audio data as numpy array.
        rate: Stretch factor; ``> 1`` speeds up (shorter), ``< 1`` slows down.

    Returns:
        Time-stretched audio as numpy array.

    Raises:
        ProcessingError: If ``rate`` is not positive.
    """
    if rate <= 0:
        raise ProcessingError(f"Time-stretch rate must be positive, got {rate}")
    import librosa

    stretched = librosa.effects.time_stretch(y=audio, rate=rate)
    return stretched.astype(np.float32)


def add_noise(
    audio: np.ndarray,
    snr_db: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add Gaussian white noise at a target signal-to-noise ratio.

    Args:
        audio: Audio data as numpy array.
        snr_db: Target SNR in dB; lower values add more noise.
        seed: Optional RNG seed for reproducible noise.

    Returns:
        Noisy audio as numpy array.
    """
    signal_power = np.mean(audio**2)
    if signal_power == 0:
        return audio.astype(np.float32)

    noise_power = signal_power / (10 ** (snr_db / 10))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape)
    return (audio + noise).astype(np.float32)


def apply_gain(
    audio: np.ndarray,
    gain_db: float,
) -> np.ndarray:
    """
    Apply a fixed gain (in dB) to audio, clipping to [-1, 1].

    Args:
        audio: Audio data as numpy array.
        gain_db: Gain to apply in dB; positive amplifies, negative attenuates.

    Returns:
        Gain-adjusted audio as numpy array.
    """
    gained = audio * (10 ** (gain_db / 20))
    return np.clip(gained, -1.0, 1.0).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40,
    margin: float = 0.1,
) -> np.ndarray:
    """
    Trim silence from beginning and end of audio.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        threshold_db: Threshold in dB below which is considered silence.
        margin: Additional margin to keep in seconds.

    Returns:
        Trimmed audio as numpy array.
    """
    # Convert to mono for analysis if needed
    if audio.ndim > 1:
        audio_mono = audio.mean(axis=0)
    else:
        audio_mono = audio

    import librosa

    # Trim silence
    trimmed, index = librosa.effects.trim(audio_mono, top_db=-threshold_db)

    # Add margin
    margin_samples = int(margin * sample_rate)
    start = max(0, index[0] - margin_samples)
    end = min(len(audio), index[1] + margin_samples)

    return audio[start:end]


__all__ = [
    "AudioSegment",
    "AudioEvent",
    "bandpass_filter",
    "lowpass_filter",
    "highpass_filter",
    "spectral_denoise",
    "segment_on_silence",
    "split_audio_on_silence",
    "detect_onsets",
    "normalize_loudness",
    "peak_normalize",
    "resample_audio",
    "trim_audio",
    "trim_silence",
]

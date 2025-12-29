import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bioamla.adapters.pydub import get_audio_info, load_audio

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AudioInfo:
    """
    Basic audio file information.

    Attributes:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        samples: Total number of samples
        bit_depth: Bit depth (if available)
        format: Audio format (e.g., 'WAV', 'FLAC')
        subtype: Audio subtype (e.g., 'PCM_16')
    """

    duration: float
    sample_rate: int
    channels: int
    samples: int
    bit_depth: Optional[int] = None
    format: Optional[str] = None
    subtype: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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


@dataclass
class AudioAnalysis:
    """
    Complete audio analysis results.

    Attributes:
        file_path: Path to the analyzed audio file
        info: Basic audio information
        amplitude: Amplitude statistics
        frequency: Frequency statistics
        silence: Silence detection results
    """

    file_path: str
    info: AudioInfo
    amplitude: AmplitudeStats
    frequency: FrequencyStats
    silence: SilenceInfo

    def to_dict(self) -> Dict[str, Any]:
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

    This function extracts metadata without loading the entire audio into memory.

    Args:
        filepath: Path to the audio file

    Returns:
        AudioInfo object with duration, sample rate, channels, etc.

    Raises:
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> info = get_audio_info("recording.wav")
        >>> print(f"Duration: {info.duration:.2f}s")
        >>> print(f"Sample rate: {info.sample_rate}Hz")
        >>> print(f"Channels: {info.channels}")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    info_dict = get_audio_info(filepath)

    # Extract bit_depth and subtype from pydub response
    # These may be None if ffprobe failed and pydub fallback was used
    bit_depth = info_dict.get("bit_depth")
    subtype = info_dict.get("subtype")

    return AudioInfo(
        duration=info_dict["duration"],
        sample_rate=info_dict["sample_rate"],
        channels=info_dict["channels"],
        samples=info_dict["samples"],
        bit_depth=bit_depth,
        format=info_dict["format"],
        subtype=subtype,
    )


def get_duration(filepath: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        filepath: Path to the audio file

    Returns:
        Duration in seconds
    """
    return get_audio_info(filepath).duration


def get_sample_rate(filepath: str) -> int:
    """
    Get the sample rate of an audio file.

    Args:
        filepath: Path to the audio file

    Returns:
        Sample rate in Hz
    """
    return get_audio_info(filepath).sample_rate


def get_channels(filepath: str) -> int:
    """
    Get the number of channels in an audio file.

    Args:
        filepath: Path to the audio file

    Returns:
        Number of audio channels
    """
    return get_audio_info(filepath).channels


# =============================================================================
# Amplitude Analysis
# =============================================================================


def calculate_rms(audio: np.ndarray) -> float:
    """
    Calculate the Root Mean Square (RMS) amplitude of audio.

    RMS is a measure of the average power of the audio signal.

    Args:
        audio: Audio data as numpy array

    Returns:
        RMS amplitude (0.0 to 1.0 for normalized audio)

    Example:
        >>> audio, sr = load_audio("recording.wav")
        >>> rms = calculate_rms(audio)
        >>> print(f"RMS amplitude: {rms:.4f}")
    """
    # Flatten to mono if multi-channel
    if audio.ndim > 1:
        audio = audio.flatten()

    return float(np.sqrt(np.mean(audio**2)))


def calculate_dbfs(amplitude: float, reference: float = 1.0) -> float:
    """
    Convert linear amplitude to decibels relative to full scale (dBFS).

    Args:
        amplitude: Linear amplitude value
        reference: Reference amplitude for full scale (default: 1.0)

    Returns:
        Amplitude in dBFS (always negative or zero for normalized audio)

    Example:
        >>> dbfs = calculate_dbfs(0.5)  # Half amplitude
        >>> print(f"Level: {dbfs:.1f} dBFS")  # -6.0 dBFS
    """
    if amplitude <= 0:
        return -np.inf

    return float(20 * np.log10(amplitude / reference))


def calculate_peak(audio: np.ndarray) -> float:
    """
    Calculate the peak absolute amplitude of audio.

    Args:
        audio: Audio data as numpy array

    Returns:
        Peak absolute amplitude (0.0 to 1.0+ for normalized audio)
    """
    if audio.ndim > 1:
        audio = audio.flatten()

    return float(np.max(np.abs(audio)))


def get_amplitude_stats(audio: np.ndarray) -> AmplitudeStats:
    """
    Calculate comprehensive amplitude statistics for audio.

    Args:
        audio: Audio data as numpy array

    Returns:
        AmplitudeStats object with RMS, peak, dBFS values, etc.

    Example:
        >>> audio, sr = load_audio("recording.wav")
        >>> stats = get_amplitude_stats(audio)
        >>> print(f"RMS: {stats.rms:.4f} ({stats.rms_db:.1f} dBFS)")
        >>> print(f"Peak: {stats.peak:.4f} ({stats.peak_db:.1f} dBFS)")
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
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        min_freq: Minimum frequency to consider (optional)
        max_freq: Maximum frequency to consider (optional)

    Returns:
        Tuple of (peak_frequency_hz, magnitude)

    Example:
        >>> audio, sr = load_audio("bird_call.wav")
        >>> freq, mag = get_peak_frequency(audio, sr)
        >>> print(f"Peak frequency: {freq:.1f} Hz")
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
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        threshold_db: Threshold in dB below peak for significant frequencies

    Returns:
        FrequencyStats object with peak, mean, min, max frequencies, etc.

    Example:
        >>> audio, sr = load_audio("recording.wav")
        >>> stats = get_frequency_stats(audio, sr)
        >>> print(f"Peak: {stats.peak_frequency:.1f} Hz")
        >>> print(f"Range: {stats.min_frequency:.1f} - {stats.max_frequency:.1f} Hz")
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
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        threshold_db: Threshold in dBFS below which is considered silence
        min_silence_duration: Minimum duration in seconds for a silent region
        min_sound_duration: Minimum duration in seconds for a sound region

    Returns:
        SilenceInfo object with silence ratio and segment information

    Example:
        >>> audio, sr = load_audio("recording.wav")
        >>> silence = detect_silence(audio, sr, threshold_db=-40)
        >>> print(f"Silence ratio: {silence.silence_ratio:.1%}")
        >>> print(f"Sound segments: {len(silence.sound_segments)}")
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
        is_silent = rms < threshold_linear
        return SilenceInfo(
            is_silent=is_silent,
            silence_ratio=1.0 if is_silent else 0.0,
            sound_ratio=0.0 if is_silent else 1.0,
            silent_segments=[(0.0, len(audio) / sample_rate)] if is_silent else [],
            sound_segments=[] if is_silent else [(0.0, len(audio) / sample_rate)],
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
    is_silent = overall_rms < threshold_linear or silence_ratio > 0.9

    return SilenceInfo(
        is_silent=is_silent,
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
        audio: Audio data as numpy array
        threshold_db: Threshold in dBFS below which is considered silence

    Returns:
        True if the audio is considered silent

    Example:
        >>> audio, sr = load_audio("recording.wav")
        >>> if is_silent(audio):
        ...     print("Audio is silent")
    """
    rms = calculate_rms(audio)
    threshold_linear = 10 ** (threshold_db / 20)
    return rms < threshold_linear


# =============================================================================
# Complete Analysis
# =============================================================================


def analyze_audio(
    filepath: str,
    silence_threshold_db: float = -40,
) -> AudioAnalysis:
    """
    Perform complete audio analysis on a file.

    This is the main function that combines all analysis features
    into a single comprehensive result.

    Args:
        filepath: Path to the audio file
        silence_threshold_db: Threshold for silence detection in dBFS

    Returns:
        AudioAnalysis object with all analysis results

    Example:
        >>> analysis = analyze_audio("recording.wav")
        >>> print(f"Duration: {analysis.info.duration:.2f}s")
        >>> print(f"RMS: {analysis.amplitude.rms_db:.1f} dBFS")
        >>> print(f"Peak frequency: {analysis.frequency.peak_frequency:.1f} Hz")
        >>> print(f"Silence ratio: {analysis.silence.silence_ratio:.1%}")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    # Get basic info
    info = get_audio_info(filepath)

    # Load audio
    audio, sr = load_audio(filepath)

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
    filepaths: List[str],
    silence_threshold_db: float = -40,
    verbose: bool = True,
) -> List[AudioAnalysis]:
    """
    Analyze multiple audio files.

    Args:
        filepaths: List of paths to audio files
        silence_threshold_db: Threshold for silence detection in dBFS
        verbose: Print progress

    Returns:
        List of AudioAnalysis objects
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


def summarize_analysis(analyses: List[AudioAnalysis]) -> Dict[str, Any]:
    """
    Summarize analysis results from multiple files.

    Args:
        analyses: List of AudioAnalysis objects

    Returns:
        Dictionary with summary statistics
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

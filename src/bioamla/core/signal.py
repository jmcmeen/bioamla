"""
Signal Processing Utilities
===========================

This module provides audio signal processing functions using scipy and librosa.

Supported operations:
- filter: Bandpass/lowpass/highpass frequency filtering
- denoise: Spectral noise reduction
- segment: Split audio on silence
- detect_events: Onset detection
- normalize: Loudness normalization
- resample: Sample rate conversion
- trim: Trim audio by time
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
from scipy import signal as scipy_signal

from bioamla.adapters.pydub import save_audio as pydub_save_audio
from bioamla.core.torchaudio import load_waveform_tensor

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
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        order: Filter order (default: 5)

    Returns:
        Filtered audio as numpy array
    """
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Clamp to valid range
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))

    if low >= high:
        raise ValueError(
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
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        cutoff_freq: Cutoff frequency in Hz
        order: Filter order (default: 5)

    Returns:
        Filtered audio as numpy array
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
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        cutoff_freq: Cutoff frequency in Hz
        order: Filter order (default: 5)

    Returns:
        Filtered audio as numpy array
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
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        noise_reduce_factor: How aggressively to reduce noise (0-2, default: 1.0)
        n_fft: FFT window size
        hop_length: Hop length for STFT

    Returns:
        Denoised audio as numpy array
    """
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
) -> List[AudioSegment]:
    """
    Split audio on silence.

    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        silence_threshold_db: Threshold in dB below which is considered silence
        min_silence_duration: Minimum silence duration in seconds to split on
        min_segment_duration: Minimum segment duration in seconds to keep

    Returns:
        List of AudioSegment objects
    """
    # Convert to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

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
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Split audio on silence and return audio chunks.

    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        silence_threshold_db: Threshold in dB below which is considered silence
        min_silence_duration: Minimum silence duration in seconds to split on
        min_segment_duration: Minimum segment duration in seconds to keep

    Returns:
        List of tuples (audio_chunk, start_time, end_time)
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
) -> List[AudioEvent]:
    """
    Detect onset events in audio.

    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        method: Detection method ('energy', 'spectral', 'complex')
        threshold: Detection threshold (0-1)

    Returns:
        List of AudioEvent objects
    """
    # Convert to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

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
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        target_db: Target loudness in dB (default: -20)

    Returns:
        Normalized audio as numpy array
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
        audio: Audio data as numpy array
        target_peak: Target peak level (0-1, default: 0.95)

    Returns:
        Normalized audio as numpy array
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
        audio: Audio data as numpy array
        orig_sr: Original sample rate in Hz
        target_sr: Target sample rate in Hz

    Returns:
        Resampled audio as numpy array
    """
    if orig_sr == target_sr:
        return audio

    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    return resampled.astype(np.float32)


# =============================================================================
# Trim Functions
# =============================================================================


def trim_audio(
    audio: np.ndarray,
    sample_rate: int,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> np.ndarray:
    """
    Trim audio to specified time range.

    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        start_time: Start time in seconds (None = beginning)
        end_time: End time in seconds (None = end)

    Returns:
        Trimmed audio as numpy array
    """
    start_sample = 0 if start_time is None else int(start_time * sample_rate)
    end_sample = len(audio) if end_time is None else int(end_time * sample_rate)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)

    if start_sample >= end_sample:
        raise ValueError(f"Invalid trim range: start={start_time}s, end={end_time}s")

    return audio[start_sample:end_sample]


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40,
    margin: float = 0.1,
) -> np.ndarray:
    """
    Trim silence from beginning and end of audio.

    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        threshold_db: Threshold in dB below which is considered silence
        margin: Additional margin to keep in seconds

    Returns:
        Trimmed audio as numpy array
    """
    # Convert to mono for analysis if needed
    if audio.ndim > 1:
        audio_mono = audio.mean(axis=0)
    else:
        audio_mono = audio

    # Trim silence
    trimmed, index = librosa.effects.trim(audio_mono, top_db=-threshold_db)

    # Add margin
    margin_samples = int(margin * sample_rate)
    start = max(0, index[0] - margin_samples)
    end = min(len(audio), index[1] + margin_samples)

    return audio[start:end]


# =============================================================================
# File Processing Functions
# =============================================================================


def load_audio(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return as numpy array.

    Args:
        filepath: Path to audio file

    Returns:
        Tuple of (audio array, sample rate)
    """
    waveform, sr = load_waveform_tensor(filepath)
    audio = waveform.numpy()

    # Convert to mono if stereo
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio = audio.mean(axis=0)
    else:
        audio = audio.squeeze()

    return audio.astype(np.float32), sr


def save_audio(filepath: str, audio: np.ndarray, sample_rate: int) -> str:
    """
    Save audio to file.

    Args:
        filepath: Path to save audio
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz

    Returns:
        Path to saved file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    pydub_save_audio(str(path), audio, sample_rate)
    return str(path)


def process_file(
    input_path: str,
    output_path: str,
    processor_fn,
    sample_rate: Optional[int] = None,
) -> str:
    """
    Process a single audio file.

    Args:
        input_path: Path to input file
        output_path: Path to output file
        processor_fn: Function that takes (audio, sample_rate) and returns processed audio
        sample_rate: Optional target sample rate for output

    Returns:
        Path to output file
    """
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
    processor_fn,
    sample_rate: Optional[int] = None,
    recursive: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Process all audio files in a directory.

    Args:
        input_dir: Path to input directory
        output_dir: Path to output directory
        processor_fn: Function that takes (audio, sample_rate) and returns processed audio
        sample_rate: Optional target sample rate for output
        recursive: Search subdirectories
        verbose: Print progress

    Returns:
        Statistics dict
    """
    from bioamla.core.utils import get_files_by_extension

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    audio_files = get_files_by_extension(
        str(input_dir), extensions=audio_extensions, recursive=recursive
    )

    if not audio_files:
        if verbose:
            print(f"No audio files found in {input_dir}")
        return {"files_processed": 0, "files_failed": 0, "output_dir": str(output_dir)}

    if verbose:
        print(f"Found {len(audio_files)} audio files to process")

    files_processed = 0
    files_failed = 0

    for audio_path in audio_files:
        audio_path = Path(audio_path)

        try:
            rel_path = audio_path.relative_to(input_dir)
        except ValueError:
            rel_path = Path(audio_path.name)

        out_path = output_dir / rel_path.with_suffix(".wav")

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
        "output_dir": str(output_dir),
    }

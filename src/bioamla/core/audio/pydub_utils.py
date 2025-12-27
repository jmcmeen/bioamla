"""
Pydub-based audio file I/O utilities.

This module provides drop-in replacements for soundfile operations using pydub,
enabling support for M4A and other audio formats via ffmpeg.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pydub import AudioSegment


def audiosegment_to_numpy(segment: AudioSegment) -> Tuple[np.ndarray, int]:
    """
    Convert pydub AudioSegment to numpy float32 array.

    Args:
        segment: AudioSegment to convert

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is float32 [-1.0, 1.0]
    """
    # Get raw samples as 16-bit integers
    samples = np.array(segment.get_array_of_samples(), dtype=np.int16)

    # Convert to float32 in range [-1.0, 1.0]
    samples = samples.astype(np.float32) / 32768.0

    # Reshape for multi-channel audio
    if segment.channels > 1:
        samples = samples.reshape((-1, segment.channels))
        # Convert stereo to mono if 2 channels (consistent with soundfile behavior)
        if segment.channels == 2:
            samples = samples.mean(axis=1)

    return samples, segment.frame_rate


def numpy_to_audiosegment(
    audio: np.ndarray, sample_rate: int, channels: Optional[int] = None
) -> AudioSegment:
    """
    Convert numpy float32 array to pydub AudioSegment.

    Args:
        audio: Audio data as numpy array (float32, range [-1.0, 1.0])
        sample_rate: Sample rate in Hz
        channels: Number of channels (auto-detected if None)

    Returns:
        AudioSegment object
    """
    # Infer channels from shape if not provided
    if channels is None:
        channels = 1 if audio.ndim == 1 else audio.shape[1]

    # Ensure proper shape
    if audio.ndim == 2 and channels == 1:
        audio = audio.flatten()

    # Convert float32 [-1.0, 1.0] to int16
    # Clip to prevent overflow
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767.0).astype(np.int16)

    # Create AudioSegment
    segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit = 2 bytes
        channels=channels,
    )
    return segment


def load_audio_pydub(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return as numpy float32 array.

    Drop-in replacement for soundfile.read(filepath, dtype="float32").

    Args:
        filepath: Path to audio file (supports WAV, MP3, FLAC, OGG, M4A, etc.)

    Returns:
        Tuple of (audio_array, sample_rate)

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If file cannot be loaded
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        segment = AudioSegment.from_file(str(path))
        return audiosegment_to_numpy(segment)
    except Exception as e:
        raise Exception(f"Error opening '{filepath}': {e}")


def save_audio_pydub(
    filepath: str,
    audio: np.ndarray,
    sample_rate: int,
    format: Optional[str] = None,
) -> None:
    """
    Save numpy audio array to file.

    Drop-in replacement for soundfile.write(filepath, audio, sample_rate).

    Args:
        filepath: Destination file path
        audio: Audio data as numpy array (float32)
        sample_rate: Sample rate in Hz
        format: Output format (auto-detected from extension if None)

    Raises:
        Exception: If file cannot be saved
    """
    path = Path(filepath)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine format from extension if not specified
    format_to_use = format
    if format_to_use is None:
        ext = path.suffix.lower()
        format_map = {
            ".wav": "wav",
            ".flac": "flac",
            ".ogg": "ogg",
            ".mp3": "mp3",
            ".m4a": "ipod",  # pydub uses "ipod" for M4A/AAC
        }
        format_to_use = format_map.get(ext, "wav")

    try:
        # Infer channels
        channels = 1 if audio.ndim == 1 else audio.shape[1]

        # Convert to AudioSegment
        segment = numpy_to_audiosegment(audio, sample_rate, channels)

        # Export to file
        segment.export(str(path), format=format_to_use)
    except Exception as e:
        raise Exception(f"Failed to save audio to '{filepath}': {e}")


def get_audio_info_pydub(filepath: str) -> dict:
    """
    Get audio file metadata without loading full audio data.

    Drop-in replacement for soundfile.info() / soundfile.SoundFile().

    Args:
        filepath: Path to audio file

    Returns:
        Dictionary with keys: duration, sample_rate, channels, samples, format

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If metadata cannot be extracted
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        segment = AudioSegment.from_file(str(path))

        duration_sec = len(segment) / 1000.0  # pydub uses milliseconds
        sample_rate = segment.frame_rate
        channels = segment.channels
        samples = int(duration_sec * sample_rate)

        return {
            "duration": duration_sec,
            "sample_rate": sample_rate,
            "channels": channels,
            "samples": samples,
            "format": path.suffix.upper().lstrip("."),
        }
    except Exception as e:
        raise Exception(f"Failed to get audio info from '{filepath}': {e}")

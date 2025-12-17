"""
Audio File Operations
=====================

Utility functions for audio file operations including loading, saving,
metadata extraction, format conversion, and audio manipulation.

Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC, WMA, AIFF, OPUS
"""

import logging
import wave
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from bioamla.utils.files import get_files_by_extension

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_AUDIO_EXTENSIONS = [
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".opus"
]


# =============================================================================
# Audio File Discovery
# =============================================================================

def get_audio_files(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> List[str]:
    """
    Get a list of audio files in a directory.

    Args:
        directory: Path to the directory to search
        extensions: List of audio file extensions to include.
            If None, uses SUPPORTED_AUDIO_EXTENSIONS
        recursive: If True, search subdirectories recursively

    Returns:
        List of audio file paths, sorted alphabetically
    """
    if extensions is None:
        extensions = SUPPORTED_AUDIO_EXTENSIONS
    return get_files_by_extension(directory, extensions, recursive)


# =============================================================================
# Audio Loading and Saving
# =============================================================================

def load_audio(
    filepath: str,
    sample_rate: Optional[int] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using soundfile/librosa.

    Supports WAV, FLAC, MP3, OGG and other formats via soundfile and librosa.

    Args:
        filepath: Path to the audio file
        sample_rate: Target sample rate. If None, uses the file's native rate.
        mono: If True, convert multi-channel audio to mono

    Returns:
        Tuple of (audio_data, sample_rate) where audio_data is a numpy array
        with shape (samples,) for mono or (channels, samples) for multi-channel
    """
    import soundfile as sf

    try:
        audio, sr = sf.read(filepath, dtype='float32')
    except Exception:
        # Fall back to librosa for formats soundfile can't handle
        import librosa
        audio, sr = librosa.load(filepath, sr=sample_rate, mono=mono)
        return audio, sr

    # Handle multi-channel audio
    if audio.ndim > 1:
        # soundfile returns (samples, channels), transpose to (channels, samples)
        audio = audio.T
        if mono:
            audio = to_mono(audio)

    # Resample if needed
    if sample_rate is not None and sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    return audio, sr


def save_audio(filepath: str, audio: np.ndarray, sample_rate: int) -> str:
    """
    Save audio data to a file.

    The output format is determined by the file extension.

    Args:
        filepath: Path to save the audio file
        audio: Audio data as numpy array, shape (samples,) or (channels, samples)
        sample_rate: Sample rate in Hz

    Returns:
        Path to the saved file
    """
    import soundfile as sf

    # Transpose if multi-channel (soundfile expects samples, channels)
    if audio.ndim > 1 and audio.shape[0] < audio.shape[1]:
        audio = audio.T

    sf.write(filepath, audio, sample_rate)
    return filepath


# =============================================================================
# Audio Metadata
# =============================================================================

def get_wav_metadata(filepath: str) -> dict:
    """
    Get metadata from a WAV file.

    Args:
        filepath: Path to the WAV file

    Returns:
        Dictionary containing WAV file metadata:
            - channels: Number of audio channels
            - sample_width: Sample width in bytes
            - sample_rate: Sample rate in Hz
            - num_frames: Total number of frames
            - duration: Duration in seconds
            - compression_type: Compression type
            - compression_name: Compression name
    """
    with wave.open(filepath, 'rb') as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        compression_type = wav_file.getcomptype()
        compression_name = wav_file.getcompname()

        duration = num_frames / sample_rate if sample_rate > 0 else 0

        return {
            'channels': channels,
            'sample_width': sample_width,
            'sample_rate': sample_rate,
            'num_frames': num_frames,
            'duration': duration,
            'compression_type': compression_type,
            'compression_name': compression_name
        }


# =============================================================================
# Audio Manipulation
# =============================================================================

def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert multi-channel audio to mono by averaging channels.

    Args:
        audio: Audio data with shape (channels, samples) or (samples,)

    Returns:
        Mono audio with shape (samples,)
    """
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[0] > audio.shape[1]:
            # Shape is (samples, channels), transpose first
            audio = audio.T
        return np.mean(audio, axis=0)
    raise ValueError(f"Unexpected audio shape: {audio.shape}")


def concatenate_audio(
    audio_list: List[np.ndarray],
    crossfade_samples: int = 0
) -> np.ndarray:
    """
    Concatenate multiple audio arrays into one.

    Args:
        audio_list: List of audio arrays to concatenate
        crossfade_samples: Number of samples to crossfade between clips.
            If 0, clips are concatenated without crossfade.

    Returns:
        Concatenated audio array
    """
    if not audio_list:
        raise ValueError("audio_list cannot be empty")

    if len(audio_list) == 1:
        return audio_list[0]

    if crossfade_samples <= 0:
        return np.concatenate(audio_list)

    # Apply crossfade between clips
    result = audio_list[0].copy()
    for next_audio in audio_list[1:]:
        # Ensure crossfade doesn't exceed either clip's length
        fade_len = min(crossfade_samples, len(result), len(next_audio))

        if fade_len > 0:
            # Create fade curves
            fade_out = np.linspace(1.0, 0.0, fade_len)
            fade_in = np.linspace(0.0, 1.0, fade_len)

            # Apply crossfade
            result[-fade_len:] = result[-fade_len:] * fade_out + next_audio[:fade_len] * fade_in

            # Append the rest of next_audio
            result = np.concatenate([result, next_audio[fade_len:]])
        else:
            result = np.concatenate([result, next_audio])

    return result


def loop_audio(
    audio: np.ndarray,
    num_loops: int,
    crossfade_samples: int = 0
) -> np.ndarray:
    """
    Loop audio a specified number of times.

    Args:
        audio: Audio data to loop
        num_loops: Number of times to repeat the audio (1 = no repetition)
        crossfade_samples: Number of samples to crossfade between loops.
            If 0, loops are concatenated without crossfade.

    Returns:
        Looped audio array
    """
    if num_loops < 1:
        raise ValueError("num_loops must be at least 1")

    if num_loops == 1:
        return audio

    audio_list = [audio] * num_loops
    return concatenate_audio(audio_list, crossfade_samples)


# =============================================================================
# Audio Format Conversion
# =============================================================================

def _convert_audio_pydub(input_path: str, output_path: str, output_format: str) -> str:
    """
    Convert audio file using pydub.

    Args:
        input_path: Path to input audio file
        output_path: Path for output file
        output_format: Target format (e.g., 'wav', 'mp3')

    Returns:
        Path to the converted file
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format=output_format)
    return output_path


def wav_to_mp3(input_path: str, output_path: str) -> str:
    """Convert WAV to MP3."""
    return _convert_audio_pydub(input_path, output_path, 'mp3')


def mp3_to_wav(input_path: str, output_path: str) -> str:
    """Convert MP3 to WAV."""
    return _convert_audio_pydub(input_path, output_path, 'wav')


def wav_to_flac(input_path: str, output_path: str) -> str:
    """Convert WAV to FLAC."""
    return _convert_audio_pydub(input_path, output_path, 'flac')


def flac_to_wav(input_path: str, output_path: str) -> str:
    """Convert FLAC to WAV."""
    return _convert_audio_pydub(input_path, output_path, 'wav')


def wav_to_ogg(input_path: str, output_path: str) -> str:
    """Convert WAV to OGG."""
    return _convert_audio_pydub(input_path, output_path, 'ogg')


def ogg_to_wav(input_path: str, output_path: str) -> str:
    """Convert OGG to WAV."""
    return _convert_audio_pydub(input_path, output_path, 'wav')


def m4a_to_wav(input_path: str, output_path: str) -> str:
    """Convert M4A to WAV."""
    return _convert_audio_pydub(input_path, output_path, 'wav')


def wav_to_m4a(input_path: str, output_path: str) -> str:
    """Convert WAV to M4A."""
    return _convert_audio_pydub(input_path, output_path, 'ipod')


def mp3_to_flac(input_path: str, output_path: str) -> str:
    """Convert MP3 to FLAC."""
    return _convert_audio_pydub(input_path, output_path, 'flac')


def flac_to_mp3(input_path: str, output_path: str) -> str:
    """Convert FLAC to MP3."""
    return _convert_audio_pydub(input_path, output_path, 'mp3')


def m4a_to_mp3(input_path: str, output_path: str) -> str:
    """Convert M4A to MP3."""
    return _convert_audio_pydub(input_path, output_path, 'mp3')


def mp3_to_ogg(input_path: str, output_path: str) -> str:
    """Convert MP3 to OGG."""
    return _convert_audio_pydub(input_path, output_path, 'ogg')


def ogg_to_mp3(input_path: str, output_path: str) -> str:
    """Convert OGG to MP3."""
    return _convert_audio_pydub(input_path, output_path, 'mp3')


def flac_to_ogg(input_path: str, output_path: str) -> str:
    """Convert FLAC to OGG."""
    return _convert_audio_pydub(input_path, output_path, 'ogg')


def ogg_to_flac(input_path: str, output_path: str) -> str:
    """Convert OGG to FLAC."""
    return _convert_audio_pydub(input_path, output_path, 'flac')


def m4a_to_flac(input_path: str, output_path: str) -> str:
    """Convert M4A to FLAC."""
    return _convert_audio_pydub(input_path, output_path, 'flac')


def m4a_to_ogg(input_path: str, output_path: str) -> str:
    """Convert M4A to OGG."""
    return _convert_audio_pydub(input_path, output_path, 'ogg')


def aac_to_wav(input_path: str, output_path: str) -> str:
    """Convert AAC to WAV."""
    return _convert_audio_pydub(input_path, output_path, 'wav')


def aac_to_mp3(input_path: str, output_path: str) -> str:
    """Convert AAC to MP3."""
    return _convert_audio_pydub(input_path, output_path, 'mp3')


def wma_to_wav(input_path: str, output_path: str) -> str:
    """Convert WMA to WAV."""
    return _convert_audio_pydub(input_path, output_path, 'wav')


def wma_to_mp3(input_path: str, output_path: str) -> str:
    """Convert WMA to MP3."""
    return _convert_audio_pydub(input_path, output_path, 'mp3')

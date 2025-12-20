"""
TorchAudio Utilities
===================

This module provides helper functions for working with audio files using PyTorch's
torchaudio library. It includes functionality for loading, resampling, splitting,
and processing audio data for machine learning applications.

These utilities form the core audio processing pipeline used throughout the
bioamla package for preparing audio data for model training and inference.
"""

import io

import torch
import torchaudio
from torchaudio.transforms import Resample

from bioamla.core.ml.config import DefaultConfig
from bioamla.core.utils import get_files_by_extension


def get_wav_info(filepath: str):
    """
    Get metadata information about a WAV audio file.

    Args:
        filepath (str): Path to the audio file

    Returns:
        AudioMetaData: TorchAudio metadata object containing sample rate,
                      number of frames, channels, etc.
    """
    return torchaudio.info(filepath)


def get_wav_files(directory: str) -> list:
    """
    Get a list of all WAV files in a directory.

    Args:
        directory (str): Path to the directory to search

    Returns:
        list: List of file paths for all WAV files found
    """
    return get_files_by_extension(directory, [".wav"])


def get_wavefile_shape(wavefile_path: str):
    """
    Get the shape (dimensions) of an audio waveform.

    Args:
        wavefile_path (str): Path to the audio file

    Returns:
        torch.Size: Shape of the waveform tensor (channels, samples)
    """
    waveform, _ = torchaudio.load(wavefile_path)
    return waveform.shape


def get_wavefile_sample_rate(wavefile_path: str):
    """
    Get the sample rate of an audio file.

    Args:
        wavefile_path (str): Path to the audio file

    Returns:
        int: Sample rate in Hz
    """
    _, sample_rate = torchaudio.load(wavefile_path)
    return sample_rate


def load_waveform_tensor(filepath: str):
    """
    Load an audio file as a waveform tensor.

    Args:
        filepath (str): Path to the audio file

    Returns:
        tuple: (waveform tensor, sample rate)
            - waveform (torch.Tensor): Audio waveform data
            - sample_rate (int): Sample rate in Hz
    """
    waveform, sample_rate = torchaudio.load(filepath)
    return (waveform, sample_rate)


def split_waveform_tensor(
    waveform_tensor: torch.Tensor, freq: int, segment_duration: int, segment_overlap: int
):
    """
    Split a waveform tensor into overlapping segments.

    This function divides a long audio waveform into smaller, potentially overlapping
    segments for processing. This is useful for handling long recordings or when
    training models on fixed-length segments.

    Args:
        waveform_tensor (torch.Tensor): Input waveform tensor
        freq (int): Sample rate of the audio
        segment_duration (int): Duration of each segment in seconds
        segment_overlap (int): Overlap between consecutive segments in seconds

    Returns:
        list: List of tuples containing (segment_tensor, start_sample, end_sample)
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


def resample_waveform_tensor(waveform_tensor: torch.Tensor, orig_freq: int, new_freq: int):
    """
    Resample a waveform tensor to a different sample rate.

    Args:
        waveform_tensor (torch.Tensor): Input waveform tensor
        orig_freq (int): Original sample rate in Hz
        new_freq (int): Target sample rate in Hz

    Returns:
        torch.Tensor: Resampled waveform tensor
    """
    resampler = Resample(orig_freq=orig_freq, new_freq=new_freq)
    waveform_tensor = resampler(waveform_tensor)
    return waveform_tensor


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = DefaultConfig.SAMPLE_RATE):
    """
    Load audio data from bytes and preprocess for model input.

    This function takes raw audio bytes (e.g., from an uploaded file or API request),
    loads it as audio data, converts to mono, resamples to target sample rate,
    and returns as a numpy array ready for model processing.

    Args:
        audio_bytes (bytes): Raw audio file data as bytes
        target_sr (int): Target sample rate for output (default from config)

    Returns:
        tuple: (audio_array, sample_rate)
            - audio_array (numpy.ndarray): Preprocessed audio as 1D array
            - sample_rate (int): Sample rate of the output audio

    Raises:
        ValueError: If audio bytes cannot be processed
    """
    try:
        # Create a file-like object from bytes
        audio_io = io.BytesIO(audio_bytes)

        # Load audio with torchaudio
        waveform, sample_rate = torchaudio.load(audio_io)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)

        # Flatten to 1D array
        audio_array = waveform.squeeze().numpy()

        return audio_array, target_sr
    except Exception as e:
        raise ValueError("Could not process audio bytes") from e

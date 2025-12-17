"""
Utility Functions
=================

This module provides utility functions for file operations, audio processing,
and compression. These functions were extracted from novus-pytils to remove
the external dependency.
"""

import logging
import os
import shutil
import struct
import wave
import zipfile
from pathlib import Path
from typing import List, Optional, Union
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


# =============================================================================
# Global Constants
# =============================================================================

SUPPORTED_AUDIO_EXTENSIONS = [
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".opus"
]


# =============================================================================
# File Operations
# =============================================================================

def get_files_by_extension(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> List[str]:
    """
    Get a list of files in a directory filtered by extension.

    Args:
        directory: Path to the directory to search
        extensions: List of file extensions to include (e.g., ['.wav', '.mp3']).
            If None, returns all files.
        recursive: If True, search subdirectories recursively

    Returns:
        List of file paths matching the criteria
    """
    if extensions is not None:
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                      for ext in extensions]

    files = []
    directory_path = Path(directory)

    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return files

    if recursive:
        for filepath in directory_path.rglob('*'):
            if filepath.is_file():
                if extensions is None or filepath.suffix.lower() in extensions:
                    files.append(str(filepath))
    else:
        for filepath in directory_path.iterdir():
            if filepath.is_file():
                if extensions is None or filepath.suffix.lower() in extensions:
                    files.append(str(filepath))

    return sorted(files)


def create_directory(path: str) -> str:
    """
    Create a directory and all parent directories if they don't exist.

    Args:
        path: Path to the directory to create

    Returns:
        The path that was created
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def file_exists(path: str) -> bool:
    """
    Check if a file exists.

    Args:
        path: Path to check

    Returns:
        True if the file exists, False otherwise
    """
    return Path(path).is_file()


def directory_exists(path: str) -> bool:
    """
    Check if a directory exists.

    Args:
        path: Path to check

    Returns:
        True if the directory exists, False otherwise
    """
    return Path(path).is_dir()


def download_file(url: str, output_path: str, show_progress: bool = True) -> str:
    """
    Download a file from a URL.

    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        show_progress: If True, print download progress

    Returns:
        Path to the downloaded file
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        create_directory(output_dir)

    if show_progress:
        print(f"Downloading {url} to {output_path}")

    urlretrieve(url, output_path)

    if show_progress:
        print(f"Download complete: {output_path}")

    return output_path


# =============================================================================
# Audio File Operations
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
        List of audio file paths
    """
    if extensions is None:
        extensions = SUPPORTED_AUDIO_EXTENSIONS
    return get_files_by_extension(directory, extensions, recursive)


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


# =============================================================================
# Compression Operations
# =============================================================================

def extract_zip_file(zip_path: str, output_dir: str) -> str:
    """
    Extract a ZIP archive to a directory.

    Args:
        zip_path: Path to the ZIP file
        output_dir: Directory to extract to

    Returns:
        Path to the output directory
    """
    create_directory(output_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    return output_dir


def create_zip_file(file_paths: List[str], output_path: str) -> str:
    """
    Create a ZIP archive from a list of files.

    Args:
        file_paths: List of file paths to include in the archive
        output_path: Path for the output ZIP file

    Returns:
        Path to the created ZIP file
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        create_directory(output_dir)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for filepath in file_paths:
            arcname = os.path.basename(filepath)
            zip_ref.write(filepath, arcname)

    return output_path


def zip_directory(directory: str, output_path: str) -> str:
    """
    Create a ZIP archive from a directory.

    Args:
        directory: Path to the directory to archive
        output_path: Path for the output ZIP file

    Returns:
        Path to the created ZIP file
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        create_directory(output_dir)

    directory_path = Path(directory)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for filepath in directory_path.rglob('*'):
            if filepath.is_file():
                arcname = filepath.relative_to(directory_path)
                zip_ref.write(filepath, arcname)

    return output_path

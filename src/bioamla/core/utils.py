"""
Utility Functions
=================

This module provides commonly used utility functions throughout the bioamla package.
It acts as a facade re-exporting utilities from various subpackages for convenience.

These utilities are re-exported from specialized packages:
- File operations from bioamla.files
- Audio file discovery from specialized functions
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Re-export from files package
from bioamla.core.files import (
    create_directory,
    directory_exists,
    download_file,
    file_exists,
    get_files_by_extension,
)

# Re-export from constants (single source of truth for constants)
from bioamla.core.constants import SUPPORTED_AUDIO_EXTENSIONS


def get_audio_files(
    directory: Union[str, Path], extensions: Optional[List[str]] = None, recursive: bool = True
) -> List[str]:
    """
    Get a list of audio files in a directory.

    Args:
        directory: Path to the directory to search
        extensions: List of audio file extensions to include.
            Defaults to SUPPORTED_AUDIO_EXTENSIONS if None.
        recursive: If True, search subdirectories recursively

    Returns:
        List of audio file paths matching the criteria
    """
    if extensions is None:
        extensions = SUPPORTED_AUDIO_EXTENSIONS
    return get_files_by_extension(directory, extensions, recursive)


def get_wav_metadata(filepath: str) -> Dict[str, Any]:
    """
    Get metadata from an audio file (WAV, MP3, FLAC, OGG, M4A, etc.).

    Args:
        filepath: Path to the audio file

    Returns:
        Dictionary with audio metadata (sample_rate, channels, duration, etc.)
    """
    from bioamla.adapters.pydub import get_audio_info

    info = get_audio_info(filepath)
    return {
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "frames": info["samples"],  # pydub calls this "samples"
        "duration": info["duration"],
        "format": info["format"],
        "subtype": info.get("subtype"),  # Now available via ffprobe
        "bit_depth": info.get("bit_depth"),  # Now available via ffprobe
    }


def extract_zip_file(zip_path: Union[str, Path], extract_to: Union[str, Path]) -> List[str]:
    """
    Extract a ZIP file to a directory.

    Args:
        zip_path: Path to the ZIP file
        extract_to: Directory to extract files to

    Returns:
        List of extracted file paths
    """
    import zipfile

    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    extracted_files = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
        extracted_files = [str(extract_to / name) for name in zf.namelist()]

    return extracted_files


def create_zip_file(files: List[Union[str, Path]], zip_path: Union[str, Path]) -> str:
    """
    Create a ZIP file from a list of files.

    Args:
        files: List of file paths to include in the ZIP
        zip_path: Path for the output ZIP file

    Returns:
        Path to the created ZIP file
    """
    import zipfile

    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            file = Path(file)
            zf.write(file, file.name)

    return str(zip_path)


def zip_directory(directory: Union[str, Path], zip_path: Union[str, Path]) -> str:
    """
    Create a ZIP file from a directory.

    Args:
        directory: Path to the directory to zip
        zip_path: Path for the output ZIP file

    Returns:
        Path to the created ZIP file
    """
    import zipfile

    directory = Path(directory)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in directory.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(directory)
                zf.write(file, arcname)

    return str(zip_path)


__all__ = [
    # Constants
    "SUPPORTED_AUDIO_EXTENSIONS",
    # File operations
    "get_files_by_extension",
    "file_exists",
    "directory_exists",
    "create_directory",
    "download_file",
    # Audio utilities
    "get_audio_files",
    "get_wav_metadata",
    # Archive utilities
    "extract_zip_file",
    "create_zip_file",
    "zip_directory",
]

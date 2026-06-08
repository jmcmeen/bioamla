"""
Audio File Discovery
====================

Helpers for discovering audio files on disk and reading lightweight audio
metadata. Stdlib-only at import time; pydub is imported lazily where needed
for metadata extraction.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bioamla.common.constants import SUPPORTED_AUDIO_EXTENSIONS
from bioamla.common.files import get_files_by_extension


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
        List of audio file paths (strings) matching the criteria.
    """
    if extensions is None:
        extensions = SUPPORTED_AUDIO_EXTENSIONS
    return get_files_by_extension(directory, extensions, recursive)


def list_audio_files(
    directory: Union[str, Path], recursive: bool = True
) -> List[Path]:
    """
    List audio files in a directory as :class:`~pathlib.Path` objects.

    Filters by :data:`SUPPORTED_AUDIO_EXTENSIONS`.

    Args:
        directory: Directory to search.
        recursive: If True, search subdirectories recursively.

    Returns:
        Sorted list of audio file paths.
    """
    return [Path(p) for p in get_audio_files(directory, recursive=recursive)]


def get_wav_metadata(filepath: str) -> Dict[str, Any]:
    """
    Get metadata from an audio file (WAV, MP3, FLAC, OGG, M4A, etc.).

    Args:
        filepath: Path to the audio file

    Returns:
        Dictionary with audio metadata (sample_rate, channels, frames, duration,
        format, subtype, bit_depth).

    Raises:
        DependencyError: If the pydub adapter / ffmpeg backend is unavailable.
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


__all__ = [
    "get_audio_files",
    "list_audio_files",
    "get_wav_metadata",
]

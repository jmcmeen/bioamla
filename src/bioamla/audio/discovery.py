"""
Audio File Discovery
====================

Helpers for discovering audio files on disk and reading lightweight audio
metadata. Stdlib-only at import time; pydub is imported lazily where needed
for metadata extraction.
"""

from pathlib import Path
from typing import Any

from bioamla.common.constants import SUPPORTED_AUDIO_EXTENSIONS
from bioamla.common.files import get_files_by_extension
from bioamla.exceptions import NotFoundError


def get_audio_files(
    directory: str | Path, extensions: list[str] | None = None, recursive: bool = True
) -> list[str]:
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


def list_audio_files(directory: str | Path, recursive: bool = True) -> list[Path]:
    """
    List audio files in a directory as :class:`~pathlib.Path` objects.

    Filters by :data:`SUPPORTED_AUDIO_EXTENSIONS`.

    Args:
        directory: Directory to search.
        recursive: If True, search subdirectories recursively.

    Returns:
        Sorted list of audio file paths.

    Raises:
        NotFoundError: If ``directory`` doesn't exist or isn't a folder.
    """
    if not Path(directory).is_dir():
        raise NotFoundError(f"Directory not found: {directory}")
    return [Path(p) for p in get_audio_files(directory, recursive=recursive)]


def get_wav_metadata(filepath: str) -> dict[str, Any]:
    """
    Get metadata from an audio file (WAV, MP3, FLAC, OGG, M4A, etc.).

    Args:
        filepath: Path to the audio file

    Returns:
        Dictionary with audio metadata (sample_rate, channels, frames, duration,
        format, subtype, bit_depth).

    """
    from bioamla.audio.info import read_audio_info_dict

    info = read_audio_info_dict(filepath)
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

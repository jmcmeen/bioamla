"""
Path Utilities
==============

Path manipulation, sanitization, and directory utilities.
"""

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename or directory name.

    Converts the name to lowercase, replaces spaces with underscores,
    removes invalid characters, and ensures a valid result.

    Args:
        name: The string to sanitize

    Returns:
        A sanitized string safe for use as a filename.
        Returns "unknown" if the result would be empty.

    Examples:
        >>> sanitize_filename("My Species Name")
        'my_species_name'
        >>> sanitize_filename("Test: File?")
        'test__file_'
        >>> sanitize_filename("")
        'unknown'
    """
    if not name:
        return "unknown"

    invalid_chars = '<>:"/\\|?*'
    sanitized = name.lower()
    sanitized = sanitized.replace(" ", "_")

    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")

    sanitized = sanitized.strip(". ")

    return sanitized if sanitized else "unknown"


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory

    Returns:
        The path to the directory as a Path object

    Note:
        Creates parent directories as needed.
    """
    path = Path(path) if isinstance(path, str) else path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_relative_path(filepath: Path, base_path: Path) -> str:
    """
    Get the relative path of a file from a base directory.

    Args:
        filepath: Absolute path to the file
        base_path: Base directory path

    Returns:
        Relative path as a string

    Note:
        Falls back to the filename if the file is not under the base path.
    """
    try:
        return str(filepath.relative_to(base_path))
    except ValueError:
        return filepath.name

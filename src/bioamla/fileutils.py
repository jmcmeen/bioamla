"""
File Utilities
==============

This module provides common file-related utility functions used throughout
the bioamla package. It consolidates file name sanitization, extension handling,
and other file-related operations.
"""

import logging
from pathlib import Path

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


def get_extension_from_url(url: str) -> str:
    """
    Extract file extension from a URL.

    Checks for common audio file extensions in the URL and returns
    the appropriate extension. Falls back to .mp3 if no extension is found.

    Args:
        url: The URL to extract the extension from

    Returns:
        The file extension including the leading dot (e.g., ".wav")
    """
    url_lower = url.lower()

    # Check for common audio extensions
    extension_map = [
        (".wav", ".wav"),
        (".m4a", ".m4a"),
        (".mp3", ".mp3"),
        (".ogg", ".ogg"),
        (".flac", ".flac"),
    ]

    for pattern, ext in extension_map:
        if pattern in url_lower:
            return ext

    return ".mp3"  # Default fallback


def get_extension_from_content_type(content_type: str) -> str:
    """
    Map HTTP Content-Type header to file extension.

    Args:
        content_type: The Content-Type header value

    Returns:
        The corresponding file extension (including dot), or empty string if unknown
    """
    content_type = content_type.lower().split(";")[0].strip()

    mapping = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/m4a": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/mp4": ".m4a",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "audio/x-flac": ".flac",
    }

    return mapping.get(content_type, "")


def find_species_name(category: str, all_categories: set) -> str:
    """
    Find the species name for a given category.

    If the category is a subspecies (e.g., "Lithobates sphenocephalus utricularius"),
    this will return the matching species name (e.g., "Lithobates sphenocephalus")
    if it exists in the set of all categories.

    Args:
        category: The category name to check
        all_categories: Set of all known category names

    Returns:
        The shortest matching species name, or the original category if no match
    """
    if not category:
        return category

    # Find all categories that are prefixes of this category
    matching_species = [
        c for c in all_categories
        if category.startswith(c) and c != category
    ]

    if matching_species:
        # Return the shortest matching species (most general)
        return min(matching_species, key=len)

    return category


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory

    Returns:
        The path to the directory

    Note:
        Creates parent directories as needed.
    """
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

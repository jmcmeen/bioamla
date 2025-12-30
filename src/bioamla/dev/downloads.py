"""
Download Utilities
==================

File download and URL handling utilities.
"""

import logging
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve

from bioamla.core.paths import ensure_directory

logger = logging.getLogger(__name__)


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

    return ".mp3"


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


def download_file(url: str, output_path: Union[str, Path], show_progress: bool = True) -> Path:
    """
    Download a file from a URL.

    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        show_progress: If True, print download progress

    Returns:
        Path to the downloaded file
    """
    output_path = Path(output_path) if isinstance(output_path, str) else output_path

    if output_path.parent and str(output_path.parent) != ".":
        ensure_directory(output_path.parent)

    if show_progress:
        print(f"Downloading {url} to {output_path}")

    urlretrieve(url, output_path)

    if show_progress:
        print(f"Download complete: {output_path}")

    return output_path

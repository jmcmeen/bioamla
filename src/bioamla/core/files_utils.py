"""
File System Operations
======================

Utility functions for file system operations including directory management,
file existence checks, and file downloads.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


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
        List of file paths matching the criteria, sorted alphabetically
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

"""
Compression Operations
======================

Utility functions for ZIP archive operations including extraction,
creation, and directory archiving.
"""

import os
import zipfile
from pathlib import Path
from typing import List

from bioamla.core.files_utils import create_directory


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

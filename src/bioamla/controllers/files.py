"""
File management controller for the bioamla package.

This module provides file operation utilities for the bioamla package,
including file downloading and archive extraction functionality. It serves
as a wrapper around novus_pytils file utilities, providing bioamla-specific
interfaces for common file management tasks.

The module is particularly useful for downloading model files, datasets,
and other resources required by bioamla's audio processing pipeline, as well
as handling compressed archives containing such resources.
"""

from typing import Optional
from novus_pytils.files import download_file, extract_zip_file


def download_file_from_url(url: str, output_path: str) -> None:
    """
    Download a file from a URL to a local file path.
    
    Downloads a file from the specified URL and saves it to the given local
    output path. This function serves as a bioamla-specific wrapper around
    the novus_pytils download functionality, ensuring consistent behavior
    across the bioamla ecosystem.
    
    The function handles HTTP/HTTPS URLs and supports various file types
    commonly used in audio processing workflows, including model files,
    datasets, and configuration files.
    
    Args:
        url (str): The complete URL of the file to download. Must be a valid
                  HTTP or HTTPS URL pointing to an accessible resource.
        output_path (str): The local file system path where the downloaded
                          file should be saved. If the directory doesn't
                          exist, it will be created automatically.
    
    Raises:
        requests.exceptions.RequestException: If the download request fails
                                            due to network issues, invalid URL,
                                            or server errors.
        OSError: If there are file system issues such as insufficient
                permissions, disk space, or invalid path.
        FileNotFoundError: If the parent directory cannot be created or
                          is inaccessible.
    
    Example:
        >>> download_file_from_url(
        ...     "https://example.com/model.pth",
        ...     "/local/path/model.pth"
        ... )
        # File downloaded successfully to /local/path/model.pth
    
    Note:
        Large files may take significant time to download. Consider
        implementing progress tracking for user-facing applications.
    """
    download_file(url, output_path)

def extract_zip_file_to_directory(zip_file_path: str, output_dir: str) -> None:
    """
    Extract a ZIP archive to a specified directory.
    
    Extracts all contents of a ZIP file to the specified output directory,
    preserving the internal directory structure of the archive. This function
    is commonly used to extract downloaded datasets, model archives, or
    other compressed resources used in bioamla's audio processing workflows.
    
    The function automatically creates the output directory if it doesn't
    exist and handles nested directory structures within the ZIP archive.
    
    Args:
        zip_file_path (str): The absolute or relative path to the ZIP file
                           that should be extracted. The file must exist
                           and be a valid ZIP archive.
        output_dir (str): The directory path where the ZIP file contents
                         will be extracted. The directory will be created
                         if it doesn't exist.
    
    Raises:
        FileNotFoundError: If the ZIP file specified by zip_file_path
                          does not exist.
        zipfile.BadZipFile: If the file is not a valid ZIP archive or
                           is corrupted.
        OSError: If there are file system issues such as insufficient
                permissions, disk space, or invalid paths.
        PermissionError: If there are insufficient permissions to create
                        the output directory or extract files.
    
    Example:
        >>> extract_zip_file_to_directory(
        ...     "/downloads/dataset.zip",
        ...     "/data/extracted"
        ... )
        # ZIP contents extracted to /data/extracted/
    
    Note:
        Existing files in the output directory may be overwritten if
        the ZIP archive contains files with the same names and paths.
    """
    extract_zip_file(zip_file_path, output_dir)

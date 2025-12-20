"""
Files Package
=============

Centralized file operations for the bioamla package.

This package consolidates all file I/O operations, path manipulation,
file discovery, and download utilities. Domain code should use this
package for all file operations rather than performing I/O directly.

Modules:
    - io: File dataclasses (TextFile, BinaryFile) and read/write functions
    - paths: Path manipulation, sanitization, and directory utilities
    - downloads: File download and URL handling utilities
    - discovery: File discovery and existence checking
"""

from bioamla.core.files.discovery import (
    create_directory,
    directory_exists,
    file_exists,
    get_files_by_extension,
)
from bioamla.core.files.downloads import (
    download_file,
    get_extension_from_content_type,
    get_extension_from_url,
)
from bioamla.core.files.io import (
    BinaryFile,
    File,
    TextFile,
    append_binary,
    append_text,
    read_binary,
    read_text,
    write_binary,
    write_text,
)
from bioamla.core.files.paths import (
    ensure_directory,
    get_relative_path,
    sanitize_filename,
)

__all__ = [
    # io
    "File",
    "TextFile",
    "BinaryFile",
    "read_text",
    "write_text",
    "read_binary",
    "write_binary",
    "append_text",
    "append_binary",
    # paths
    "sanitize_filename",
    "ensure_directory",
    "get_relative_path",
    # downloads
    "download_file",
    "get_extension_from_url",
    "get_extension_from_content_type",
    # discovery
    "get_files_by_extension",
    "file_exists",
    "directory_exists",
    "create_directory",
]

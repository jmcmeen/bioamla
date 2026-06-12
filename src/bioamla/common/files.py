"""
Files Module
============

Pure stdlib file and path operations:
- File discovery and existence checking
- Text/binary read/write helpers
- Download utilities and URL helpers
- Path utilities and sanitization
- ZIP archive helpers
- Raising path validators (require_exists, prepare_output_path)

This module is dependency-free and uses :mod:`pathlib` / :func:`open` /
:mod:`shutil` directly. It does NOT define a File/TextFile/BinaryFile
dataclass hierarchy.
"""

import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from bioamla.exceptions import NotFoundError

logger = logging.getLogger(__name__)


# =============================================================================
# File I/O Utility Functions
# =============================================================================


def read_text(filepath: str | Path, encoding: str = "utf-8") -> str:
    """
    Read entire text file contents.

    Args:
        filepath: Path to the text file
        encoding: Character encoding

    Returns:
        File contents as string
    """
    return Path(filepath).read_text(encoding=encoding)


def write_text(filepath: str | Path, content: str, encoding: str = "utf-8") -> int:
    """
    Write content to a text file, creating parent directories as needed.

    Args:
        filepath: Path to the text file
        content: String content to write
        encoding: Character encoding

    Returns:
        Number of characters written
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)
    return len(content)


def read_binary(filepath: str | Path) -> bytes:
    """
    Read entire binary file contents.

    Args:
        filepath: Path to the binary file

    Returns:
        File contents as bytes
    """
    return Path(filepath).read_bytes()


def write_binary(filepath: str | Path, content: bytes) -> int:
    """
    Write content to a binary file, creating parent directories as needed.

    Args:
        filepath: Path to the binary file
        content: Bytes content to write

    Returns:
        Number of bytes written
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return len(content)


def append_text(filepath: str | Path, content: str, encoding: str = "utf-8") -> int:
    """
    Append content to a text file, creating parent directories as needed.

    Args:
        filepath: Path to the text file
        content: String content to append
        encoding: Character encoding

    Returns:
        Number of characters written
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="a", encoding=encoding) as f:
        return f.write(content)


def append_binary(filepath: str | Path, content: bytes) -> int:
    """
    Append content to a binary file, creating parent directories as needed.

    Args:
        filepath: Path to the binary file
        content: Bytes content to append

    Returns:
        Number of bytes written
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="ab") as f:
        return f.write(content)


# =============================================================================
# File Discovery and Existence Checking
# =============================================================================


def get_files_by_extension(
    directory: str | Path, extensions: list[str] | None = None, recursive: bool = True
) -> list[str]:
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
        extensions = [
            ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
        ]

    files = []
    directory_path = Path(directory)

    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return files

    if recursive:
        for filepath in directory_path.rglob("*"):
            if filepath.is_file():
                if extensions is None or filepath.suffix.lower() in extensions:
                    files.append(str(filepath))
    else:
        for filepath in directory_path.iterdir():
            if filepath.is_file():
                if extensions is None or filepath.suffix.lower() in extensions:
                    files.append(str(filepath))

    return sorted(files)


def file_exists(path: str | Path) -> bool:
    """Check if a file exists."""
    return Path(path).is_file()


def directory_exists(path: str | Path) -> bool:
    """Check if a directory exists."""
    return Path(path).is_dir()


def create_directory(path: str | Path) -> Path:
    """
    Create a directory and all parent directories if they don't exist.

    Args:
        path: Path to the directory to create

    Returns:
        The path that was created as a Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it (and parents) if necessary.

    Args:
        path: Path to the directory

    Returns:
        The path to the directory as a Path object
    """
    path = Path(path) if isinstance(path, str) else path
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Raising Path Validators (replace BaseService._validate_*)
# =============================================================================


def require_exists(path: str | Path) -> Path:
    """
    Return ``Path(path)`` if it exists, else raise :class:`NotFoundError`.

    Args:
        path: Path that must exist.

    Returns:
        The path as a :class:`~pathlib.Path`.

    Raises:
        NotFoundError: If the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise NotFoundError(f"Path does not exist: {path}")
    return p


def prepare_output_path(path: str | Path) -> Path:
    """
    Return ``Path(path)`` after ensuring its parent directory exists.

    Args:
        path: Destination path whose parent should be created.

    Returns:
        The path as a :class:`~pathlib.Path`.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# Download Utilities
# =============================================================================


def get_extension_from_url(url: str) -> str:
    """
    Extract a file extension from a URL.

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
    Map an HTTP Content-Type header to a file extension.

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


def download_file(url: str, output_path: str | Path, show_progress: bool = True) -> Path:
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


# =============================================================================
# Path Utilities
# =============================================================================


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


# =============================================================================
# ZIP Archive Utilities
# =============================================================================


def extract_zip_file(zip_path: str | Path, extract_to: str | Path) -> list[str]:
    """
    Extract a ZIP file to a directory.

    Args:
        zip_path: Path to the ZIP file
        extract_to: Directory to extract files to

    Returns:
        List of extracted file paths
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    extracted_files = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
        extracted_files = [str(extract_to / name) for name in zf.namelist()]

    return extracted_files


def create_zip_file(files: list[str | Path], zip_path: str | Path) -> str:
    """
    Create a ZIP file from a list of files.

    Args:
        files: List of file paths to include in the ZIP
        zip_path: Path for the output ZIP file

    Returns:
        Path to the created ZIP file
    """
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            file = Path(file)
            zf.write(file, file.name)

    return str(zip_path)


def zip_directory(directory: str | Path, zip_path: str | Path) -> str:
    """
    Create a ZIP file from a directory.

    Args:
        directory: Path to the directory to zip
        zip_path: Path for the output ZIP file

    Returns:
        Path to the created ZIP file
    """
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
    # I/O helpers
    "read_text",
    "write_text",
    "read_binary",
    "write_binary",
    "append_text",
    "append_binary",
    # Discovery / existence
    "get_files_by_extension",
    "file_exists",
    "directory_exists",
    "create_directory",
    "ensure_directory",
    # Validators
    "require_exists",
    "prepare_output_path",
    # Download / URL helpers
    "download_file",
    "get_extension_from_url",
    "get_extension_from_content_type",
    # Path utilities
    "sanitize_filename",
    "get_relative_path",
    # ZIP helpers
    "extract_zip_file",
    "create_zip_file",
    "zip_directory",
]

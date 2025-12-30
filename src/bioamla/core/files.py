"""
Files Module
============

Consolidated file and path operations including:
- File dataclasses (TextFile, BinaryFile)
- File discovery and existence checking
- Download utilities
- File I/O utility functions
- Path utilities and sanitization
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


# =============================================================================
# Base File Class
# =============================================================================


@dataclass
class File(ABC):
    """
    Abstract base class for file operations.

    Attributes:
        path: Path to the file
        mode: File mode ('r', 'w', 'a', 'rb', 'wb', etc.)
        encoding: Character encoding (for text files)
    """

    path: Union[str, Path]
    mode: str = "r"
    encoding: Optional[str] = None
    _handle: Optional[IO] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Convert path to Path object if string."""
        if isinstance(self.path, str):
            self.path = Path(self.path)

    @property
    def is_open(self) -> bool:
        """Check if the file handle is open."""
        return self._handle is not None and not self._handle.closed

    @property
    def exists(self) -> bool:
        """Check if the file exists."""
        return self.path.exists()

    @property
    def name(self) -> str:
        """Get the file name."""
        return self.path.name

    @property
    def stem(self) -> str:
        """Get the file name without extension."""
        return self.path.stem

    @property
    def suffix(self) -> str:
        """Get the file extension."""
        return self.path.suffix

    @property
    def size(self) -> int:
        """Get the file size in bytes."""
        if self.exists:
            return self.path.stat().st_size
        return 0

    @property
    def handle(self) -> Optional[IO]:
        """Get the underlying file handle for direct access (e.g., for csv module)."""
        return self._handle

    @abstractmethod
    def open(self) -> "File":
        """Open the file."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the file."""
        pass

    @abstractmethod
    def read(self) -> Any:
        """Read the file contents."""
        pass

    @abstractmethod
    def write(self, content: Any) -> int:
        """Write content to the file."""
        pass

    def __enter__(self) -> "File":
        """Context manager entry."""
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "mode": self.mode,
            "encoding": self.encoding,
            "is_open": self.is_open,
            "exists": self.exists,
            "size": self.size if self.exists else None,
        }


# =============================================================================
# Text File Class
# =============================================================================


@dataclass
class TextFile(File):
    """
    Dataclass for text file operations.

    Attributes:
        path: Path to the file
        mode: File mode ('r', 'w', 'a', 'r+', 'w+', 'a+')
        encoding: Character encoding (default: 'utf-8')
        newline: Newline handling (None, '', '\\n', '\\r', '\\r\\n')
                 Use '' for CSV files to prevent extra blank rows

    Example:
        >>> with TextFile("example.txt", mode="w") as f:
        ...     f.write("Hello, World!")
        >>> with TextFile("example.txt") as f:
        ...     content = f.read()
        >>> print(content)
        Hello, World!

        # For CSV files:
        >>> with TextFile("data.csv", mode="w", newline="") as f:
        ...     writer = csv.writer(f._handle)
        ...     writer.writerow(["col1", "col2"])
    """

    mode: str = "r"
    encoding: Optional[str] = "utf-8"
    newline: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate mode and initialize."""
        super().__post_init__()
        valid_modes = {"r", "w", "a", "r+", "w+", "a+", "x", "x+"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid text mode '{self.mode}'. Must be one of {valid_modes}")

    def open(self) -> "TextFile":
        """
        Open the text file.

        Returns:
            Self for method chaining

        Raises:
            FileNotFoundError: If file doesn't exist and mode is 'r'
            IOError: If file cannot be opened
        """
        if self.is_open:
            logger.warning(f"File already open: {self.path}")
            return self

        try:
            self._handle = open(
                self.path, mode=self.mode, encoding=self.encoding, newline=self.newline
            )
            logger.debug(f"Opened text file: {self.path}")
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Text file not found: {self.path}") from err
        except OSError as e:
            raise OSError(f"Cannot open text file {self.path}: {e}") from e

        return self

    def close(self) -> None:
        """Close the text file."""
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
            logger.debug(f"Closed text file: {self.path}")
        self._handle = None

    def read(self, size: int = -1) -> str:
        """
        Read content from the text file.

        Args:
            size: Number of characters to read (-1 for all)

        Returns:
            File contents as string

        Raises:
            IOError: If file is not open for reading
        """
        if not self.is_open:
            raise OSError(f"File is not open: {self.path}")
        if "r" not in self.mode and "+" not in self.mode:
            raise OSError(f"File is not open for reading: {self.path}")

        return self._handle.read(size)

    def readline(self) -> str:
        """
        Read a single line from the text file.

        Returns:
            A single line as string

        Raises:
            IOError: If file is not open for reading
        """
        if not self.is_open:
            raise OSError(f"File is not open: {self.path}")
        if "r" not in self.mode and "+" not in self.mode:
            raise OSError(f"File is not open for reading: {self.path}")

        return self._handle.readline()

    def readlines(self) -> list[str]:
        """
        Read all lines from the text file.

        Returns:
            List of lines

        Raises:
            IOError: If file is not open for reading
        """
        if not self.is_open:
            raise OSError(f"File is not open: {self.path}")
        if "r" not in self.mode and "+" not in self.mode:
            raise OSError(f"File is not open for reading: {self.path}")

        return self._handle.readlines()

    def write(self, content: str) -> int:
        """
        Write content to the text file.

        Args:
            content: String content to write

        Returns:
            Number of characters written

        Raises:
            IOError: If file is not open for writing
            TypeError: If content is not a string
        """
        if not self.is_open:
            raise OSError(f"File is not open: {self.path}")
        if "r" == self.mode:
            raise OSError(f"File is not open for writing: {self.path}")
        if not isinstance(content, str):
            raise TypeError(f"Content must be str, not {type(content).__name__}")

        return self._handle.write(content)

    def writelines(self, lines: list[str]) -> None:
        """
        Write multiple lines to the text file.

        Args:
            lines: List of strings to write

        Raises:
            IOError: If file is not open for writing
        """
        if not self.is_open:
            raise OSError(f"File is not open: {self.path}")
        if "r" == self.mode:
            raise OSError(f"File is not open for writing: {self.path}")

        self._handle.writelines(lines)

    def flush(self) -> None:
        """Flush the file buffer."""
        if self.is_open:
            self._handle.flush()

    def seek(self, offset: int, whence: int = 0) -> int:
        """
        Move the file cursor.

        Args:
            offset: Position offset
            whence: Reference point (0=start, 1=current, 2=end)

        Returns:
            New cursor position
        """
        if not self.is_open:
            raise OSError(f"File is not open: {self.path}")
        return self._handle.seek(offset, whence)

    def tell(self) -> int:
        """
        Get the current cursor position.

        Returns:
            Current position in the file
        """
        if not self.is_open:
            raise OSError(f"File is not open: {self.path}")
        return self._handle.tell()


# =============================================================================
# Binary File Class
# =============================================================================


@dataclass
class BinaryFile(File):
    """
    Dataclass for binary file operations.

    Attributes:
        path: Path to the file
        mode: File mode ('rb', 'wb', 'ab', 'r+b', 'w+b', 'a+b')

    Example:
        >>> with BinaryFile("data.bin", mode="wb") as f:
        ...     f.write(b"\\x00\\x01\\x02\\x03")
        >>> with BinaryFile("data.bin", mode="rb") as f:
        ...     data = f.read()
        >>> print(data)
        b'\\x00\\x01\\x02\\x03'
    """

    mode: str = "rb"
    encoding: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate mode and initialize."""
        super().__post_init__()
        valid_modes = {"rb", "wb", "ab", "r+b", "w+b", "a+b", "xb", "x+b"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid binary mode '{self.mode}'. Must be one of {valid_modes}")

    def open(self) -> "BinaryFile":
        """
        Open the binary file.

        Returns:
            Self for method chaining

        Raises:
            FileNotFoundError: If file doesn't exist and mode is 'rb'
            IOError: If file cannot be opened
        """
        if self.is_open:
            logger.warning(f"File already open: {self.path}")
            return self

        try:
            self._handle = open(self.path, mode=self.mode)
            logger.debug(f"Opened binary file: {self.path}")
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Binary file not found: {self.path}") from err
        except OSError as e:
            raise OSError(f"Cannot open binary file {self.path}: {e}") from e

        return self

    def close(self) -> None:
        """Close the binary file."""
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
            logger.debug(f"Closed binary file: {self.path}")
        self._handle = None

    def read(self, size: int = -1) -> bytes:
        """
        Read content from the binary file.

        Args:
            size: Number of bytes to read (-1 for all)

        Returns:
            File contents as bytes

        Raises:
            IOError: If file is not open for reading
        """
        if not self.is_open:
            raise OSError(f"File is not open: {self.path}")
        if "r" not in self.mode and "+" not in self.mode:
            raise OSError(f"File is not open for reading: {self.path}")

        return self._handle.read(size)

    def write(self, content: bytes) -> int:
        """
        Write content to the binary file.

        Args:
            content: Bytes content to write

        Returns:
            Number of bytes written

        Raises:
            IOError: If file is not open for writing
            TypeError: If content is not bytes
        """
        if not self.is_open:
            raise OSError(f"File is not open: {self.path}")
        if self.mode == "rb":
            raise OSError(f"File is not open for writing: {self.path}")
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError(f"Content must be bytes or bytearray, not {type(content).__name__}")

        return self._handle.write(content)

    def flush(self) -> None:
        """Flush the file buffer."""
        if self.is_open:
            self._handle.flush()

    def seek(self, offset: int, whence: int = 0) -> int:
        """
        Move the file cursor.

        Args:
            offset: Position offset
            whence: Reference point (0=start, 1=current, 2=end)

        Returns:
            New cursor position
        """
        if not self.is_open:
            raise OSError(f"File is not open: {self.path}")
        return self._handle.seek(offset, whence)

    def tell(self) -> int:
        """
        Get the current cursor position.

        Returns:
            Current position in the file
        """
        if not self.is_open:
            raise OSError(f"File is not open: {self.path}")
        return self._handle.tell()


# =============================================================================
# File I/O Utility Functions
# =============================================================================


def read_text(filepath: Union[str, Path], encoding: str = "utf-8") -> str:
    """
    Read entire text file contents.

    Args:
        filepath: Path to the text file
        encoding: Character encoding

    Returns:
        File contents as string

    Example:
        >>> content = read_text("example.txt")
    """
    with TextFile(filepath, mode="r", encoding=encoding) as f:
        return f.read()


def write_text(filepath: Union[str, Path], content: str, encoding: str = "utf-8") -> int:
    """
    Write content to a text file.

    Args:
        filepath: Path to the text file
        content: String content to write
        encoding: Character encoding

    Returns:
        Number of characters written

    Example:
        >>> write_text("example.txt", "Hello, World!")
    """
    with TextFile(filepath, mode="w", encoding=encoding) as f:
        return f.write(content)


def read_binary(filepath: Union[str, Path]) -> bytes:
    """
    Read entire binary file contents.

    Args:
        filepath: Path to the binary file

    Returns:
        File contents as bytes

    Example:
        >>> data = read_binary("data.bin")
    """
    with BinaryFile(filepath, mode="rb") as f:
        return f.read()


def write_binary(filepath: Union[str, Path], content: bytes) -> int:
    """
    Write content to a binary file.

    Args:
        filepath: Path to the binary file
        content: Bytes content to write

    Returns:
        Number of bytes written

    Example:
        >>> write_binary("data.bin", b"\\x00\\x01\\x02")
    """
    with BinaryFile(filepath, mode="wb") as f:
        return f.write(content)


def append_text(filepath: Union[str, Path], content: str, encoding: str = "utf-8") -> int:
    """
    Append content to a text file.

    Args:
        filepath: Path to the text file
        content: String content to append
        encoding: Character encoding

    Returns:
        Number of characters written

    Example:
        >>> append_text("log.txt", "New log entry\\n")
    """
    with TextFile(filepath, mode="a", encoding=encoding) as f:
        return f.write(content)


def append_binary(filepath: Union[str, Path], content: bytes) -> int:
    """
    Append content to a binary file.

    Args:
        filepath: Path to the binary file
        content: Bytes content to append

    Returns:
        Number of bytes written

    Example:
        >>> append_binary("data.bin", b"\\x04\\x05\\x06")
    """
    with BinaryFile(filepath, mode="ab") as f:
        return f.write(content)


# =============================================================================
# File Discovery and Existence Checking
# =============================================================================


def get_files_by_extension(
    directory: Union[str, Path], extensions: Optional[List[str]] = None, recursive: bool = True
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


def file_exists(path: Union[str, Path]) -> bool:
    """
    Check if a file exists.

    Args:
        path: Path to check

    Returns:
        True if the file exists, False otherwise
    """
    return Path(path).is_file()


def directory_exists(path: Union[str, Path]) -> bool:
    """
    Check if a directory exists.

    Args:
        path: Path to check

    Returns:
        True if the directory exists, False otherwise
    """
    return Path(path).is_dir()


def create_directory(path: Union[str, Path]) -> Path:
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


# =============================================================================
# Download Utilities
# =============================================================================


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

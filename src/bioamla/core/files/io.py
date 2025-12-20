"""
File handling dataclasses for text and binary files.

This module provides dataclasses for working with text and binary files,
including operations for opening, closing, reading, and writing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Dict, Optional, Union

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
# Utility Functions
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

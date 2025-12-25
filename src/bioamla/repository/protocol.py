"""Abstract protocol for file repository operations."""

from pathlib import Path
from typing import List, Protocol, Union


class FileRepositoryProtocol(Protocol):
    """Protocol defining file repository operations.

    All file I/O in services must go through this repository interface
    to enable testing with mocks and alternative implementations.
    """

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a file or directory exists."""
        ...

    def is_file(self, path: Union[str, Path]) -> bool:
        """Check if path is a file."""
        ...

    def is_dir(self, path: Union[str, Path]) -> bool:
        """Check if path is a directory."""
        ...

    def read_binary(self, path: Union[str, Path]) -> bytes:
        """Read file contents as bytes."""
        ...

    def read_text(self, path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read file contents as text."""
        ...

    def write_binary(self, path: Union[str, Path], data: bytes) -> None:
        """Write bytes to file."""
        ...

    def write_text(self, path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
        """Write text to file."""
        ...

    def list_files(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[Path]:
        """List files in directory matching pattern."""
        ...

    def list_dirs(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[Path]:
        """List directories matching pattern."""
        ...

    def mkdir(self, path: Union[str, Path], parents: bool = True) -> None:
        """Create directory (and parents if needed)."""
        ...

    def delete_file(self, path: Union[str, Path]) -> None:
        """Delete a file."""
        ...

    def delete_dir(self, path: Union[str, Path], recursive: bool = False) -> None:
        """Delete a directory."""
        ...

    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> None:
        """Copy a file."""
        ...

    def move_file(self, source: Union[str, Path], destination: Union[str, Path]) -> None:
        """Move/rename a file."""
        ...

    def get_size(self, path: Union[str, Path]) -> int:
        """Get file size in bytes."""
        ...

    def get_extension(self, path: Union[str, Path]) -> str:
        """Get file extension."""
        ...

    def resolve(self, path: Union[str, Path]) -> Path:
        """Resolve to absolute path."""
        ...

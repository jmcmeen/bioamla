"""Mock file repository for testing."""

from pathlib import Path
from typing import Dict, List, Union


class MockFileRepository:
    """Mock implementation of FileRepositoryProtocol for testing.

    Allows tests to simulate file operations without touching the filesystem.
    """

    def __init__(self) -> None:
        """Initialize mock repository with empty filesystem."""
        self.files: Dict[str, bytes] = {}
        self.directories: set = set()

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a file or directory exists."""
        path_str = str(path)
        return path_str in self.files or path_str in self.directories

    def is_file(self, path: Union[str, Path]) -> bool:
        """Check if path is a file."""
        return str(path) in self.files

    def is_dir(self, path: Union[str, Path]) -> bool:
        """Check if path is a directory."""
        return str(path) in self.directories

    def read_binary(self, path: Union[str, Path]) -> bytes:
        """Read file contents as bytes."""
        path_str = str(path)
        if path_str not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        return self.files[path_str]

    def read_text(self, path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read file contents as text."""
        return self.read_binary(path).decode(encoding)

    def write_binary(self, path: Union[str, Path], data: bytes) -> None:
        """Write bytes to file."""
        path_str = str(path)
        # Ensure parent directory exists
        parent = str(Path(path).parent)
        if parent and parent not in self.directories:
            self.directories.add(parent)
        self.files[path_str] = data

    def write_text(
        self, path: Union[str, Path], content: str, encoding: str = "utf-8"
    ) -> None:
        """Write text to file."""
        self.write_binary(path, content.encode(encoding))

    def list_files(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[Path]:
        """List files in directory matching pattern."""
        dir_str = str(directory)
        results = []

        for file_path in self.files.keys():
            file_path_obj = Path(file_path)
            # Check if file is in this directory
            if str(file_path_obj.parent) == dir_str or (
                recursive and file_path.startswith(dir_str)
            ):
                results.append(file_path_obj)

        return results

    def list_dirs(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[Path]:
        """List directories matching pattern."""
        dir_str = str(directory)
        results = []

        for dir_path in self.directories:
            dir_path_obj = Path(dir_path)
            # Check if directory is in this directory
            if str(dir_path_obj.parent) == dir_str or (
                recursive and dir_path.startswith(dir_str)
            ):
                results.append(dir_path_obj)

        return results

    def mkdir(self, path: Union[str, Path], parents: bool = True) -> None:
        """Create directory (and parents if needed)."""
        path_str = str(path)
        self.directories.add(path_str)
        if parents:
            parent = str(Path(path).parent)
            if parent and parent not in self.directories:
                self.mkdir(parent, parents=True)

    def delete_file(self, path: Union[str, Path]) -> None:
        """Delete a file."""
        path_str = str(path)
        if path_str in self.files:
            del self.files[path_str]

    def delete_dir(self, path: Union[str, Path], recursive: bool = False) -> None:
        """Delete a directory."""
        path_str = str(path)
        if recursive:
            # Delete all files and subdirectories
            keys_to_delete = [
                k for k in self.files.keys() if k.startswith(path_str)
            ]
            for k in keys_to_delete:
                del self.files[k]
            dirs_to_delete = [
                d for d in self.directories if d.startswith(path_str)
            ]
            for d in dirs_to_delete:
                self.directories.discard(d)
        self.directories.discard(path_str)

    def copy_file(
        self, source: Union[str, Path], destination: Union[str, Path]
    ) -> None:
        """Copy a file."""
        source_str = str(source)
        if source_str not in self.files:
            raise FileNotFoundError(f"File not found: {source}")
        self.write_binary(destination, self.read_binary(source))

    def move_file(
        self, source: Union[str, Path], destination: Union[str, Path]
    ) -> None:
        """Move/rename a file."""
        self.copy_file(source, destination)
        self.delete_file(source)

    def get_size(self, path: Union[str, Path]) -> int:
        """Get file size in bytes."""
        return len(self.read_binary(path))

    def get_extension(self, path: Union[str, Path]) -> str:
        """Get file extension."""
        return Path(path).suffix.lstrip(".")

    def resolve(self, path: Union[str, Path]) -> Path:
        """Resolve to absolute path."""
        return Path(path).resolve()

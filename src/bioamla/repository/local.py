"""Local filesystem implementation of FileRepositoryProtocol."""

import shutil
from pathlib import Path
from typing import List, Union


class LocalFileRepository:
    """Implementation of FileRepositoryProtocol using local filesystem."""

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a file or directory exists."""
        return Path(path).exists()

    def is_file(self, path: Union[str, Path]) -> bool:
        """Check if path is a file."""
        return Path(path).is_file()

    def is_dir(self, path: Union[str, Path]) -> bool:
        """Check if path is a directory."""
        return Path(path).is_dir()

    def read_binary(self, path: Union[str, Path]) -> bytes:
        """Read file contents as bytes."""
        return Path(path).read_bytes()

    def read_text(self, path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read file contents as text."""
        return Path(path).read_text(encoding=encoding)

    def write_binary(self, path: Union[str, Path], data: bytes) -> None:
        """Write bytes to file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def write_text(self, path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
        """Write text to file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)

    def list_files(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[Path]:
        """List files in directory matching pattern."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return []

        search_pattern = f"**/{pattern}" if recursive else pattern
        return [p for p in dir_path.glob(search_pattern) if p.is_file()]

    def list_dirs(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[Path]:
        """List directories matching pattern."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return []

        search_pattern = f"**/{pattern}" if recursive else pattern
        return [p for p in dir_path.glob(search_pattern) if p.is_dir()]

    def mkdir(self, path: Union[str, Path], parents: bool = True) -> None:
        """Create directory (and parents if needed)."""
        Path(path).mkdir(parents=parents, exist_ok=True)

    def delete_file(self, path: Union[str, Path]) -> None:
        """Delete a file."""
        Path(path).unlink()

    def delete_dir(self, path: Union[str, Path], recursive: bool = False) -> None:
        """Delete a directory."""
        p = Path(path)
        if recursive:
            shutil.rmtree(p)
        else:
            p.rmdir()

    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> None:
        """Copy a file."""
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def move_file(self, source: Union[str, Path], destination: Union[str, Path]) -> None:
        """Move/rename a file."""
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    def get_size(self, path: Union[str, Path]) -> int:
        """Get file size in bytes."""
        return Path(path).stat().st_size

    def get_extension(self, path: Union[str, Path]) -> str:
        """Get file extension."""
        return Path(path).suffix.lstrip(".")

    def resolve(self, path: Union[str, Path]) -> Path:
        """Resolve to absolute path."""
        return Path(path).resolve()

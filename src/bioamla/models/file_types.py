"""File type definitions and metadata structures."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FileType(Enum):
    """Supported file types in BioAMLA."""

    # Audio formats
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"

    # Data formats
    JSON = "json"
    CSV = "csv"
    NPY = "npy"
    PKL = "pkl"
    H5 = "h5"
    PICKLE = "pickle"

    # Annotation formats
    ARBIMON = "arbimon"
    RAVEN = "raven"
    KALEIDOSCOPE = "kaleidoscope"

    # Model formats
    PYTORCH = "pt"
    ONNX = "onnx"

    # Other
    TXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class FileMetadata:
    """Metadata about a file."""

    path: Path
    file_type: FileType
    size_bytes: int
    exists: bool
    is_file: bool = True
    is_readable: bool = True

    @classmethod
    def from_path(cls, path: Path) -> "FileMetadata":
        """Create FileMetadata from a file path."""
        suffix = path.suffix.lstrip(".").lower() if path.suffix else ""

        # Map file extension to FileType
        type_map = {
            "wav": FileType.WAV,
            "mp3": FileType.MP3,
            "flac": FileType.FLAC,
            "ogg": FileType.OGG,
            "m4a": FileType.M4A,
            "json": FileType.JSON,
            "csv": FileType.CSV,
            "npy": FileType.NPY,
            "pkl": FileType.PKL,
            "pickle": FileType.PICKLE,
            "h5": FileType.H5,
            "txt": FileType.TXT,
            "pt": FileType.PYTORCH,
            "onnx": FileType.ONNX,
        }

        file_type = type_map.get(suffix, FileType.UNKNOWN)
        size = path.stat().st_size if path.exists() else 0

        return cls(
            path=path,
            file_type=file_type,
            size_bytes=size,
            exists=path.exists(),
            is_file=path.is_file() if path.exists() else True,
        )

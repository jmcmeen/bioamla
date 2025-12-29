# services/base.py
"""
Base class and utilities for all services.
"""

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar

if TYPE_CHECKING:
    from bioamla.repository.protocol import FileRepositoryProtocol

T = TypeVar("T")



@dataclass
class ServiceResult(Generic[T]):
    """
    Result object returned by service operations.

    Provides a consistent interface for views to handle operation outcomes.
    """

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        data: T = None,
        message: str = None,
        warnings: List[str] = None,
        **metadata,
    ) -> "ServiceResult[T]":
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            message=message,
            warnings=warnings or [],
            metadata=metadata,
        )

    @classmethod
    def fail(cls, error: str, warnings: List[str] = None, **metadata) -> "ServiceResult[T]":
        """Create a failed result."""
        return cls(
            success=False,
            error=error,
            warnings=warnings or [],
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            "success": self.success,
            "message": self.message,
            "error": self.error,
            "warnings": self.warnings,
        }

        # Serialize data if present
        if self.data is not None:
            if hasattr(self.data, "to_dict"):
                result["data"] = self.data.to_dict()
            elif is_dataclass(self.data):
                result["data"] = asdict(self.data)
            elif isinstance(self.data, (dict, list, str, int, float, bool)):
                result["data"] = self.data
            else:
                result["data"] = str(self.data)
        else:
            result["data"] = None

        # Include non-empty metadata
        if self.metadata:
            result["metadata"] = self.metadata

        return result


@dataclass
class BatchProgress:
    """Progress information for batch operations."""

    total: int
    completed: int = 0
    current_file: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    @property
    def percent(self) -> float:
        """Get completion percentage."""
        return (self.completed / self.total * 100) if self.total > 0 else 0

    @property
    def remaining(self) -> int:
        """Get remaining items."""
        return self.total - self.completed


# Type alias for progress callback
ProgressCallback = Callable[[BatchProgress], None]


class BaseService:
    """
    Base class for all services.

    Provides common functionality for:
    - File discovery and validation
    - Batch processing with progress
    - Error handling and result formatting
    """

    def __init__(self, file_repository: Optional["FileRepositoryProtocol"] = None) -> None:
        """Initialize the service.

        Args:
            file_repository: Optional file repository for dependency injection.
                           Required for file-based services, not needed for API/in-memory services.
        """
        self.file_repository = file_repository
        self._progress_callback: Optional[ProgressCallback] = None

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """Set a callback for progress updates during batch operations."""
        self._progress_callback = callback

    def _report_progress(self, progress: BatchProgress) -> None:
        """Report progress if a callback is set."""
        if self._progress_callback:
            self._progress_callback(progress)

    def _get_audio_files(
        self,
        path: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        Discover audio files in a path.

        Args:
            path: File or directory path
            recursive: Search subdirectories
            extensions: File extensions to include (default: common audio formats)

        Returns:
            List of audio file paths
        """
        from bioamla.core.utils import get_audio_files

        path = Path(path)

        if path.is_file():
            return [path]

        # get_audio_files handles directory scanning
        files = get_audio_files(str(path), recursive=recursive)
        return [Path(f) for f in files]

    def _validate_input_path(self, path: str, must_exist: bool = True) -> Optional[str]:
        """
        Validate an input path.

        Returns:
            None if valid, error message if invalid
        """
        p = Path(path)
        if must_exist and not p.exists():
            return f"Path does not exist: {path}"
        return None

    def _validate_output_path(
        self,
        path: str,
        allow_overwrite: bool = True,
        create_parents: bool = True,
    ) -> Optional[str]:
        """
        Validate an output path.

        Returns:
            None if valid, error message if invalid
        """
        p = Path(path)

        if not allow_overwrite and p.exists():
            return f"Output path already exists: {path}"

        if create_parents:
            p.parent.mkdir(parents=True, exist_ok=True)

        return None

    def _process_batch(
        self,
        files: List[Path],
        processor: Callable[[Path], T],
        description: str = "Processing",
    ) -> Iterator[tuple[Path, Optional[T], Optional[str]]]:
        """
        Process a batch of files with progress reporting.

        Args:
            files: List of files to process
            processor: Function to process each file
            description: Description for progress reporting

        Yields:
            Tuple of (file_path, result, error) for each file
        """
        progress = BatchProgress(total=len(files))
        self._report_progress(progress)

        for filepath in files:
            progress.current_file = str(filepath)
            self._report_progress(progress)

            try:
                result = processor(filepath)
                yield (filepath, result, None)
            except Exception as e:
                error = str(e)
                progress.errors.append(f"{filepath.name}: {error}")
                yield (filepath, None, error)

            progress.completed += 1
            self._report_progress(progress)

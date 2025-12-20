# controllers/base.py
"""
Base Controller
===============

Base class and utilities for all controllers.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class ToDictMixin:
    """
    Mixin that adds to_dict() method to dataclasses.

    Handles nested dataclasses, lists, and common types automatically.
    Override _to_dict_extra() to add custom serialization logic.

    Example:
        @dataclass
        class MyResult(ToDictMixin):
            name: str
            count: int

        result = MyResult(name="test", count=5)
        data = result.to_dict()  # {"name": "test", "count": 5}
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary, handling nested structures."""
        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} is not a dataclass")

        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            result[f.name] = self._serialize_value(value)

        # Allow subclasses to add extra fields
        extra = self._to_dict_extra()
        if extra:
            result.update(extra)

        return result

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value for dict output."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if is_dataclass(value):
            return asdict(value)
        # Fallback for other types
        return str(value)

    def _to_dict_extra(self) -> Optional[Dict[str, Any]]:
        """Override to add extra fields to dict output."""
        return None


@dataclass
class ControllerResult(Generic[T]):
    """
    Result object returned by controller operations.

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
    ) -> "ControllerResult[T]":
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            message=message,
            warnings=warnings or [],
            metadata=metadata,
        )

    @classmethod
    def fail(cls, error: str, warnings: List[str] = None, **metadata) -> "ControllerResult[T]":
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


class BaseController:
    """
    Base class for all controllers.

    Provides common functionality for:
    - File discovery and validation
    - Batch processing with progress
    - Error handling and result formatting
    - Run tracking in project repository (supports JSON, SQLite, PostgreSQL)
    """

    def __init__(self):
        self._progress_callback: Optional[ProgressCallback] = None
        self._current_run_id: Optional[str] = None
        self._storage = None

    @property
    def storage(self):
        """Get the storage instance (lazy initialization)."""
        if self._storage is None:
            from bioamla.core.storage import get_storage

            self._storage = get_storage()
        return self._storage

    def _start_run(
        self,
        name: str,
        action: str,
        input_path: str = "",
        output_path: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Start a new run in the project repository.

        Args:
            name: Run name/description
            action: Action being performed (e.g., 'predict', 'embed', 'indices')
            input_path: Input file/directory
            output_path: Output file/directory
            parameters: Run parameters

        Returns:
            Run ID if in a project, None otherwise
        """
        run_id = self.storage.create_run(
            name=name,
            action=action,
            controller=self.__class__.__name__,
            input_path=input_path,
            output_path=output_path,
            parameters=parameters,
        )
        self._current_run_id = run_id
        return run_id

    def _complete_run(
        self,
        run_id: Optional[str] = None,
        status: str = "completed",
        results: Optional[Dict[str, Any]] = None,
        output_files: Optional[List[str]] = None,
    ) -> bool:
        """
        Mark a run as complete.

        Args:
            run_id: Run ID (defaults to current run)
            status: Final status (completed, failed, cancelled)
            results: Run results/summary
            output_files: List of output file paths

        Returns:
            True if successful, False otherwise
        """
        run_id = run_id or self._current_run_id
        if run_id is None:
            return False

        success = self.storage.update_run(
            run_id=run_id,
            status=status,
            results=results,
            output_files=output_files,
        )

        if success and run_id == self._current_run_id:
            self._current_run_id = None

        return success

    def _fail_run(
        self,
        error_message: str,
        run_id: Optional[str] = None,
    ) -> bool:
        """
        Mark a run as failed.

        Args:
            error_message: Error message
            run_id: Run ID (defaults to current run)

        Returns:
            True if successful, False otherwise
        """
        run_id = run_id or self._current_run_id
        if run_id is None:
            return False

        success = self.storage.fail_run(run_id, error_message)

        if success and run_id == self._current_run_id:
            self._current_run_id = None

        return success

    def _save_run_artifact(
        self,
        filename: str,
        data: Any,
        run_id: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Save an artifact to the run directory.

        Args:
            filename: Name of the file to save
            data: Data to save (dict for JSON, str for text)
            run_id: Run ID (defaults to current run)

        Returns:
            Path to saved file, or None if failed
        """
        run_id = run_id or self._current_run_id
        if run_id is None:
            return None

        return self.storage.save_artifact(run_id, filename, data)

    def _get_run_artifact(
        self,
        filename: str,
        run_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Load an artifact from the run directory.

        Args:
            filename: Artifact filename
            run_id: Run ID (defaults to current run)

        Returns:
            Loaded data or None
        """
        run_id = run_id or self._current_run_id
        if run_id is None:
            return None

        return self.storage.get_artifact(run_id, filename)

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

"""Base class for batch processing services."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, TypeVar

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.base import BaseService

T = TypeVar("T")


class BatchServiceBase(BaseService, ABC):
    """Base class for batch processing services.

    Handles common batch operations like file discovery, parallel processing,
    and result aggregation.

    Note: file_repository is optional. Batch services that process files should
    provide a file_repository, but batch services that perform API calls or
    in-memory operations may not need one.
    """

    def __init__(self, file_repository: Optional[FileRepositoryProtocol] = None) -> None:
        """Initialize batch service with optional file repository.

        Args:
            file_repository: Optional file repository for file discovery and I/O.
                           Required for batch services that process files.
                           Not needed for batch services that perform API calls or in-memory operations.
        """
        super().__init__(file_repository)

    @abstractmethod
    def process_file(self, file_path: Path) -> Any:
        """Process a single file. Must be implemented by subclasses."""
        ...

    def process_batch(
        self,
        config: BatchConfig,
        file_filter: Optional[Callable[[Path], bool]] = None,
    ) -> BatchResult:
        """Process multiple files with given configuration.

        Args:
            config: Batch processing configuration
            file_filter: Optional function to filter which files to process

        Returns:
            BatchResult with processing summary

        Raises:
            ValueError: If file_repository is None (required for file-based batch processing)
        """
        start_time = datetime.now()
        result = BatchResult(start_time=start_time.isoformat())

        # Find files to process
        if self.file_repository is None:
            raise ValueError(
                "file_repository is required for file-based batch processing. "
                "Pass a FileRepositoryProtocol instance to the constructor."
            )

        input_path = Path(config.input_dir)
        files = self.file_repository.list_files(
            input_path,
            pattern="*",
            recursive=config.recursive,
        )

        if file_filter:
            files = [f for f in files if file_filter(f)]

        result.total_files = len(files)

        # Process files
        if config.max_workers > 1:
            result = self._process_parallel(files, config, result)
        else:
            result = self._process_sequential(files, config, result)

        # Finalize result
        end_time = datetime.now()
        result.end_time = end_time.isoformat()
        result.duration_seconds = (end_time - start_time).total_seconds()

        return result

    def _process_sequential(
        self, files: List[Path], config: BatchConfig, result: BatchResult
    ) -> BatchResult:
        """Process files sequentially."""
        for file_path in files:
            try:
                self.process_file(file_path)
                result.successful += 1
            except Exception as e:
                result.failed += 1
                error_msg = f"{file_path}: {str(e)}"
                result.errors.append(error_msg)
                if not config.continue_on_error:
                    raise
                if not config.quiet:
                    print(f"Error processing {file_path}: {e}")

        return result

    def _process_parallel(
        self, files: List[Path], config: BatchConfig, result: BatchResult
    ) -> BatchResult:
        """Process files in parallel."""
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(self.process_file, file_path): file_path for file_path in files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()
                    result.successful += 1
                except Exception as e:
                    result.failed += 1
                    error_msg = f"{file_path}: {str(e)}"
                    result.errors.append(error_msg)
                    if not config.continue_on_error:
                        raise
                    if not config.quiet:
                        print(f"Error processing {file_path}: {e}")

        return result

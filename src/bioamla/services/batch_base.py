"""Base class for batch processing services."""

from abc import ABC, abstractmethod
from concurrent.futures import as_completed
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
        self._csv_handler: Optional[Any] = None  # Lazy import to avoid circular dependency
        self._csv_context: Optional[Any] = None  # Current CSV context during processing

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
        """Process files in parallel using ProcessPoolExecutor for true parallelism."""
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
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

    def process_batch_csv(
        self,
        config: BatchConfig,
        file_filter: Optional[Callable[[Path], bool]] = None,
    ) -> BatchResult:
        """Process files from metadata CSV.

        Args:
            config: Batch processing configuration with input_file set
            file_filter: Optional function to filter which files to process

        Returns:
            BatchResult with processing summary

        Raises:
            ValueError: If file_repository is None or config.input_file is None
        """
        # Lazy import to avoid circular dependency
        from bioamla.services.batch_csv import BatchCSVHandler

        start_time = datetime.now()
        result = BatchResult(start_time=start_time.isoformat())

        # Validate repository
        if self.file_repository is None:
            raise ValueError(
                "file_repository is required for CSV-based batch processing. "
                "Pass a FileRepositoryProtocol instance to the constructor."
            )

        if config.input_file is None:
            raise ValueError("config.input_file must be specified for CSV-based processing")

        # Initialize CSV handler
        self._csv_handler = BatchCSVHandler(self.file_repository)

        # Load CSV context
        self._csv_context = self._csv_handler.load_csv(config.input_file, config.output_dir)

        # Filter files if filter provided
        rows_to_process = self._csv_context.rows
        if file_filter:
            rows_to_process = [row for row in rows_to_process if file_filter(row.file_path)]

        result.total_files = len(rows_to_process)

        # Process files
        if config.max_workers > 1:
            result = self._process_csv_parallel(rows_to_process, config, result)
        else:
            result = self._process_csv_sequential(rows_to_process, config, result)

        # Write updated CSV
        output_csv = self._csv_handler.write_csv(self._csv_context)
        if not config.quiet:
            print(f"Updated metadata CSV written to: {output_csv}")

        # Finalize result
        end_time = datetime.now()
        result.end_time = end_time.isoformat()
        result.duration_seconds = (end_time - start_time).total_seconds()

        return result

    def _process_csv_sequential(
        self, rows: List[Any], config: BatchConfig, result: BatchResult
    ) -> BatchResult:
        """Process CSV rows sequentially."""
        for row in rows:
            try:
                # Check if file exists
                if not row.file_path.exists():
                    raise FileNotFoundError(f"File not found: {row.file_path}")

                # Process the file
                self.process_file(row.file_path)
                result.successful += 1
            except Exception as e:
                result.failed += 1
                error_msg = f"{row.file_path}: {str(e)}"
                result.errors.append(error_msg)
                if not config.continue_on_error:
                    raise
                if not config.quiet:
                    print(f"Error processing {row.file_path}: {e}")

        return result

    def _process_csv_parallel(
        self, rows: List[Any], config: BatchConfig, result: BatchResult
    ) -> BatchResult:
        """Process CSV rows in parallel using ProcessPoolExecutor for true parallelism."""
        import sys
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {}

            # Check file existence before submitting to executor (fail fast)
            for row in rows:
                if not row.file_path.exists():
                    # Handle missing file immediately
                    result.failed += 1
                    error_msg = f"{row.file_path}: File not found"
                    result.errors.append(error_msg)
                    if not config.continue_on_error:
                        raise FileNotFoundError(error_msg)
                    if not config.quiet:
                        print(f"Error: {error_msg}", flush=True)
                else:
                    # Submit only existing files
                    futures[executor.submit(self.process_file, row.file_path)] = row

            for future in as_completed(futures):
                row = futures[future]
                try:
                    future.result()
                    result.successful += 1
                except Exception as e:
                    result.failed += 1
                    error_msg = f"{row.file_path}: {str(e)}"
                    result.errors.append(error_msg)
                    if not config.continue_on_error:
                        raise
                    if not config.quiet:
                        print(f"Error processing {row.file_path}: {e}", flush=True)

        # Ensure all output is flushed before returning
        sys.stdout.flush()
        sys.stderr.flush()

        return result

    def process_batch_auto(
        self,
        config: BatchConfig,
        file_filter: Optional[Callable[[Path], bool]] = None,
    ) -> BatchResult:
        """Auto-dispatch to CSV or directory mode based on config.

        Convenience method that routes to process_batch_csv() or process_batch()
        based on whether config.input_file is set.

        Args:
            config: Batch processing configuration
            file_filter: Optional function to filter which files to process

        Returns:
            BatchResult with processing summary
        """
        if config.input_file:
            return self.process_batch_csv(config, file_filter)
        else:
            return self.process_batch(config, file_filter)

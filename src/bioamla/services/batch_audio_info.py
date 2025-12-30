"""Batch audio info service."""

import csv
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.batch_base import BatchServiceBase


class BatchAudioInfoService(BatchServiceBase):
    """Service for batch audio metadata extraction.

    This service extracts metadata (duration, sample rate, channels, format, etc.)
    from audio files and outputs to CSV.
    """

    def __init__(
        self,
        file_repository: FileRepositoryProtocol,
    ) -> None:
        """Initialize batch audio info service.

        Args:
            file_repository: File repository for file discovery
        """
        super().__init__(file_repository)
        self._all_results: List[Dict[str, Any]] = []

    def process_file(self, file_path: Path) -> Any:
        """Extract metadata from a single audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary with audio metadata

        Raises:
            RuntimeError: If metadata extraction fails
        """
        try:
            # Use fast ffprobe-based metadata extraction
            from bioamla.adapters.pydub import get_audio_info

            # Get audio metadata
            info = get_audio_info(str(file_path))

            # Add filepath to result
            result = {
                "filepath": str(file_path),
                "filename": file_path.name,
                "duration": info["duration"],
                "sample_rate": info["sample_rate"],
                "channels": info["channels"],
                "samples": info["samples"],
                "format": info["format"],
                "codec": info.get("codec", ""),
                "bit_depth": info.get("bit_depth", ""),
                "subtype": info.get("subtype", ""),
            }

            # Collect for aggregated output
            self._all_results.append(result)

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to extract audio metadata: {e}")

    def _process_parallel(
        self, files: List[Path], config: BatchConfig, result: BatchResult
    ) -> BatchResult:
        """Override to collect results from parallel workers.

        Args:
            files: List of file paths to process
            config: Batch configuration
            result: BatchResult to update

        Returns:
            Updated BatchResult
        """
        import sys
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {executor.submit(self.process_file, file_path): file_path for file_path in files}

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    # Collect result from worker process
                    file_result = future.result()
                    self._all_results.append(file_result)
                    result.successful += 1
                except Exception as e:
                    result.failed += 1
                    error_msg = f"{file_path}: {str(e)}"
                    result.errors.append(error_msg)
                    if not config.continue_on_error:
                        raise
                    if not config.quiet:
                        print(f"Error processing {file_path}: {e}", flush=True)

        # Ensure all output is flushed before returning
        sys.stdout.flush()
        sys.stderr.flush()

        return result

    def _process_csv_sequential(
        self, rows: Any, config: BatchConfig, result: BatchResult
    ) -> BatchResult:
        """Override to merge audio info into CSV rows.

        Args:
            rows: List of MetadataRow objects to process
            config: Batch configuration
            result: BatchResult to update

        Returns:
            Updated BatchResult
        """
        # Call parent to do the actual processing
        result = super()._process_csv_sequential(rows, config, result)

        # Merge audio info into CSV rows
        if self._csv_context is not None and self._csv_handler is not None:
            # Create a mapping from file path to audio info
            results_by_path = {Path(r["filepath"]): r for r in self._all_results}

            for row in self._csv_context.rows:
                if row.file_path in results_by_path:
                    info_data = results_by_path[row.file_path].copy()
                    # Remove filepath/filename (redundant with CSV path columns)
                    info_data.pop("filepath", None)
                    info_data.pop("filename", None)
                    # Merge info into row metadata
                    self._csv_handler.merge_analysis_results(row, info_data)

        return result

    def _process_csv_parallel(
        self, rows: Any, config: BatchConfig, result: BatchResult
    ) -> BatchResult:
        """Override to collect results from parallel workers and merge into CSV.

        Args:
            rows: List of MetadataRow objects to process
            config: Batch configuration
            result: BatchResult to update

        Returns:
            Updated BatchResult
        """
        import sys
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Use ProcessPoolExecutor for true parallelism
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
                    # Collect result from worker process
                    file_result = future.result()
                    self._all_results.append(file_result)
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

        # Merge audio info into CSV rows
        if self._csv_context is not None and self._csv_handler is not None:
            # Create a mapping from file path to audio info
            results_by_path = {Path(r["filepath"]): r for r in self._all_results}

            for row in self._csv_context.rows:
                if row.file_path in results_by_path:
                    info_data = results_by_path[row.file_path].copy()
                    # Remove filepath/filename (redundant with CSV path columns)
                    info_data.pop("filepath", None)
                    info_data.pop("filename", None)
                    # Merge info into row metadata
                    self._csv_handler.merge_analysis_results(row, info_data)

        return result

    def _write_aggregated_results(self, output_dir: Optional[str]) -> None:
        """Write all audio info results to a CSV file.

        Args:
            output_dir: Directory to write results to (None for CSV in-place mode)
        """
        if not self._all_results:
            return

        # In CSV mode without output_dir, results are already merged into metadata CSV
        if output_dir is None:
            return

        output_path = Path(output_dir) / "audio_info.csv"
        self.file_repository.mkdir(str(output_path.parent), parents=True)

        # Write CSV to in-memory buffer
        buffer = StringIO()

        # Get all field names from results
        if self._all_results:
            fieldnames = list(self._all_results[0].keys())
            writer = csv.DictWriter(buffer, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._all_results:
                writer.writerow(row)

            # Write buffer contents to file via repository
            self.file_repository.write_text(output_path, buffer.getvalue())

    def info_batch(
        self,
        config: BatchConfig,
    ) -> BatchResult:
        """Extract audio metadata for audio files batch-wise.

        Args:
            config: Batch processing configuration

        Returns:
            BatchResult with processing summary
        """
        self._all_results = []

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        result = self.process_batch_auto(config, file_filter=audio_filter)
        self._write_aggregated_results(config.output_dir)
        return result

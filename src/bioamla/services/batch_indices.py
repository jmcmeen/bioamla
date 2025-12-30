"""Batch acoustic indices service."""

import csv
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.audio_file import AudioData
from bioamla.services.batch_base import BatchServiceBase
from bioamla.services.indices import IndicesService


class BatchIndicesService(BatchServiceBase):
    """Service for batch acoustic indices computation.

    This service delegates to IndicesService for actual index calculations,
    following the dependency injection pattern.
    """

    def __init__(
        self,
        file_repository: FileRepositoryProtocol,
        indices_service: IndicesService,
    ) -> None:
        """Initialize batch indices service.

        Args:
            file_repository: File repository for file discovery
            indices_service: Single-file indices service to delegate to
        """
        super().__init__(file_repository)
        self.indices_service = indices_service
        self._current_indices: Optional[List[str]] = None
        self._current_params: Dict[str, Any] = {}
        self._all_results: List[Dict[str, Any]] = []

    def process_file(self, file_path: Path) -> Any:
        """Process a single audio file by delegating to IndicesService.

        Args:
            file_path: Path to the audio file to process

        Returns:
            Indices result

        Raises:
            ValueError: If indices are not set
            RuntimeError: If the underlying service operation fails
        """
        if self._current_indices is None:
            raise ValueError("Indices not set. Call calculate_batch first.")

        try:
            # Load audio file using hybrid loader (soundfile for WAV/FLAC, pydub for M4A/MP3)
            from bioamla.adapters.pydub import load_audio

            # Read audio file (always returns mono float32)
            audio, sample_rate = load_audio(str(file_path))

            # Create AudioData object
            audio_data = AudioData(
                samples=audio,
                sample_rate=sample_rate,
                channels=1,
                source_path=str(file_path),
            )

            # Calculate indices
            result = self.indices_service.calculate(
                audio_data,
                indices=self._current_indices,
                **self._current_params,
            )

            if not result.success:
                raise RuntimeError(result.error)

            # Convert result to dict and add filepath
            indices_dict = result.data.to_dict()
            indices_dict["filepath"] = str(file_path)

            # Collect for aggregated output
            self._all_results.append(indices_dict)

            return indices_dict

        except Exception as e:
            raise RuntimeError(f"Failed to calculate indices: {e}")

    def _process_csv_sequential(
        self, rows: Any, config: BatchConfig, result: BatchResult
    ) -> BatchResult:
        """Override to merge indices results into CSV rows.

        Args:
            rows: List of MetadataRow objects to process
            config: Batch configuration
            result: BatchResult to update

        Returns:
            Updated BatchResult
        """
        # Call parent to do the actual processing
        result = super()._process_csv_sequential(rows, config, result)

        # Merge indices results into CSV rows
        if self._csv_context is not None and self._csv_handler is not None:
            # Create a mapping from file path to indices results
            results_by_path = {Path(r["filepath"]): r for r in self._all_results}

            for row in self._csv_context.rows:
                if row.file_path in results_by_path:
                    indices_data = results_by_path[row.file_path].copy()
                    # Remove filepath from indices data (it's redundant)
                    indices_data.pop("filepath", None)
                    # Merge indices into row metadata
                    self._csv_handler.merge_analysis_results(row, indices_data)

        return result

    def _process_csv_parallel(
        self, rows: Any, config: BatchConfig, result: BatchResult
    ) -> BatchResult:
        """Override to merge indices results into CSV rows during parallel processing.

        Args:
            rows: List of MetadataRow objects to process
            config: Batch configuration
            result: BatchResult to update

        Returns:
            Updated BatchResult
        """
        # Call parent to do the actual processing
        result = super()._process_csv_parallel(rows, config, result)

        # Merge indices results into CSV rows
        if self._csv_context is not None and self._csv_handler is not None:
            # Create a mapping from file path to indices results
            results_by_path = {Path(r["filepath"]): r for r in self._all_results}

            for row in self._csv_context.rows:
                if row.file_path in results_by_path:
                    indices_data = results_by_path[row.file_path].copy()
                    # Remove filepath from indices data (it's redundant)
                    indices_data.pop("filepath", None)
                    # Merge indices into row metadata
                    self._csv_handler.merge_analysis_results(row, indices_data)

        return result

    def _write_aggregated_results(self, output_dir: Optional[str]) -> None:
        """Write all indices results to a CSV file.

        Args:
            output_dir: Directory to write results to (None for CSV in-place mode)
        """
        if not self._all_results:
            return

        # In CSV mode without output_dir, results are already merged into metadata CSV
        if output_dir is None:
            return

        output_path = Path(output_dir) / "indices_results.csv"
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

    def calculate_batch(
        self,
        config: BatchConfig,
        indices: Optional[List[str]] = None,
        **kwargs,
    ) -> BatchResult:
        """Calculate acoustic indices for audio files batch-wise.

        Args:
            config: Batch processing configuration
            indices: List of indices to compute (default: all)
            **kwargs: Additional parameters for index calculations

        Returns:
            BatchResult with processing summary
        """
        self._current_indices = indices
        self._current_params = kwargs
        self._all_results = []

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        result = self.process_batch_auto(config, file_filter=audio_filter)
        self._write_aggregated_results(config.output_dir)
        return result

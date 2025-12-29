"""Batch audio transformation service."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.audio_transform import AudioTransformService
from bioamla.services.batch_base import BatchServiceBase


class BatchAudioTransformService(BatchServiceBase):
    """Service for batch audio transformations (resample, normalize, segment, visualize).

    This service delegates to AudioTransformService for actual file processing,
    following the dependency injection pattern.
    """

    def __init__(
        self,
        file_repository: FileRepositoryProtocol,
        audio_transform_service: AudioTransformService,
    ) -> None:
        """Initialize batch audio transform service.

        Args:
            file_repository: File repository for file discovery
            audio_transform_service: Single-file audio transform service to delegate to
        """
        super().__init__(file_repository)
        self.audio_transform_service = audio_transform_service
        self._current_operation: Optional[str] = None
        self._current_config: Dict[str, Any] = {}
        self._output_paths: Dict[Path, Path] = {}  # Track input -> output path mapping for CSV updates
        self._segment_mapping: Dict[Path, List[Any]] = {}  # Track input -> segment list for CSV row expansion

    def process_file(self, file_path: Path) -> Any:
        """Process a single audio file by delegating to AudioTransformService.

        Dispatches to the appropriate method based on _current_operation.

        Args:
            file_path: Path to the audio file to process

        Returns:
            Result of the operation

        Raises:
            ValueError: If operation is not set or unknown
            RuntimeError: If the underlying service operation fails
        """
        if self._current_operation is None:
            raise ValueError("Operation not set. Call a batch method first.")

        # Calculate output path and output_dir
        # CSV mode: use BatchCSVHandler to resolve output path
        if self._csv_context is not None and self._csv_handler is not None:
            # Determine new extension based on operation
            new_extension = None
            if self._current_operation == "convert":
                target_format = self._current_config.get("target_format", "wav")
                new_extension = f".{target_format}"
            elif self._current_operation == "visualize":
                new_extension = ".png"

            output_path = self._csv_handler.resolve_output_path(
                file_path, self._csv_context, new_extension=new_extension
            )
            # For segment operation, we need output_dir to be the parent of output_path
            output_dir = output_path.parent
        else:
            # Directory mode: use output_dir from config
            output_dir = Path(self._current_config.get("output_dir", "."))
            output_path = output_dir / file_path.name

        # Ensure output directory exists
        self.file_repository.mkdir(str(output_path.parent), parents=True)

        # Dispatch to appropriate operation
        if self._current_operation == "resample":
            result = self.audio_transform_service.resample_file(
                str(file_path),
                str(output_path),
                target_rate=self._current_config["target_sr"],
            )
        elif self._current_operation == "normalize":
            result = self.audio_transform_service.normalize_file(
                str(file_path),
                str(output_path),
                target_db=self._current_config.get("target_db", -20.0),
                peak=self._current_config.get("peak", False),
            )
        elif self._current_operation == "trim":
            result = self.audio_transform_service.trim_file(
                str(file_path),
                str(output_path),
                start=self._current_config.get("start"),
                end=self._current_config.get("end"),
                trim_silence=self._current_config.get("trim_silence", False),
                silence_threshold_db=self._current_config.get("silence_threshold_db", -40.0),
            )
        elif self._current_operation == "filter":
            result = self.audio_transform_service.filter_file(
                str(file_path),
                str(output_path),
                lowpass=self._current_config.get("lowpass"),
                highpass=self._current_config.get("highpass"),
                bandpass=self._current_config.get("bandpass"),
                order=self._current_config.get("order", 5),
            )
        elif self._current_operation == "denoise":
            result = self.audio_transform_service.denoise_file(
                str(file_path),
                str(output_path),
                strength=self._current_config.get("strength", 1.0),
            )
        elif self._current_operation == "segment":
            # FIX: Pass output_dir directly and use prefix parameter
            result = self.audio_transform_service.segment_file(
                str(file_path),
                str(output_dir),  # Just the parent directory, not output_dir/stem
                duration=self._current_config["segment_duration"],
                overlap=self._current_config.get("overlap", 0.0),
                prefix=file_path.stem,  # Use prefix parameter for filename
            )

            # Capture segment info for CSV row expansion
            if result.success and result.data and isinstance(result.data, dict):
                segments = result.data.get("segments", [])
                if segments and self._csv_context is not None:
                    self._segment_mapping[file_path] = segments
        elif self._current_operation == "visualize":
            # Only update extension in directory mode (CSV mode handles it in resolve_output_path)
            if self._csv_context is None:
                output_path = output_path.with_suffix(".png")
            result = self.audio_transform_service.visualize_file(
                str(file_path),
                str(output_path),
                viz_type=self._current_config.get("plot_type", "mel"),
                show_legend=self._current_config.get("show_legend", True),
            )
        elif self._current_operation == "convert":
            # Convert using AudioFileService
            import numpy as np

            from bioamla.services.audio_file import AudioFileService

            audio_file_service = AudioFileService(file_repository=self.file_repository)
            target_format = self._current_config.get("target_format", "wav")
            target_sr = self._current_config.get("target_sr")
            target_channels = self._current_config.get("target_channels")

            # Update output path with new format (only in directory mode, CSV mode handles it already)
            if self._csv_context is None:
                output_path = output_path.with_suffix(f".{target_format}")

            # Load audio
            open_result = audio_file_service.open(str(file_path))
            if not open_result.success:
                raise RuntimeError(open_result.error)

            audio_data = open_result.data

            # Handle sample rate conversion
            if target_sr is not None and target_sr != audio_data.sample_rate:
                from scipy import signal
                num_samples = int(len(audio_data.samples) * target_sr / audio_data.sample_rate)
                audio_data.samples = signal.resample(audio_data.samples, num_samples)
                audio_data.sample_rate = target_sr

            # Handle channel conversion
            if target_channels is not None and target_channels != audio_data.channels:
                if target_channels == 1 and audio_data.channels == 2:
                    if audio_data.samples.ndim == 2:
                        audio_data.samples = audio_data.samples.mean(axis=1)
                    audio_data.channels = 1
                elif target_channels == 2 and audio_data.channels == 1:
                    audio_data.samples = np.column_stack([audio_data.samples, audio_data.samples])
                    audio_data.channels = 2

            # Save converted audio
            result = audio_file_service.save(audio_data, str(output_path))

            # Delete original file if requested and conversion was successful
            if result.success and self._current_config.get("delete_original", False):
                # Only delete if output path is different from input path
                if output_path != file_path:
                    try:
                        self.file_repository.delete_file(str(file_path))
                    except Exception as e:
                        # Log warning but don't fail the conversion
                        import warnings
                        warnings.warn(f"Failed to delete original file {file_path}: {e}")
        else:
            raise ValueError(f"Unknown operation: {self._current_operation}")

        if not result.success:
            raise RuntimeError(result.error)

        # Track output path for CSV row updates (only needed for file-changing operations)
        if self._csv_context is not None:
            self._output_paths[file_path] = output_path

        return result.data

    def _process_csv_sequential(
        self, rows: Any, config: BatchConfig, result: BatchResult
    ) -> BatchResult:
        """Override to update CSV rows during processing.

        Args:
            rows: List of MetadataRow objects to process
            config: Batch configuration
            result: BatchResult to update

        Returns:
            Updated BatchResult
        """
        # Call parent to do the actual processing
        result = super()._process_csv_sequential(rows, config, result)

        # Update CSV rows with new paths (for operations that change files)
        if self._csv_context is not None and self._csv_handler is not None:
            for row in self._csv_context.rows:
                if row.file_path in self._output_paths:
                    new_path = self._output_paths[row.file_path]
                    self._csv_handler.update_row_path(row, new_path, self._csv_context)

        return result

    def _process_csv_parallel(
        self, rows: Any, config: BatchConfig, result: BatchResult
    ) -> BatchResult:
        """Override to update CSV rows during parallel processing.

        Args:
            rows: List of MetadataRow objects to process
            config: Batch configuration
            result: BatchResult to update

        Returns:
            Updated BatchResult
        """
        import sys
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock

        # Thread-safe storage for output paths
        output_paths_lock = Lock()
        local_output_paths: Dict[Path, Path] = {}

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
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
                    futures[executor.submit(self._process_file_safe, row.file_path, output_paths_lock, local_output_paths)] = row

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

        # Update CSV rows with new paths (for operations that change files)
        if self._csv_context is not None and self._csv_handler is not None:
            for row in self._csv_context.rows:
                if row.file_path in local_output_paths:
                    new_path = local_output_paths[row.file_path]
                    self._csv_handler.update_row_path(row, new_path, self._csv_context)

        # Ensure all output is flushed before returning
        sys.stdout.flush()
        sys.stderr.flush()

        return result

    def _process_file_safe(
        self, file_path: Path, lock: Any, output_paths: Dict[Path, Path]
    ) -> Any:
        """Thread-safe wrapper around process_file that collects output paths.

        Args:
            file_path: Path to file to process
            lock: Threading lock for output_paths dict
            output_paths: Dict to collect output paths (thread-safe with lock)

        Returns:
            Result of process_file
        """
        result = self.process_file(file_path)

        # Safely collect output path if in CSV mode
        if self._csv_context is not None and file_path in self._output_paths:
            with lock:
                output_paths[file_path] = self._output_paths[file_path]

        return result

    def convert_batch(
        self,
        config: BatchConfig,
        target_format: str = "wav",
        target_sr: Optional[int] = None,
        target_channels: Optional[int] = None,
        delete_original: bool = False,
    ) -> BatchResult:
        """Convert audio files to target format with optional resampling/channel conversion.

        Args:
            config: Batch processing configuration
            target_format: Target audio format (wav, mp3, flac, ogg)
            target_sr: Optional target sample rate in Hz
            target_channels: Optional target channel count
            delete_original: Delete original files after successful conversion (default: False)

        Returns:
            BatchResult with processing summary
        """
        self._current_operation = "convert"
        self._current_config = {
            "target_format": target_format,
            "target_sr": target_sr,
            "target_channels": target_channels,
            "output_dir": config.output_dir,
            "delete_original": delete_original,
        }
        self._output_paths = {}  # Reset for new batch

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch_auto(config, file_filter=audio_filter)

    def resample_batch(
        self,
        config: BatchConfig,
        target_sr: int = 22050,
    ) -> BatchResult:
        """Resample audio files to target sample rate.

        Args:
            config: Batch processing configuration
            target_sr: Target sample rate in Hz

        Returns:
            BatchResult with processing summary
        """
        self._current_operation = "resample"
        self._current_config = {
            "target_sr": target_sr,
            "output_dir": config.output_dir,
        }
        self._output_paths = {}  # Reset for new batch

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch_auto(config, file_filter=audio_filter)

    def normalize_batch(
        self,
        config: BatchConfig,
        target_db: float = -20.0,
        peak: bool = False,
    ) -> BatchResult:
        """Normalize audio levels in batch.

        Args:
            config: Batch processing configuration
            target_db: Target loudness in dB
            peak: Use peak normalization instead of RMS

        Returns:
            BatchResult with processing summary
        """
        self._current_operation = "normalize"
        self._current_config = {
            "target_db": target_db,
            "peak": peak,
            "output_dir": config.output_dir,
        }
        self._output_paths = {}  # Reset for new batch

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch_auto(config, file_filter=audio_filter)

    def trim_batch(
        self,
        config: BatchConfig,
        start: Optional[float] = None,
        end: Optional[float] = None,
        trim_silence: bool = False,
        silence_threshold_db: float = -40.0,
    ) -> BatchResult:
        """Trim audio files by time or remove silence.

        Args:
            config: Batch processing configuration
            start: Start time in seconds
            end: End time in seconds
            trim_silence: Trim silence from start/end instead
            silence_threshold_db: Silence threshold in dB

        Returns:
            BatchResult with processing summary
        """
        self._current_operation = "trim"
        self._current_config = {
            "start": start,
            "end": end,
            "trim_silence": trim_silence,
            "silence_threshold_db": silence_threshold_db,
            "output_dir": config.output_dir,
        }
        self._output_paths = {}  # Reset for new batch

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch_auto(config, file_filter=audio_filter)

    def segment_batch(
        self,
        config: BatchConfig,
        segment_duration: float,
        overlap: float = 0.0,
    ) -> BatchResult:
        """Segment audio files into chunks.

        For CSV mode, creates multiple output rows per input file (row expansion).

        Args:
            config: Batch processing configuration
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments in seconds

        Returns:
            BatchResult with processing summary
        """
        self._current_operation = "segment"
        self._current_config = {
            "segment_duration": segment_duration,
            "overlap": overlap,
            "output_dir": config.output_dir,
        }
        self._segment_mapping = {}  # Initialize tracking

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        # For CSV mode, we need to override the processing to expand rows BEFORE CSV write
        if config.input_file:
            # CSV mode - manually process with custom post-processing
            result = self._segment_batch_csv(config, audio_filter)
        else:
            # Directory mode - standard processing
            result = self.process_batch_auto(config, file_filter=audio_filter)

        return result

    def _segment_batch_csv(
        self,
        config: BatchConfig,
        file_filter: Optional[Callable[[Path], bool]] = None,
    ) -> BatchResult:
        """Process segment batch in CSV mode with row expansion.

        This is a custom implementation that expands CSV rows BEFORE writing the output CSV.
        """
        from datetime import datetime

        from bioamla.services.batch_csv import BatchCSVHandler

        start_time = datetime.now()
        result = BatchResult(start_time=start_time.isoformat())

        # Initialize CSV handler
        self._csv_handler = BatchCSVHandler(self.file_repository)
        self._csv_context = self._csv_handler.load_csv(config.input_file, config.output_dir)

        # Filter files if filter provided
        rows_to_process = self._csv_context.rows
        if file_filter:
            rows_to_process = [row for row in rows_to_process if file_filter(row.file_path)]

        result.total_files = len(rows_to_process)

        # Process files (sequential only for now)
        result = self._process_csv_sequential(rows_to_process, config, result)

        # BEFORE writing CSV: Expand rows for segments
        new_all_rows = []
        for original_row in self._csv_context.rows:
            if original_row.file_path in self._segment_mapping:
                # File was segmented - expand into multiple rows
                segments = self._segment_mapping[original_row.file_path]
                segment_rows = self._csv_handler.expand_row_for_segments(
                    original_row, segments, self._csv_context
                )
                new_all_rows.extend(segment_rows)
            else:
                # File was not segmented (likely error) - keep original
                new_all_rows.append(original_row)

        # Replace rows in context
        self._csv_context.rows = new_all_rows

        # Add new fieldnames for segment metadata
        new_fields = ["parent_file", "segment_id", "start_time", "end_time", "duration"]
        for field in new_fields:
            if field not in self._csv_context.fieldnames:
                self._csv_context.fieldnames.append(field)

        # NOW write the CSV with expanded rows
        output_csv = self._csv_handler.write_csv(self._csv_context)
        if not config.quiet:
            print(f"Updated metadata CSV written to: {output_csv}")

        # Finalize result
        end_time = datetime.now()
        result.end_time = end_time.isoformat()
        result.duration_seconds = (end_time - start_time).total_seconds()

        return result

    def visualize_batch(
        self,
        config: BatchConfig,
        plot_type: str = "mel",
        show_legend: bool = True,
    ) -> BatchResult:
        """Generate visualizations for audio files.

        Args:
            config: Batch processing configuration
            plot_type: Type of visualization (mel, stft, mfcc, waveform)
            show_legend: If True, show axes, title, and colorbar. If False, clean image only.

        Returns:
            BatchResult with processing summary
        """
        self._current_operation = "visualize"
        self._current_config = {
            "plot_type": plot_type,
            "show_legend": show_legend,
            "output_dir": config.output_dir,
        }
        self._output_paths = {}  # Reset for new batch

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch_auto(config, file_filter=audio_filter)

    def filter_batch(
        self,
        config: BatchConfig,
        lowpass: Optional[float] = None,
        highpass: Optional[float] = None,
        bandpass: Optional[tuple[float, float]] = None,
        order: int = 5,
    ) -> BatchResult:
        """Apply frequency filter to audio files.

        Args:
            config: Batch processing configuration
            lowpass: Lowpass cutoff frequency in Hz
            highpass: Highpass cutoff frequency in Hz
            bandpass: Tuple of (low, high) for bandpass filter
            order: Filter order (default: 5)

        Returns:
            BatchResult with processing summary
        """
        self._current_operation = "filter"
        self._current_config = {
            "lowpass": lowpass,
            "highpass": highpass,
            "bandpass": bandpass,
            "order": order,
            "output_dir": config.output_dir,
        }
        self._output_paths = {}  # Reset for new batch

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch_auto(config, file_filter=audio_filter)

    def denoise_batch(
        self,
        config: BatchConfig,
        strength: float = 1.0,
    ) -> BatchResult:
        """Apply spectral noise reduction to audio files.

        Args:
            config: Batch processing configuration
            strength: Noise reduction strength (0-2, default: 1.0)

        Returns:
            BatchResult with processing summary
        """
        self._current_operation = "denoise"
        self._current_config = {
            "strength": strength,
            "output_dir": config.output_dir,
        }
        self._output_paths = {}  # Reset for new batch

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch_auto(config, file_filter=audio_filter)

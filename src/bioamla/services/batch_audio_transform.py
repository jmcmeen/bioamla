"""Batch audio transformation service."""

from pathlib import Path
from typing import Any, Dict, Optional

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

        # Calculate output path
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
        elif self._current_operation == "segment":
            result = self.audio_transform_service.segment_file(
                str(file_path),
                str(output_dir / file_path.stem),
                duration=self._current_config["segment_duration"],
                overlap=self._current_config.get("overlap", 0.0),
            )
        elif self._current_operation == "visualize":
            output_path = output_path.with_suffix(".png")
            result = self.audio_transform_service.visualize_file(
                str(file_path),
                str(output_path),
                viz_type=self._current_config.get("plot_type", "mel"),
            )
        else:
            raise ValueError(f"Unknown operation: {self._current_operation}")

        if not result.success:
            raise RuntimeError(result.error)

        return result.data

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

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

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

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

    def segment_batch(
        self,
        config: BatchConfig,
        segment_duration: float,
        overlap: float = 0.0,
    ) -> BatchResult:
        """Segment audio files into chunks.

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

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

    def visualize_batch(
        self,
        config: BatchConfig,
        plot_type: str = "mel",
    ) -> BatchResult:
        """Generate visualizations for audio files.

        Args:
            config: Batch processing configuration
            plot_type: Type of visualization (mel, stft, mfcc, waveform)

        Returns:
            BatchResult with processing summary
        """
        self._current_operation = "visualize"
        self._current_config = {
            "plot_type": plot_type,
            "output_dir": config.output_dir,
        }

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

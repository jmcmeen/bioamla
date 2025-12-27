"""Batch detection service."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.batch_base import BatchServiceBase
from bioamla.services.detection import DetectionService


class BatchDetectionService(BatchServiceBase):
    """Service for batch detection operations (energy, ribbit, peaks, accelerating).

    This service delegates to DetectionService for actual file processing,
    following the dependency injection pattern.
    """

    def __init__(
        self,
        file_repository: FileRepositoryProtocol,
        detection_service: DetectionService,
    ) -> None:
        """Initialize batch detection service.

        Args:
            file_repository: File repository for file discovery
            detection_service: Single-file detection service to delegate to
        """
        super().__init__(file_repository)
        self.detection_service = detection_service
        self._current_method: Optional[str] = None
        self._current_params: Dict[str, Any] = {}
        self._all_detections: list = []

    def process_file(self, file_path: Path) -> Any:
        """Process a single audio file by delegating to DetectionService.

        Dispatches to the appropriate method based on _current_method.

        Args:
            file_path: Path to the audio file to process

        Returns:
            Detection result

        Raises:
            ValueError: If method is not set or unknown
            RuntimeError: If the underlying service operation fails
        """
        if self._current_method is None:
            raise ValueError("Detection method not set. Call a batch method first.")

        # Dispatch to appropriate detection method
        if self._current_method == "energy":
            result = self.detection_service.detect_energy(
                str(file_path),
                low_freq=self._current_params.get("low_freq", 500.0),
                high_freq=self._current_params.get("high_freq", 5000.0),
                threshold_db=self._current_params.get("threshold_db", -20.0),
                min_duration=self._current_params.get("min_duration", 0.05),
            )
        elif self._current_method == "ribbit":
            result = self.detection_service.detect_ribbit(
                str(file_path),
                pulse_rate_hz=self._current_params.get("pulse_rate_hz", 10.0),
                pulse_rate_tolerance=self._current_params.get("pulse_rate_tolerance", 0.2),
                low_freq=self._current_params.get("low_freq", 500.0),
                high_freq=self._current_params.get("high_freq", 5000.0),
                window_duration=self._current_params.get("window_duration", 2.0),
                min_score=self._current_params.get("min_score", 0.3),
            )
        elif self._current_method == "peaks":
            result = self.detection_service.detect_peaks(
                str(file_path),
                snr_threshold=self._current_params.get("snr_threshold", 2.0),
                min_peak_distance=self._current_params.get("min_peak_distance", 0.01),
                low_freq=self._current_params.get("low_freq"),
                high_freq=self._current_params.get("high_freq"),
            )
        elif self._current_method == "accelerating":
            result = self.detection_service.detect_accelerating(
                str(file_path),
                min_pulses=self._current_params.get("min_pulses", 5),
                acceleration_threshold=self._current_params.get("acceleration_threshold", 1.5),
                deceleration_threshold=self._current_params.get("deceleration_threshold"),
                low_freq=self._current_params.get("low_freq", 500.0),
                high_freq=self._current_params.get("high_freq", 5000.0),
                window_duration=self._current_params.get("window_duration", 3.0),
            )
        else:
            raise ValueError(f"Unknown detection method: {self._current_method}")

        if not result.success:
            raise RuntimeError(result.error)

        # Collect detections for aggregated output
        self._all_detections.append(result.data.to_dict())

        return result.data

    def _write_aggregated_results(self, output_dir: str) -> None:
        """Write all detection results to a JSON file.

        Args:
            output_dir: Directory to write results to
        """
        if not self._all_detections:
            return

        output_path = Path(output_dir) / f"detections_{self._current_method}.json"
        self.file_repository.mkdir(str(output_path.parent), parents=True)

        content = json.dumps(self._all_detections, indent=2)
        self.file_repository.write_text(output_path, content)

    def detect_energy_batch(
        self,
        config: BatchConfig,
        low_freq: float = 500.0,
        high_freq: float = 5000.0,
        threshold_db: float = -20.0,
        min_duration: float = 0.05,
    ) -> BatchResult:
        """Detect energy peaks in audio files batch-wise.

        Args:
            config: Batch processing configuration
            low_freq: Low frequency bound (Hz)
            high_freq: High frequency bound (Hz)
            threshold_db: Detection threshold (dB)
            min_duration: Minimum detection duration (s)

        Returns:
            BatchResult with processing summary
        """
        self._current_method = "energy"
        self._current_params = {
            "low_freq": low_freq,
            "high_freq": high_freq,
            "threshold_db": threshold_db,
            "min_duration": min_duration,
        }
        self._all_detections = []

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        result = self.process_batch_auto(config, file_filter=audio_filter)
        self._write_aggregated_results(config.output_dir)
        return result

    def detect_ribbit_batch(
        self,
        config: BatchConfig,
        pulse_rate_hz: float = 10.0,
        pulse_rate_tolerance: float = 0.2,
        low_freq: float = 500.0,
        high_freq: float = 5000.0,
        window_duration: float = 2.0,
        min_score: float = 0.3,
    ) -> BatchResult:
        """Detect ribbit calls in audio files batch-wise.

        Args:
            config: Batch processing configuration
            pulse_rate_hz: Expected pulse rate in Hz
            pulse_rate_tolerance: Tolerance around expected pulse rate
            low_freq: Low frequency bound (Hz)
            high_freq: High frequency bound (Hz)
            window_duration: Analysis window duration (s)
            min_score: Minimum detection score

        Returns:
            BatchResult with processing summary
        """
        self._current_method = "ribbit"
        self._current_params = {
            "pulse_rate_hz": pulse_rate_hz,
            "pulse_rate_tolerance": pulse_rate_tolerance,
            "low_freq": low_freq,
            "high_freq": high_freq,
            "window_duration": window_duration,
            "min_score": min_score,
        }
        self._all_detections = []

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        result = self.process_batch_auto(config, file_filter=audio_filter)
        self._write_aggregated_results(config.output_dir)
        return result

    def detect_peaks_batch(
        self,
        config: BatchConfig,
        snr_threshold: float = 2.0,
        min_peak_distance: float = 0.01,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
    ) -> BatchResult:
        """Detect peaks in audio files batch-wise.

        Args:
            config: Batch processing configuration
            snr_threshold: Signal-to-noise ratio threshold
            min_peak_distance: Minimum peak distance (s)
            low_freq: Low frequency bound (Hz)
            high_freq: High frequency bound (Hz)

        Returns:
            BatchResult with processing summary
        """
        self._current_method = "peaks"
        self._current_params = {
            "snr_threshold": snr_threshold,
            "min_peak_distance": min_peak_distance,
            "low_freq": low_freq,
            "high_freq": high_freq,
        }
        self._all_detections = []

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        result = self.process_batch_auto(config, file_filter=audio_filter)
        self._write_aggregated_results(config.output_dir)
        return result

    def detect_accelerating_batch(
        self,
        config: BatchConfig,
        min_pulses: int = 5,
        acceleration_threshold: float = 1.5,
        deceleration_threshold: Optional[float] = None,
        low_freq: float = 500.0,
        high_freq: float = 5000.0,
        window_duration: float = 3.0,
    ) -> BatchResult:
        """Detect accelerating or decelerating call patterns in batch.

        Args:
            config: Batch processing configuration
            min_pulses: Minimum pulses to detect pattern
            acceleration_threshold: Acceleration threshold (final_rate/initial_rate)
            deceleration_threshold: Deceleration threshold (optional)
            low_freq: Low frequency bound (Hz)
            high_freq: High frequency bound (Hz)
            window_duration: Analysis window duration (s)

        Returns:
            BatchResult with processing summary
        """
        self._current_method = "accelerating"
        self._current_params = {
            "min_pulses": min_pulses,
            "acceleration_threshold": acceleration_threshold,
            "deceleration_threshold": deceleration_threshold,
            "low_freq": low_freq,
            "high_freq": high_freq,
            "window_duration": window_duration,
        }
        self._all_detections = []

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        result = self.process_batch_auto(config, file_filter=audio_filter)
        self._write_aggregated_results(config.output_dir)
        return result

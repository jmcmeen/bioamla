"""Batch detection service."""

from pathlib import Path
from typing import Any

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.batch_base import BatchServiceBase


class BatchDetectionService(BatchServiceBase):
    """Service for batch detection operations (energy, ribbit, peaks, accelerating)."""

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize batch detection service."""
        super().__init__(file_repository)

    def process_file(self, file_path: Path) -> Any:
        """Process a single audio file for detection.

        Subclass will override to implement specific detection methods.
        """
        pass

    def detect_energy_batch(self, config: BatchConfig) -> BatchResult:
        """Detect energy peaks in audio files batch-wise."""
        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

    def detect_ribbit_batch(self, config: BatchConfig) -> BatchResult:
        """Detect ribbit calls in audio files batch-wise."""
        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

    def detect_peaks_batch(self, config: BatchConfig) -> BatchResult:
        """Detect peaks in audio files batch-wise."""
        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

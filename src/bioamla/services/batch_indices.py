"""Batch acoustic indices service."""

from pathlib import Path
from typing import Any

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.batch_base import BatchServiceBase


class BatchIndicesService(BatchServiceBase):
    """Service for batch acoustic indices computation."""

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize batch indices service."""
        super().__init__(file_repository)

    def process_file(self, file_path: Path) -> Any:
        """Process a single audio file for indices computation.

        Subclass will override to implement specific indices calculation.
        """
        pass

    def calculate_batch(self, config: BatchConfig) -> BatchResult:
        """Calculate acoustic indices for audio files batch-wise."""
        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

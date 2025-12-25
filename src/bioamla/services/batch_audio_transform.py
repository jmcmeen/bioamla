"""Batch audio transformation service."""

from pathlib import Path
from typing import Any

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.batch_base import BatchServiceBase


class BatchAudioTransformService(BatchServiceBase):
    """Service for batch audio transformations (convert, resample, normalize, segment)."""

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize batch audio transform service."""
        super().__init__(file_repository)

    def process_file(self, file_path: Path) -> Any:
        """Process a single audio file.

        Subclass will override to implement specific transformations.
        """
        pass

    def convert_batch(
        self, config: BatchConfig, output_format: str = "wav"
    ) -> BatchResult:
        """Convert audio files to target format."""
        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

    def resample_batch(self, config: BatchConfig, target_sr: int = 22050) -> BatchResult:
        """Resample audio files to target sample rate."""
        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

    def normalize_batch(self, config: BatchConfig) -> BatchResult:
        """Normalize audio levels in batch."""
        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

    def segment_batch(self, config: BatchConfig, segment_duration: float) -> BatchResult:
        """Segment audio files into chunks."""
        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

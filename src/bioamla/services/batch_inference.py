"""Batch model inference service (AST-only)."""

from pathlib import Path
from typing import Any

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.batch_base import BatchServiceBase


class BatchInferenceService(BatchServiceBase):
    """Service for batch model inference (AST-only for now).

    Once AST is perfected, this will be extended to support other model types.
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize batch inference service."""
        super().__init__(file_repository)

    def process_file(self, file_path: Path) -> Any:
        """Process a single audio file for inference.

        Subclass will override to implement specific model inference.
        """
        pass

    def predict_batch(self, config: BatchConfig, model_path: str) -> BatchResult:
        """Run predictions on audio files batch-wise (AST-only)."""
        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

    def embed_batch(self, config: BatchConfig, model_path: str) -> BatchResult:
        """Extract embeddings from audio files batch-wise (AST-only)."""
        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch(config, file_filter=audio_filter)

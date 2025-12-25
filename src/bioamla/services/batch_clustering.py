"""Batch clustering service."""

from pathlib import Path
from typing import Any

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.batch_base import BatchServiceBase


class BatchClusteringService(BatchServiceBase):
    """Service for batch clustering operations."""

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize batch clustering service."""
        super().__init__(file_repository)

    def process_file(self, file_path: Path) -> Any:
        """Process a single embedding file for clustering.

        Subclass will override to implement specific clustering methods.
        """
        pass

    def cluster_batch(self, config: BatchConfig) -> BatchResult:
        """Cluster embeddings from batch files."""
        def data_filter(path: Path) -> bool:
            data_exts = {".npy", ".pkl", ".pickle", ".json"}
            return path.suffix.lower() in data_exts

        return self.process_batch(config, file_filter=data_filter)

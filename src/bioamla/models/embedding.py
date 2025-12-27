"""Embedding and feature extraction models."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from bioamla.models.base import ToDictMixin


@dataclass
class EmbeddingInfo(ToDictMixin):
    """Information about extracted embeddings."""

    filepath: str
    shape: Tuple[int, ...]
    embedding_dim: int
    num_segments: int
    normalized: bool
    model: str
    layer: str


@dataclass
class BatchEmbeddingSummary(ToDictMixin):
    """Summary of batch embedding extraction."""

    total_files: int
    files_processed: int
    files_failed: int
    embedding_dim: int
    total_embeddings: int
    output_path: Optional[str] = None
    errors: List[str] = field(default_factory=lambda: [])

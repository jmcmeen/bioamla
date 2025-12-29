"""Acoustic indices models."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bioamla.models.base import ToDictMixin


@dataclass
class IndicesResult(ToDictMixin):
    """Result containing computed acoustic indices."""

    indices: Dict[str, Any]
    source_path: Optional[str] = None
    h_spectral: Optional[float] = None
    h_temporal: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = dict(self.indices) if isinstance(self.indices, dict) else {}
        if self.source_path:
            result["filepath"] = self.source_path
        if self.h_spectral is not None:
            result["h_spectral"] = self.h_spectral
        if self.h_temporal is not None:
            result["h_temporal"] = self.h_temporal
        return result


@dataclass
class TemporalIndicesResult(ToDictMixin):
    """Result containing temporal indices analysis."""

    windows: List[Dict[str, Any]]
    source_path: Optional[str] = None
    window_duration: float = 60.0
    hop_duration: float = 60.0
    total_duration: float = 0.0

    @property
    def num_windows(self) -> int:
        """Number of analysis windows."""
        return len(self.windows)


@dataclass
class BatchIndicesResult(ToDictMixin):
    """Result containing batch indices computation."""

    results: List[Dict[str, Any]]
    successful: int = 0
    failed: int = 0
    output_path: Optional[str] = None

    @property
    def total(self) -> int:
        """Total number of files processed."""
        return self.successful + self.failed

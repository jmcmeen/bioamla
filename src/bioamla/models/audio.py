"""Audio data and processing models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from bioamla.models.base import ToDictMixin


@dataclass
class AudioData(ToDictMixin):
    """Container for audio data with metadata.

    This is the primary data transfer object between services.
    AudioFileService produces AudioData, AudioTransformService transforms it,
    and AudioFileService persists it.
    """

    samples: np.ndarray
    sample_rate: int
    channels: int = 1
    source_path: Optional[str] = None
    is_modified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.samples) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Total number of samples."""
        return len(self.samples)

    def copy(self) -> "AudioData":
        """Create a deep copy of the audio data."""
        return AudioData(
            samples=self.samples.copy(),
            sample_rate=self.sample_rate,
            channels=self.channels,
            source_path=self.source_path,
            is_modified=self.is_modified,
            metadata=self.metadata.copy(),
        )

    def mark_modified(self) -> "AudioData":
        """Return a copy marked as modified."""
        copy = self.copy()
        copy.is_modified = True
        return copy


@dataclass
class AudioMetadata(ToDictMixin):
    """Metadata about an audio file."""

    filepath: str
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None
    format: Optional[str] = None


@dataclass
class ProcessedAudio(ToDictMixin):
    """Result of processing an audio file."""

    input_path: str
    output_path: str
    operation: str
    sample_rate: int
    duration_seconds: float


@dataclass
class AnalysisResult(ToDictMixin):
    """Result of audio analysis."""

    filepath: str
    duration_seconds: float
    sample_rate: int
    channels: int
    rms_db: float
    peak_db: float
    silence_ratio: float
    frequency_stats: Dict[str, float]


@dataclass
class BatchResult(ToDictMixin):
    """Result of a batch operation."""

    processed: int
    failed: int
    output_path: Optional[str] = None
    errors: List[str] = field(default_factory=lambda: [])


@dataclass
class TransformResult(ToDictMixin):
    """Result of an audio transform operation."""

    audio: AudioData
    operation: str
    parameters: Dict[str, Any]

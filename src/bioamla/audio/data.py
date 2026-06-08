"""
Audio Data Container
====================

The :class:`AudioData` dataclass — the primary in-memory audio data transfer
object. Reconciles the two historical definitions (``services/audio_file.py``
and ``models/audio.py``) into a single superset: it carries the same fields and
methods as both and includes ``to_dict()`` via :class:`ToDictMixin`.

numpy-only; no heavy/optional dependencies.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class ToDictMixin:
    """
    Mixin that adds ``to_dict()`` to dataclasses.

    Handles nested dataclasses, lists, and common types automatically.
    Override ``_to_dict_extra()`` to add custom serialization logic.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary, handling nested structures."""
        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} is not a dataclass")

        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            result[f.name] = self._serialize_value(value)

        # Allow subclasses to add extra fields
        extra = self._to_dict_extra()
        if extra:
            result.update(extra)

        return result

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value for dict output."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if is_dataclass(value):
            return asdict(value)
        # Fallback for other types
        return str(value)

    def _to_dict_extra(self) -> Optional[Dict[str, Any]]:
        """Override to add extra fields to dict output."""
        return None


@dataclass
class AudioData(ToDictMixin):
    """
    Container for audio data with metadata.

    This is the primary in-memory data transfer object for audio. I/O functions
    in :mod:`bioamla.audio.io` produce and consume ``AudioData``; transforms
    operate on it.

    Attributes:
        samples: Audio samples as a numpy array.
        sample_rate: Sample rate in Hz.
        channels: Number of channels (1 = mono).
        source_path: Path the audio was loaded from, if any.
        is_modified: Whether the samples have been modified since loading.
        metadata: Arbitrary metadata dictionary.
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
    """
    Metadata about an audio file.

    Attributes:
        filepath: Path to the file.
        duration_seconds: Duration in seconds.
        sample_rate: Sample rate in Hz.
        channels: Number of channels.
        bit_depth: Bit depth, if known.
        format: Container/codec format, if known.
    """

    filepath: str
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None
    format: Optional[str] = None


@dataclass
class ProcessedAudio(ToDictMixin):
    """
    Result of processing an audio file.

    Attributes:
        input_path: Source file path.
        output_path: Destination file path.
        operation: Human-readable description of the operation performed.
        sample_rate: Sample rate of the output in Hz.
        duration_seconds: Duration of the output in seconds.
    """

    input_path: str
    output_path: str
    operation: str
    sample_rate: int
    duration_seconds: float


__all__ = ["AudioData", "ToDictMixin", "AudioMetadata", "ProcessedAudio"]

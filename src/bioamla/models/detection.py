"""Detection result models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bioamla.models.base import ToDictMixin


@dataclass
class DetectionInfo(ToDictMixin):
    """Information about a single detection."""

    start_time: float
    end_time: float
    confidence: float
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult(ToDictMixin):
    """Result of detection on a single file."""

    filepath: str
    detector_type: str
    num_detections: int
    detections: List[DetectionInfo] = field(default_factory=list)


@dataclass
class BatchDetectionResult(ToDictMixin):
    """Result of batch detection."""

    total_files: int
    files_with_detections: int
    total_detections: int
    output_dir: Optional[str] = None
    errors: List[str] = field(default_factory=lambda: [])

"""RIBBIT detector models."""

from dataclasses import dataclass, field
from typing import List, Optional

from bioamla.models.base import ToDictMixin


@dataclass
class DetectionSummary(ToDictMixin):
    """Summary of RIBBIT detection results."""

    filepath: str
    profile_name: str
    num_detections: int
    total_detection_time: float
    detection_percentage: float
    duration: float
    processing_time: float


@dataclass
class BatchDetectionSummary(ToDictMixin):
    """Summary of batch RIBBIT detection."""

    total_files: int
    files_with_detections: int
    total_detections: int
    total_duration: float
    total_detection_time: float
    detection_percentage: float
    output_path: Optional[str] = None
    errors: List[str] = field(default_factory=lambda: [])

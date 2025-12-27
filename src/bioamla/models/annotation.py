"""Annotation-related models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bioamla.models.base import ToDictMixin


@dataclass
class AnnotationResult(ToDictMixin):
    """Result of annotation operations."""

    annotations: List[Any]
    file_path: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None


@dataclass
class ClipExtractionResult(ToDictMixin):
    """Result of clip extraction operation."""

    total_clips: int = 0
    extracted_clips: List[str] = field(default_factory=list)
    failed_clips: List[str] = field(default_factory=list)
    output_directory: Optional[str] = None


@dataclass
class MeasurementResult(ToDictMixin):
    """Result of measurement computation."""

    annotation_id: Optional[str] = None
    measurements: Dict[str, float] = field(default_factory=dict)

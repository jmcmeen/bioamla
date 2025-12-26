"""Batch processing configuration and result models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bioamla.services.base import ToDictMixin


@dataclass
class BatchConfig(ToDictMixin):
    """Configuration for batch operations."""

    input_dir: str
    output_dir: str
    recursive: bool = True
    max_workers: int = 1
    continue_on_error: bool = True
    quiet: bool = False
    output_template: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult(ToDictMixin):
    """Generic result of batch processing."""

    total_files: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    output_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

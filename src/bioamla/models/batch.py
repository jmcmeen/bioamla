"""Batch processing configuration and result models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from bioamla.models.base import ToDictMixin


@dataclass
class BatchConfig(ToDictMixin):
    """Configuration for batch operations.

    Supports two input modes (mutually exclusive):
    - Directory mode: Provide input_dir to process all files in directory
    - CSV metadata mode: Provide input_file pointing to CSV with file_name column

    Note: For programmatic usage, validation can be bypassed by setting
    _skip_validation=True. This is useful for testing or advanced use cases.
    """

    input_dir: Optional[str] = None
    input_file: Optional[str] = None
    output_dir: str = ""
    recursive: bool = True
    max_workers: int = 1
    continue_on_error: bool = True
    quiet: bool = False
    output_template: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _skip_validation: bool = False  # Internal flag for testing/advanced usage

    def __post_init__(self) -> None:
        """Validate mutual exclusivity of input_dir and input_file.

        Validation can be skipped by setting _skip_validation=True (for testing/advanced usage).
        """
        if self._skip_validation:
            return

        if self.input_dir is None and self.input_file is None:
            raise ValueError(
                "Either input_dir or input_file must be specified. "
                "For testing or advanced usage, set _skip_validation=True to bypass this check."
            )
        if self.input_dir is not None and self.input_file is not None:
            raise ValueError(
                "input_dir and input_file are mutually exclusive. "
                "For testing or advanced usage, set _skip_validation=True to bypass this check."
            )


@dataclass
class SegmentInfo:
    """Information about a created audio segment."""

    segment_path: Path
    segment_id: int
    start_time: float
    end_time: float
    duration: float


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

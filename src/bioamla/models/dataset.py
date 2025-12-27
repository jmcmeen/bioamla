"""Dataset management and manipulation models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from bioamla.models.base import ToDictMixin


@dataclass
class MergeResult(ToDictMixin):
    """Result of dataset merge operation."""

    datasets_merged: int
    total_files: int
    files_copied: int
    files_skipped: int
    files_converted: int
    output_dir: str


@dataclass
class AugmentResult(ToDictMixin):
    """Result of dataset augmentation."""

    files_processed: int
    files_created: int
    output_dir: str


@dataclass
class LicenseResult(ToDictMixin):
    """Result of license generation."""

    output_path: str
    attributions_count: int
    file_size: int


@dataclass
class BatchLicenseResult(ToDictMixin):
    """Result of batch license generation."""

    datasets_found: int
    datasets_processed: int
    datasets_failed: int
    results: List[Dict[str, Any]] = field(default_factory=list)

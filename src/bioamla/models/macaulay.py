# models/macaulay.py
"""
Data models for Macaulay Library operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import ToDictMixin


@dataclass
class MLRecording(ToDictMixin):
    """Information about a Macaulay Library recording."""

    asset_id: str
    catalog_id: str
    species_code: str
    common_name: str
    scientific_name: str
    rating: int
    duration: Optional[float]
    location: str
    country: str
    user_display_name: str
    download_url: str


@dataclass
class SearchResult(ToDictMixin):
    """Result of a Macaulay Library search."""

    total_results: int
    recordings: List[MLRecording]
    query_params: Dict[str, Any]


@dataclass
class DownloadResult(ToDictMixin):
    """Result of a Macaulay Library download operation."""

    total: int
    downloaded: int
    failed: int
    skipped: int
    output_dir: str
    errors: List[str] = field(default_factory=list)

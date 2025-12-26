# models/xeno_canto.py
"""
Data models for Xeno-canto operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import ToDictMixin


@dataclass
class XCRecording(ToDictMixin):
    """Information about a Xeno-canto recording."""

    id: str
    scientific_name: str
    common_name: str
    quality: str
    sound_type: str
    length: str
    location: str
    country: str
    recordist: str
    url: str
    download_url: str
    license: str


@dataclass
class SearchResult(ToDictMixin):
    """Result of a Xeno-canto search."""

    total_results: int
    recordings: List[XCRecording]
    query_params: Dict[str, Any]


@dataclass
class DownloadResult(ToDictMixin):
    """Result of a Xeno-canto download operation."""

    total: int
    downloaded: int
    failed: int
    skipped: int
    output_dir: str
    errors: List[str] = field(default_factory=list)

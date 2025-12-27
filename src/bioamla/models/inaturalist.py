"""iNaturalist catalog interaction models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from bioamla.models.base import ToDictMixin


@dataclass
class SearchResult(ToDictMixin):
    """Result of an iNaturalist search."""

    total_results: int
    observations: List[Dict[str, Any]]
    query_params: Dict[str, Any]


@dataclass
class DownloadResult(ToDictMixin):
    """Result of an iNaturalist download operation."""

    total_observations: int
    total_sounds: int
    observations_with_multiple_sounds: int
    skipped_existing: int
    failed_downloads: int
    output_dir: str
    metadata_file: str
    errors: List[str] = field(default_factory=lambda: [])


@dataclass
class TaxonInfo(ToDictMixin):
    """Information about a taxon."""

    taxon_id: int
    name: str
    common_name: str
    observation_count: int


@dataclass
class ProjectStats(ToDictMixin):
    """Statistics for an iNaturalist project."""

    id: int
    title: str
    description: str
    slug: str
    observation_count: int
    species_count: int
    observers_count: int
    created_at: str
    project_type: str
    place: str
    url: str


@dataclass
class ObservationInfo(ToDictMixin):
    """Information about a single observation."""

    id: int
    taxon_name: str
    common_name: str
    observed_on: str
    location: str
    place_guess: str
    observer: str
    quality_grade: str
    sounds: List[Dict[str, Any]]
    url: str

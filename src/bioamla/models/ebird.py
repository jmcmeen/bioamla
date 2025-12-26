# models/ebird.py
"""
Data models for eBird operations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import ToDictMixin


@dataclass
class EBirdObservation(ToDictMixin):
    """Information about an eBird observation."""

    species_code: str
    common_name: str
    scientific_name: str
    location_name: str
    observation_date: str
    how_many: Optional[int]
    latitude: Optional[float]
    longitude: Optional[float]


@dataclass
class ValidationResult(ToDictMixin):
    """Result of species validation at a location."""

    species_code: str
    is_valid: bool
    nearby_observations: int
    total_species_in_area: int
    most_recent_observation: Optional[str]


@dataclass
class NearbyResult(ToDictMixin):
    """Result of nearby observations query."""

    observations: List[EBirdObservation]
    total_count: int
    query_params: Dict[str, Any]

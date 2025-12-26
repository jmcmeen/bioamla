# models/species.py
"""
Data models for species operations.
"""

from dataclasses import dataclass

from .base import ToDictMixin


@dataclass
class SpeciesInfo(ToDictMixin):
    """Information about a species."""

    scientific_name: str
    common_name: str
    species_code: str
    family: str
    order: str
    source: str


@dataclass
class SearchMatch(ToDictMixin):
    """A search match result."""

    scientific_name: str
    common_name: str
    species_code: str
    family: str
    score: float

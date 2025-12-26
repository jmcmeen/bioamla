# services/species.py
"""
Service for species name lookup and conversion operations.
"""

from dataclasses import dataclass
from typing import List

from .base import BaseService, ServiceResult, ToDictMixin


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


class SpeciesService(BaseService):
    """
    Service for species lookup operations.

    Provides high-level methods for:
    - Looking up species information
    - Converting between scientific and common names
    - Fuzzy searching for species
    - Managing species cache
    """

    def __init__(self) -> None:
        """Initialize species service."""
        super().__init__()

    def lookup(self, name: str, ebird_only: bool = False) -> ServiceResult[SpeciesInfo]:
        """
        Look up species information by name.

        Args:
            name: Species name (scientific or common)
            ebird_only: If True, only search eBird taxonomy

        Returns:
            Result with species information
        """
        try:
            from bioamla.core.catalogs import species

            info = species.get_species_info(name, ebird_only=ebird_only)
            if not info:
                return ServiceResult.fail(f"Species not found: {name}")

            result = SpeciesInfo(
                scientific_name=info.scientific_name,
                common_name=info.common_name,
                species_code=info.species_code,
                family=info.family,
                order=info.order,
                source=info.source,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Found: {result.scientific_name}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Lookup failed: {e}")

    def scientific_to_common(self, scientific_name: str) -> ServiceResult[str]:
        """
        Convert scientific name to common name.

        Args:
            scientific_name: Scientific name (e.g., "Turdus migratorius")

        Returns:
            Result with common name
        """
        try:
            from bioamla.core.catalogs import species

            common = species.scientific_to_common(scientific_name)
            if not common:
                return ServiceResult.fail(f"No common name found for: {scientific_name}")

            return ServiceResult.ok(
                data=common,
                message=f"{scientific_name} → {common}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Conversion failed: {e}")

    def common_to_scientific(self, common_name: str) -> ServiceResult[str]:
        """
        Convert common name to scientific name.

        Args:
            common_name: Common name (e.g., "American Robin")

        Returns:
            Result with scientific name
        """
        try:
            from bioamla.core.catalogs import species

            scientific = species.common_to_scientific(common_name)
            if not scientific:
                return ServiceResult.fail(f"No scientific name found for: {common_name}")

            return ServiceResult.ok(
                data=scientific,
                message=f"{common_name} → {scientific}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Conversion failed: {e}")

    def search(self, query: str, limit: int = 10) -> ServiceResult[List[SearchMatch]]:
        """
        Fuzzy search for species by name.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Result with list of matching species
        """
        try:
            from bioamla.core.catalogs import species

            results = species.search(query, limit=limit)

            matches = [
                SearchMatch(
                    scientific_name=r["scientific_name"],
                    common_name=r["common_name"],
                    species_code=r["species_code"],
                    family=r["family"],
                    score=r["score"],
                )
                for r in results
            ]

            return ServiceResult.ok(
                data=matches,
                message=f"Found {len(matches)} matching species",
            )
        except Exception as e:
            return ServiceResult.fail(f"Search failed: {e}")

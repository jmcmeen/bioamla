# models/species.py
"""
Data models for species operations.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import ToDictMixin

# API endpoints
EBIRD_TAXONOMY_URL = "https://api.ebird.org/v2/ref/taxonomy/ebird"
INAT_TAXA_URL = "https://api.inaturalist.org/v1/taxa"


@dataclass
class SpeciesInfo(ToDictMixin):
    """
    Information about a species.

    Attributes:
        scientific_name: Full scientific name (Genus species).
        common_name: Common English name.
        species_code: eBird species code.
        taxon_id: Taxonomy ID (for iNaturalist).
        family: Family name.
        order: Order name.
        genus: Genus name.
        species: Species epithet.
        category: Taxonomic category (species, subspecies, etc.).
        source: Data source (ebird, inat, etc.).
    """

    scientific_name: str
    common_name: str = ""
    species_code: str = ""
    taxon_id: Optional[int] = None
    family: str = ""
    order: str = ""
    genus: str = ""
    species: str = ""
    category: str = "species"
    source: str = ""

    @classmethod
    def from_ebird_response(cls, data: Dict[str, Any]) -> "SpeciesInfo":
        """Create SpeciesInfo from eBird taxonomy API response data."""
        sci_name = data.get("sciName", "")
        parts = sci_name.split()

        return cls(
            scientific_name=sci_name,
            common_name=data.get("comName", ""),
            species_code=data.get("speciesCode", ""),
            family=data.get("familyComName", ""),
            order=data.get("order", ""),
            genus=parts[0] if parts else "",
            species=parts[1] if len(parts) > 1 else "",
            category=data.get("category", "species"),
            source="ebird",
        )

    @classmethod
    def from_inat_response(cls, data: Dict[str, Any]) -> "SpeciesInfo":
        """Create SpeciesInfo from iNaturalist taxa API response data."""
        sci_name = data.get("name", "")
        parts = sci_name.split()

        return cls(
            scientific_name=sci_name,
            common_name=data.get("preferred_common_name", ""),
            taxon_id=data.get("id"),
            family=data.get("iconic_taxon_name", ""),
            genus=parts[0] if parts else "",
            species=parts[1] if len(parts) > 1 else "",
            category=data.get("rank", "species"),
            source="inat",
        )


@dataclass
class SearchMatch(ToDictMixin):
    """A search match result with similarity score."""

    scientific_name: str
    common_name: str
    species_code: str
    family: str
    score: float

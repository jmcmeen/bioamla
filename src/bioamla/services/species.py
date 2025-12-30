# services/species.py
"""
Service for species name lookup and conversion operations.

Provides utilities for converting between scientific and common species names
using multiple data sources including eBird taxonomy and iNaturalist.

Features:
- Scientific name to common name conversion
- Common name to scientific name lookup
- Fuzzy matching for approximate name searches
- Support for multiple taxonomies (eBird, iNaturalist)

Example:
    >>> service = SpeciesService()
    >>> result = service.scientific_to_common("Turdus migratorius")
    >>> if result.success:
    ...     print(result.data)  # "American Robin"
"""

import csv
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from bioamla.core.files import TextFile
from bioamla.core.constants import APICache, APIClient, RateLimiter
from bioamla.models.species import (
    EBIRD_TAXONOMY_URL,
    INAT_TAXA_URL,
    SearchMatch,
    SpeciesInfo,
)

from .base import BaseService, ServiceResult

logger = logging.getLogger(__name__)


class SpeciesService(BaseService):
    """
    Service for species lookup operations.

    Provides high-level methods for:
    - Looking up species information
    - Converting between scientific and common names
    - Fuzzy searching for species
    - Managing species taxonomy cache

    Example:
        >>> service = SpeciesService()
        >>> result = service.lookup("American Robin")
        >>> if result.success:
        ...     print(f"{result.data.scientific_name} ({result.data.family})")
    """

    def __init__(self) -> None:
        """Initialize species service with rate limiting and caching."""
        super().__init__()
        # eBird taxonomy is stable, cache for 7 days
        self._cache = APICache(ttl_seconds=7 * 24 * 3600)
        self._rate_limiter = RateLimiter(requests_per_second=1.0)
        self._client = APIClient(
            rate_limiter=self._rate_limiter,
            user_agent="bioamla/1.0 (bioacoustics research tool)",
            cache=self._cache,
        )
        # In-memory taxonomy cache (loaded on demand)
        self._taxonomy_cache: Dict[str, Dict[str, Any]] = {}
        self._taxonomy_loaded: bool = False

    def _normalize_name(self, name: str) -> str:
        """Normalize a species name for comparison."""
        return re.sub(r"[^\w\s]", "", name.lower().strip())

    def _load_ebird_taxonomy(self) -> None:
        """Load eBird taxonomy into memory cache."""
        if self._taxonomy_loaded:
            return

        try:
            # eBird taxonomy API (JSON format) - uses disk cache
            taxa = self._client.get(
                EBIRD_TAXONOMY_URL,
                params={"fmt": "json"},
            )

            for taxon in taxa:
                sci_name = taxon.get("sciName", "")
                common_name = taxon.get("comName", "")
                species_code = taxon.get("speciesCode", "")

                if sci_name:
                    entry = {
                        "scientific_name": sci_name,
                        "common_name": common_name,
                        "species_code": species_code,
                        "family": taxon.get("familyComName", ""),
                        "order": taxon.get("order", ""),
                        "category": taxon.get("category", "species"),
                    }

                    # Index by multiple keys
                    self._taxonomy_cache[self._normalize_name(sci_name)] = entry
                    if common_name:
                        self._taxonomy_cache[self._normalize_name(common_name)] = entry
                    if species_code:
                        self._taxonomy_cache[species_code.lower()] = entry

            self._taxonomy_loaded = True
            logger.info(f"Loaded {len(taxa)} taxa from eBird taxonomy")

        except Exception as e:
            logger.warning(f"Failed to load eBird taxonomy: {e}")

    def _search_inat_taxon(self, name: str) -> Optional[SpeciesInfo]:
        """Search iNaturalist for a taxon."""
        try:
            response = self._client.get(
                INAT_TAXA_URL,
                params={"q": name, "is_active": True, "per_page": 5},
            )

            results = response.get("results", [])
            for taxon in results:
                if taxon.get("rank") in ("species", "subspecies"):
                    return SpeciesInfo.from_inat_response(taxon)
        except Exception as e:
            logger.debug(f"iNaturalist lookup failed: {e}")
        return None

    def lookup(self, name: str, ebird_only: bool = False) -> ServiceResult[SpeciesInfo]:
        """
        Look up species information by name.

        Args:
            name: Species name (scientific, common, or species code).
            ebird_only: If True, only search eBird taxonomy (no iNaturalist fallback).

        Returns:
            ServiceResult with SpeciesInfo on success.

        Example:
            >>> result = service.lookup("American Robin")
            >>> if result.success:
            ...     print(f"{result.data.scientific_name} ({result.data.family})")
        """
        try:
            self._load_ebird_taxonomy()

            # Try multiple lookup strategies
            for lookup_name in [name, self._normalize_name(name), name.lower()]:
                if lookup_name in self._taxonomy_cache:
                    entry = self._taxonomy_cache[lookup_name]
                    sci_name = entry["scientific_name"]
                    parts = sci_name.split()
                    result = SpeciesInfo(
                        scientific_name=sci_name,
                        common_name=entry.get("common_name", ""),
                        species_code=entry.get("species_code", ""),
                        family=entry.get("family", ""),
                        order=entry.get("order", ""),
                        genus=parts[0] if parts else "",
                        species=parts[1] if len(parts) > 1 else "",
                        category=entry.get("category", "species"),
                        source="ebird",
                    )
                    return ServiceResult.ok(
                        data=result,
                        message=f"Found: {result.scientific_name}",
                    )

            # Try iNaturalist fallback (unless ebird_only is set)
            if not ebird_only:
                info = self._search_inat_taxon(name)
                if info:
                    return ServiceResult.ok(
                        data=info,
                        message=f"Found: {info.scientific_name}",
                    )

            return ServiceResult.fail(f"Species not found: {name}")

        except Exception as e:
            logger.error(f"Species lookup failed: {e}")
            return ServiceResult.fail(f"Lookup failed: {e}")

    def scientific_to_common(
        self,
        scientific_name: str,
        fallback_inat: bool = True,
    ) -> ServiceResult[str]:
        """
        Convert a scientific name to its common name.

        Args:
            scientific_name: Scientific name (e.g., "Turdus migratorius").
            fallback_inat: Fall back to iNaturalist if not found in eBird.

        Returns:
            ServiceResult with common name on success.

        Example:
            >>> result = service.scientific_to_common("Turdus migratorius")
            >>> if result.success:
            ...     print(result.data)  # "American Robin"
        """
        try:
            self._load_ebird_taxonomy()

            normalized = self._normalize_name(scientific_name)

            # Check eBird taxonomy
            if normalized in self._taxonomy_cache:
                common = self._taxonomy_cache[normalized].get("common_name")
                if common:
                    return ServiceResult.ok(
                        data=common,
                        message=f"{scientific_name} → {common}",
                    )

            # Try iNaturalist fallback
            if fallback_inat:
                info = self._search_inat_taxon(scientific_name)
                if info and info.common_name:
                    return ServiceResult.ok(
                        data=info.common_name,
                        message=f"{scientific_name} → {info.common_name}",
                    )

            return ServiceResult.fail(f"No common name found for: {scientific_name}")

        except Exception as e:
            logger.error(f"Name conversion failed: {e}")
            return ServiceResult.fail(f"Conversion failed: {e}")

    def common_to_scientific(
        self,
        common_name: str,
        fallback_inat: bool = True,
    ) -> ServiceResult[str]:
        """
        Convert a common name to its scientific name.

        Args:
            common_name: Common name (e.g., "American Robin").
            fallback_inat: Fall back to iNaturalist if not found in eBird.

        Returns:
            ServiceResult with scientific name on success.

        Example:
            >>> result = service.common_to_scientific("American Robin")
            >>> if result.success:
            ...     print(result.data)  # "Turdus migratorius"
        """
        try:
            self._load_ebird_taxonomy()

            normalized = self._normalize_name(common_name)

            # Check eBird taxonomy
            if normalized in self._taxonomy_cache:
                scientific = self._taxonomy_cache[normalized].get("scientific_name")
                if scientific:
                    return ServiceResult.ok(
                        data=scientific,
                        message=f"{common_name} → {scientific}",
                    )

            # Try iNaturalist fallback
            if fallback_inat:
                info = self._search_inat_taxon(common_name)
                if info and info.scientific_name:
                    return ServiceResult.ok(
                        data=info.scientific_name,
                        message=f"{common_name} → {info.scientific_name}",
                    )

            return ServiceResult.fail(f"No scientific name found for: {common_name}")

        except Exception as e:
            logger.error(f"Name conversion failed: {e}")
            return ServiceResult.fail(f"Conversion failed: {e}")

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5,
    ) -> ServiceResult[List[SearchMatch]]:
        """
        Fuzzy search for species by name.

        Args:
            query: Search query (partial name).
            limit: Maximum results to return.
            min_score: Minimum similarity score (0.0-1.0).

        Returns:
            ServiceResult with list of SearchMatch on success.

        Example:
            >>> result = service.search("robin")
            >>> if result.success:
            ...     for match in result.data:
            ...         print(f"{match.common_name} - score: {match.score:.2f}")
        """
        try:
            self._load_ebird_taxonomy()

            query_normalized = self._normalize_name(query)
            matches: List[Tuple[float, SearchMatch]] = []
            seen: set = set()

            for _key, entry in self._taxonomy_cache.items():
                sci_name = entry.get("scientific_name", "")
                if sci_name in seen:
                    continue

                # Calculate similarity scores
                sci_score = SequenceMatcher(
                    None, query_normalized, self._normalize_name(sci_name)
                ).ratio()
                common_score = SequenceMatcher(
                    None, query_normalized, self._normalize_name(entry.get("common_name", ""))
                ).ratio()

                # Also check for substring matches
                if query_normalized in self._normalize_name(sci_name):
                    sci_score = max(sci_score, 0.8)
                if query_normalized in self._normalize_name(entry.get("common_name", "")):
                    common_score = max(common_score, 0.8)

                best_score = max(sci_score, common_score)

                if best_score >= min_score:
                    seen.add(sci_name)
                    matches.append(
                        (
                            best_score,
                            SearchMatch(
                                scientific_name=sci_name,
                                common_name=entry.get("common_name", ""),
                                species_code=entry.get("species_code", ""),
                                family=entry.get("family", ""),
                                score=best_score,
                            ),
                        )
                    )

            # Sort by score descending
            matches.sort(key=lambda x: x[0], reverse=True)
            results = [m[1] for m in matches[:limit]]

            return ServiceResult.ok(
                data=results,
                message=f"Found {len(results)} matching species",
            )

        except Exception as e:
            logger.error(f"Species search failed: {e}")
            return ServiceResult.fail(f"Search failed: {e}")

    def get_species_code(self, name: str) -> ServiceResult[str]:
        """
        Get the eBird species code for a name.

        Args:
            name: Scientific or common name.

        Returns:
            ServiceResult with eBird species code on success.

        Example:
            >>> result = service.get_species_code("American Robin")
            >>> if result.success:
            ...     print(result.data)  # "amerob"
        """
        try:
            self._load_ebird_taxonomy()

            normalized = self._normalize_name(name)
            if normalized in self._taxonomy_cache:
                code = self._taxonomy_cache[normalized].get("species_code")
                if code:
                    return ServiceResult.ok(data=code, message=f"Species code: {code}")

            return ServiceResult.fail(f"No species code found for: {name}")

        except Exception as e:
            logger.error(f"Species code lookup failed: {e}")
            return ServiceResult.fail(f"Lookup failed: {e}")

    def code_to_name(self, species_code: str) -> ServiceResult[Tuple[str, str]]:
        """
        Convert a species code to names.

        Args:
            species_code: eBird species code.

        Returns:
            ServiceResult with tuple of (scientific_name, common_name) on success.

        Example:
            >>> result = service.code_to_name("amerob")
            >>> if result.success:
            ...     sci, common = result.data
            ...     print(f"{sci} ({common})")  # "Turdus migratorius (American Robin)"
        """
        try:
            self._load_ebird_taxonomy()

            code_lower = species_code.lower()
            if code_lower in self._taxonomy_cache:
                entry = self._taxonomy_cache[code_lower]
                sci = entry.get("scientific_name", "")
                common = entry.get("common_name", "")
                return ServiceResult.ok(
                    data=(sci, common),
                    message=f"{species_code} → {sci} ({common})",
                )

            return ServiceResult.fail(f"Unknown species code: {species_code}")

        except Exception as e:
            logger.error(f"Code lookup failed: {e}")
            return ServiceResult.fail(f"Lookup failed: {e}")

    def batch_convert(
        self,
        names: List[str],
        direction: str = "scientific_to_common",
    ) -> ServiceResult[Dict[str, Optional[str]]]:
        """
        Convert a batch of names.

        Args:
            names: List of names to convert.
            direction: "scientific_to_common" or "common_to_scientific".

        Returns:
            ServiceResult with dictionary mapping input names to converted names.

        Example:
            >>> result = service.batch_convert(["Turdus migratorius", "Strix varia"])
            >>> if result.success:
            ...     print(result.data)
            ...     # {'Turdus migratorius': 'American Robin', 'Strix varia': 'Barred Owl'}
        """
        try:
            results: Dict[str, Optional[str]] = {}

            for name in names:
                if direction == "scientific_to_common":
                    result = self.scientific_to_common(name, fallback_inat=False)
                else:
                    result = self.common_to_scientific(name, fallback_inat=False)

                results[name] = result.data if result.success else None

            converted = sum(1 for v in results.values() if v is not None)
            return ServiceResult.ok(
                data=results,
                message=f"Converted {converted}/{len(names)} names",
            )

        except Exception as e:
            logger.error(f"Batch conversion failed: {e}")
            return ServiceResult.fail(f"Batch conversion failed: {e}")

    def validate_name(self, name: str) -> ServiceResult[bool]:
        """
        Check if a species name is valid.

        Args:
            name: Scientific or common name to validate.

        Returns:
            ServiceResult with True if the name is valid.

        Example:
            >>> result = service.validate_name("Turdus migratorius")
            >>> print(result.data)  # True
        """
        result = self.lookup(name, ebird_only=True)
        if result.success:
            return ServiceResult.ok(data=True, message=f"Valid: {name}")
        return ServiceResult.ok(data=False, message=f"Not found: {name}")

    def export_taxonomy(
        self,
        output_path: Union[str, Path],
        format: str = "csv",
        taxa_filter: Optional[str] = None,
    ) -> ServiceResult[Path]:
        """
        Export the loaded taxonomy to a file.

        Args:
            output_path: Path to save the file.
            format: Output format ("csv" or "json").
            taxa_filter: Optional filter (e.g., "Aves" for birds only).

        Returns:
            ServiceResult with Path to the exported file on success.

        Example:
            >>> result = service.export_taxonomy("./birds.csv", taxa_filter="Aves")
            >>> if result.success:
            ...     print(f"Exported to: {result.data}")
        """
        try:
            self._load_ebird_taxonomy()

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Collect unique species
            species: Dict[str, Dict[str, Any]] = {}
            for entry in self._taxonomy_cache.values():
                sci_name = entry.get("scientific_name", "")
                if sci_name and sci_name not in species:
                    if taxa_filter and taxa_filter.lower() not in entry.get("order", "").lower():
                        continue
                    species[sci_name] = entry

            records = list(species.values())

            if format == "json":
                with TextFile(output_path, mode="w", encoding="utf-8") as f:
                    json.dump(records, f.handle, indent=2)
            else:
                if records:
                    with TextFile(output_path, mode="w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f.handle, fieldnames=records[0].keys())
                        writer.writeheader()
                        writer.writerows(records)

            return ServiceResult.ok(
                data=output_path,
                message=f"Exported {len(records)} taxa to {output_path}",
            )

        except Exception as e:
            logger.error(f"Taxonomy export failed: {e}")
            return ServiceResult.fail(f"Export failed: {e}")

    def clear_cache(self) -> ServiceResult[None]:
        """
        Clear the in-memory taxonomy cache.

        This will force the taxonomy to be reloaded from the API
        on the next lookup.

        Returns:
            ServiceResult indicating success.
        """
        self._taxonomy_cache = {}
        self._taxonomy_loaded = False
        return ServiceResult.ok(data=None, message="Taxonomy cache cleared")


def find_species_name(category: str, all_categories: set) -> str:
    """
    Find the species name for a given category.

    If the category is a subspecies (e.g., "Lithobates sphenocephalus utricularius"),
    this will return the matching species name (e.g., "Lithobates sphenocephalus")
    if it exists in the set of all categories.

    Args:
        category: The category name to check.
        all_categories: Set of all known category names.

    Returns:
        The shortest matching species name, or the original category if no match.

    Note:
        This is a standalone utility function, not a service method, as it doesn't
        require API access or instance state.
    """
    if not category:
        return category

    # Find all categories that are prefixes of this category
    matching_species = [c for c in all_categories if category.startswith(c) and c != category]

    if matching_species:
        # Return the shortest matching species (most general)
        return min(matching_species, key=len)

    return category

"""
Species Name Conversion
=======================

Provides utilities for converting between scientific and common species names
using multiple data sources including eBird taxonomy and iNaturalist.

Features:
- Scientific name to common name conversion
- Common name to scientific name lookup
- Fuzzy matching for approximate name searches
- Support for multiple taxonomies (eBird, iNaturalist)
- Caching for offline use

Example:
    >>> from bioamla.api import species
    >>>
    >>> # Convert scientific to common name
    >>> common = species.scientific_to_common("Turdus migratorius")
    >>> print(common)  # "American Robin"
    >>>
    >>> # Convert common to scientific name
    >>> scientific = species.common_to_scientific("American Robin")
    >>> print(scientific)  # "Turdus migratorius"
    >>>
    >>> # Fuzzy search
    >>> results = species.search("robin")
    >>> for r in results:
    ...     print(f"{r['common_name']} - {r['scientific_name']}")
"""

import csv
import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from bioamla.core.base_api import APICache, APIClient, RateLimiter
from bioamla.core.files import TextFile

logger = logging.getLogger(__name__)

# API endpoints
EBIRD_TAXONOMY_URL = "https://api.ebird.org/v2/ref/taxonomy/ebird"
INAT_TAXA_URL = "https://api.inaturalist.org/v1/taxa"

# Default rate limiter and cache
_rate_limiter = RateLimiter(requests_per_second=1.0)
_cache = APICache(
    cache_dir=Path.home() / ".cache" / "bioamla" / "species",
    default_ttl=86400 * 7,  # 1 week
)
_client = APIClient(
    rate_limiter=_rate_limiter,
    cache=_cache,
    user_agent="bioamla/1.0 (bioacoustics research tool)",
)

# In-memory taxonomy cache (loaded on demand)
_taxonomy_cache: Dict[str, Dict[str, Any]] = {}
_taxonomy_loaded: bool = False


@dataclass
class SpeciesInfo:
    """
    Information about a species.

    Attributes:
        scientific_name: Full scientific name (Genus species).
        common_name: Common English name.
        species_code: eBird species code.
        taxon_id: Taxonomy ID.
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scientific_name": self.scientific_name,
            "common_name": self.common_name,
            "species_code": self.species_code,
            "taxon_id": self.taxon_id,
            "family": self.family,
            "order": self.order,
            "genus": self.genus,
            "species": self.species,
            "category": self.category,
            "source": self.source,
        }


def _normalize_name(name: str) -> str:
    """Normalize a species name for comparison."""
    return re.sub(r"[^\w\s]", "", name.lower().strip())


def _load_ebird_taxonomy() -> None:
    """Load eBird taxonomy into memory cache."""
    global _taxonomy_cache, _taxonomy_loaded

    if _taxonomy_loaded:
        return

    cache_key = "ebird_taxonomy_full"
    cached = _cache.get(cache_key)

    if cached:
        _taxonomy_cache = cached
        _taxonomy_loaded = True
        return

    try:
        # eBird taxonomy API (CSV format)
        response = _client.session.get(
            EBIRD_TAXONOMY_URL,
            params={"fmt": "json"},
            timeout=60,
        )
        response.raise_for_status()
        taxa = response.json()

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
                _taxonomy_cache[_normalize_name(sci_name)] = entry
                if common_name:
                    _taxonomy_cache[_normalize_name(common_name)] = entry
                if species_code:
                    _taxonomy_cache[species_code.lower()] = entry

        # Cache for later use
        _cache.set(cache_key, _taxonomy_cache, ttl=86400 * 30)  # 30 days
        _taxonomy_loaded = True
        logger.info(f"Loaded {len(taxa)} taxa from eBird taxonomy")

    except Exception as e:
        logger.warning(f"Failed to load eBird taxonomy: {e}")


def _search_inat_taxon(name: str) -> Optional[SpeciesInfo]:
    """Search iNaturalist for a taxon."""
    try:
        response = _client.get(
            INAT_TAXA_URL,
            params={"q": name, "is_active": True, "per_page": 5},
        )

        results = response.get("results", [])
        for taxon in results:
            if taxon.get("rank") in ("species", "subspecies"):
                return SpeciesInfo(
                    scientific_name=taxon.get("name", ""),
                    common_name=taxon.get("preferred_common_name", ""),
                    taxon_id=taxon.get("id"),
                    family=taxon.get("iconic_taxon_name", ""),
                    category=taxon.get("rank", "species"),
                    source="inat",
                )
    except Exception as e:
        logger.debug(f"iNaturalist lookup failed: {e}")
    return None


def scientific_to_common(
    scientific_name: str,
    fallback_inat: bool = True,
) -> Optional[str]:
    """
    Convert a scientific name to its common name.

    Args:
        scientific_name: Scientific name (e.g., "Turdus migratorius").
        fallback_inat: Fall back to iNaturalist if not found in eBird.

    Returns:
        Common name or None if not found.

    Example:
        >>> scientific_to_common("Turdus migratorius")
        'American Robin'
        >>> scientific_to_common("Strix varia")
        'Barred Owl'
    """
    _load_ebird_taxonomy()

    normalized = _normalize_name(scientific_name)

    # Check eBird taxonomy
    if normalized in _taxonomy_cache:
        return _taxonomy_cache[normalized].get("common_name")

    # Try iNaturalist fallback
    if fallback_inat:
        info = _search_inat_taxon(scientific_name)
        if info and info.common_name:
            return info.common_name

    return None


def common_to_scientific(
    common_name: str,
    fallback_inat: bool = True,
) -> Optional[str]:
    """
    Convert a common name to its scientific name.

    Args:
        common_name: Common name (e.g., "American Robin").
        fallback_inat: Fall back to iNaturalist if not found in eBird.

    Returns:
        Scientific name or None if not found.

    Example:
        >>> common_to_scientific("American Robin")
        'Turdus migratorius'
        >>> common_to_scientific("Barred Owl")
        'Strix varia'
    """
    _load_ebird_taxonomy()

    normalized = _normalize_name(common_name)

    # Check eBird taxonomy
    if normalized in _taxonomy_cache:
        return _taxonomy_cache[normalized].get("scientific_name")

    # Try iNaturalist fallback
    if fallback_inat:
        info = _search_inat_taxon(common_name)
        if info and info.scientific_name:
            return info.scientific_name

    return None


def get_species_info(name: str) -> Optional[SpeciesInfo]:
    """
    Get full species information by name.

    Args:
        name: Scientific name, common name, or species code.

    Returns:
        SpeciesInfo object or None if not found.

    Example:
        >>> info = get_species_info("American Robin")
        >>> print(f"{info.scientific_name} ({info.family})")
        'Turdus migratorius (Thrushes)'
    """
    _load_ebird_taxonomy()

    # Try multiple lookup strategies
    for lookup_name in [name, _normalize_name(name), name.lower()]:
        if lookup_name in _taxonomy_cache:
            entry = _taxonomy_cache[lookup_name]
            sci_name = entry["scientific_name"]
            parts = sci_name.split()
            return SpeciesInfo(
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

    # Try iNaturalist
    info = _search_inat_taxon(name)
    if info:
        return info

    return None


def search(
    query: str,
    limit: int = 10,
    min_score: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Fuzzy search for species by name.

    Args:
        query: Search query (partial name).
        limit: Maximum results to return.
        min_score: Minimum similarity score (0.0-1.0).

    Returns:
        List of matching species with similarity scores.

    Example:
        >>> results = search("robin")
        >>> for r in results:
        ...     print(f"{r['common_name']} - score: {r['score']:.2f}")
    """
    _load_ebird_taxonomy()

    query_normalized = _normalize_name(query)
    matches: List[Tuple[float, Dict[str, Any]]] = []
    seen: set = set()

    for _key, entry in _taxonomy_cache.items():
        sci_name = entry.get("scientific_name", "")
        if sci_name in seen:
            continue

        # Calculate similarity scores
        sci_score = SequenceMatcher(None, query_normalized, _normalize_name(sci_name)).ratio()
        common_score = SequenceMatcher(
            None, query_normalized, _normalize_name(entry.get("common_name", ""))
        ).ratio()

        # Also check for substring matches
        if query_normalized in _normalize_name(sci_name):
            sci_score = max(sci_score, 0.8)
        if query_normalized in _normalize_name(entry.get("common_name", "")):
            common_score = max(common_score, 0.8)

        best_score = max(sci_score, common_score)

        if best_score >= min_score:
            seen.add(sci_name)
            matches.append(
                (
                    best_score,
                    {
                        "scientific_name": sci_name,
                        "common_name": entry.get("common_name", ""),
                        "species_code": entry.get("species_code", ""),
                        "family": entry.get("family", ""),
                        "score": best_score,
                    },
                )
            )

    # Sort by score descending
    matches.sort(key=lambda x: x[0], reverse=True)

    return [m[1] for m in matches[:limit]]


def get_species_code(name: str) -> Optional[str]:
    """
    Get the eBird species code for a name.

    Args:
        name: Scientific or common name.

    Returns:
        eBird species code or None.

    Example:
        >>> get_species_code("American Robin")
        'amerob'
    """
    _load_ebird_taxonomy()

    normalized = _normalize_name(name)
    if normalized in _taxonomy_cache:
        return _taxonomy_cache[normalized].get("species_code")
    return None


def code_to_name(species_code: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert a species code to names.

    Args:
        species_code: eBird species code.

    Returns:
        Tuple of (scientific_name, common_name).

    Example:
        >>> code_to_name("amerob")
        ('Turdus migratorius', 'American Robin')
    """
    _load_ebird_taxonomy()

    code_lower = species_code.lower()
    if code_lower in _taxonomy_cache:
        entry = _taxonomy_cache[code_lower]
        return entry.get("scientific_name"), entry.get("common_name")
    return None, None


def batch_convert(
    names: List[str],
    direction: str = "scientific_to_common",
) -> Dict[str, Optional[str]]:
    """
    Convert a batch of names.

    Args:
        names: List of names to convert.
        direction: "scientific_to_common" or "common_to_scientific".

    Returns:
        Dictionary mapping input names to converted names.

    Example:
        >>> batch_convert(["Turdus migratorius", "Strix varia"])
        {'Turdus migratorius': 'American Robin', 'Strix varia': 'Barred Owl'}
    """
    converter = (
        scientific_to_common if direction == "scientific_to_common" else common_to_scientific
    )
    return {name: converter(name) for name in names}


def validate_name(name: str) -> bool:
    """
    Check if a species name is valid.

    Args:
        name: Scientific or common name to validate.

    Returns:
        True if the name is found in the taxonomy.

    Example:
        >>> validate_name("Turdus migratorius")
        True
        >>> validate_name("Fake Species")
        False
    """
    return get_species_info(name) is not None


def find_species_name(category: str, all_categories: set) -> str:
    """
    Find the species name for a given category.

    If the category is a subspecies (e.g., "Lithobates sphenocephalus utricularius"),
    this will return the matching species name (e.g., "Lithobates sphenocephalus")
    if it exists in the set of all categories.

    Args:
        category: The category name to check
        all_categories: Set of all known category names

    Returns:
        The shortest matching species name, or the original category if no match
    """
    if not category:
        return category

    # Find all categories that are prefixes of this category
    matching_species = [c for c in all_categories if category.startswith(c) and c != category]

    if matching_species:
        # Return the shortest matching species (most general)
        return min(matching_species, key=len)

    return category


def export_taxonomy(
    output_path: Union[str, Path],
    format: str = "csv",
    taxa_filter: Optional[str] = None,
) -> Path:
    """
    Export the loaded taxonomy to a file.

    Args:
        output_path: Path to save the file.
        format: Output format ("csv" or "json").
        taxa_filter: Optional filter (e.g., "Aves" for birds only).

    Returns:
        Path to the exported file.

    Example:
        >>> export_taxonomy("./birds.csv", taxa_filter="Aves")
    """
    _load_ebird_taxonomy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect unique species
    species: Dict[str, Dict[str, Any]] = {}
    for entry in _taxonomy_cache.values():
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

    return output_path


def clear_cache() -> int:
    """
    Clear the species cache.

    Returns:
        Number of cache entries cleared.
    """
    global _taxonomy_cache, _taxonomy_loaded
    _taxonomy_cache = {}
    _taxonomy_loaded = False
    return _cache.clear()

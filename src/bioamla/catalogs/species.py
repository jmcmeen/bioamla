"""Species name lookup and conversion.

Converts between scientific names, common names, and eBird species codes using
the eBird taxonomy (cached in memory) with an optional iNaturalist fallback.

Failures raise :class:`~bioamla.exceptions.SpeciesError`; unknown indices/empty
queries raise :class:`~bioamla.exceptions.InvalidInputError`.
"""

import csv
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from bioamla.catalogs._models import (
    EBIRD_TAXONOMY_URL,
    INAT_TAXA_URL,
    SearchMatch,
    SpeciesInfo,
)
from bioamla.common.http import APICache, APIClient, RateLimiter
from bioamla.exceptions import SpeciesError

logger = logging.getLogger(__name__)

# Module-level HTTP client (eBird taxonomy is stable -> cache for 7 days).
_cache = APICache(ttl_seconds=7 * 24 * 3600)
_rate_limiter = RateLimiter(requests_per_second=1.0)
_client = APIClient(
    rate_limiter=_rate_limiter,
    user_agent="bioamla/1.0 (bioacoustics research tool)",
    cache=_cache,
)

# Module-level in-memory taxonomy cache (must persist across calls).
_taxonomy_cache: dict[str, dict[str, Any]] = {}
_taxonomy_loaded: bool = False


def _normalize_name(name: str) -> str:
    """Normalize a species name for comparison."""
    return re.sub(r"[^\w\s]", "", name.lower().strip())


def _load_ebird_taxonomy() -> None:
    """Load eBird taxonomy into the module-level cache (idempotent).

    Failures are logged and swallowed: the taxonomy may load partially or not
    at all, and callers handle the resulting "not found" outcomes.
    """
    global _taxonomy_loaded
    if _taxonomy_loaded:
        return

    try:
        taxa = _client.get(EBIRD_TAXONOMY_URL, params={"fmt": "json"})
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
                _taxonomy_cache[_normalize_name(sci_name)] = entry
                if common_name:
                    _taxonomy_cache[_normalize_name(common_name)] = entry
                if species_code:
                    _taxonomy_cache[species_code.lower()] = entry
        _taxonomy_loaded = True
        logger.info(f"Loaded {len(taxa)} taxa from eBird taxonomy")
    except Exception as e:
        logger.warning(f"Failed to load eBird taxonomy: {e}")


def _search_inat_taxon(name: str) -> SpeciesInfo | None:
    """Search iNaturalist for a taxon by name; returns None on failure/miss."""
    try:
        response = _client.get(INAT_TAXA_URL, params={"q": name, "is_active": True, "per_page": 5})
        for taxon in response.get("results", []):
            if taxon.get("rank") in ("species", "subspecies"):
                return SpeciesInfo.from_inat_response(taxon)
    except Exception as e:
        logger.debug(f"iNaturalist lookup failed: {e}")
    return None


def clear_taxonomy_cache() -> None:
    """Clear the in-memory taxonomy cache, forcing a reload on next lookup."""
    global _taxonomy_loaded
    _taxonomy_cache.clear()
    _taxonomy_loaded = False


def lookup(name: str, ebird_only: bool = False) -> SpeciesInfo:
    """Look up species information by name.

    Args:
        name: Species name (scientific, common, or species code).
        ebird_only: If True, only search eBird taxonomy (no iNaturalist fallback).

    Raises:
        SpeciesError: if the species is not found.
    """
    _load_ebird_taxonomy()

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

    if not ebird_only:
        info = _search_inat_taxon(name)
        if info:
            return info

    raise SpeciesError(f"Species not found: {name}")


def scientific_to_common(scientific_name: str, fallback_inat: bool = True) -> str:
    """Convert a scientific name to its common name.

    Raises:
        SpeciesError: if no common name is found.
    """
    _load_ebird_taxonomy()

    normalized = _normalize_name(scientific_name)
    if normalized in _taxonomy_cache:
        common = _taxonomy_cache[normalized].get("common_name")
        if common:
            return common

    if fallback_inat:
        info = _search_inat_taxon(scientific_name)
        if info and info.common_name:
            return info.common_name

    raise SpeciesError(f"No common name found for: {scientific_name}")


def common_to_scientific(common_name: str, fallback_inat: bool = True) -> str:
    """Convert a common name to its scientific name.

    Raises:
        SpeciesError: if no scientific name is found.
    """
    _load_ebird_taxonomy()

    normalized = _normalize_name(common_name)
    if normalized in _taxonomy_cache:
        scientific = _taxonomy_cache[normalized].get("scientific_name")
        if scientific:
            return scientific

    if fallback_inat:
        info = _search_inat_taxon(common_name)
        if info and info.scientific_name:
            return info.scientific_name

    raise SpeciesError(f"No scientific name found for: {common_name}")


def search(query: str, limit: int = 10, min_score: float = 0.5) -> list[SearchMatch]:
    """Fuzzy search for species by name.

    Args:
        query: Search query (partial name).
        limit: Maximum results to return.
        min_score: Minimum similarity score (0.0-1.0).

    Returns:
        List of :class:`SearchMatch`, sorted by descending score.
    """
    _load_ebird_taxonomy()

    query_normalized = _normalize_name(query)
    matches: list[tuple[float, SearchMatch]] = []
    seen: set = set()

    for _key, entry in _taxonomy_cache.items():
        sci_name = entry.get("scientific_name", "")
        if sci_name in seen:
            continue

        sci_score = SequenceMatcher(None, query_normalized, _normalize_name(sci_name)).ratio()
        common_score = SequenceMatcher(
            None, query_normalized, _normalize_name(entry.get("common_name", ""))
        ).ratio()

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
                    SearchMatch(
                        scientific_name=sci_name,
                        common_name=entry.get("common_name", ""),
                        species_code=entry.get("species_code", ""),
                        family=entry.get("family", ""),
                        score=best_score,
                    ),
                )
            )

    matches.sort(key=lambda x: x[0], reverse=True)
    return [m[1] for m in matches[:limit]]


def get_species_code(name: str) -> str:
    """Get the eBird species code for a name.

    Raises:
        SpeciesError: if no species code is found.
    """
    _load_ebird_taxonomy()

    normalized = _normalize_name(name)
    if normalized in _taxonomy_cache:
        code = _taxonomy_cache[normalized].get("species_code")
        if code:
            return code

    raise SpeciesError(f"No species code found for: {name}")


def code_to_name(species_code: str) -> tuple[str, str]:
    """Convert an eBird species code to a ``(scientific_name, common_name)`` tuple.

    Raises:
        SpeciesError: if the code is unknown.
    """
    _load_ebird_taxonomy()

    code_lower = species_code.lower()
    if code_lower in _taxonomy_cache:
        entry = _taxonomy_cache[code_lower]
        return entry.get("scientific_name", ""), entry.get("common_name", "")

    raise SpeciesError(f"Unknown species code: {species_code}")


def batch_convert(
    names: list[str],
    direction: str = "scientific_to_common",
) -> dict[str, str | None]:
    """Convert a batch of names.

    Args:
        names: List of names to convert.
        direction: "scientific_to_common" or "common_to_scientific".

    Returns:
        Dict mapping each input name to its converted name (or ``None`` if the
        conversion failed for that name). Does not raise on individual misses.
    """
    results: dict[str, str | None] = {}
    for name in names:
        try:
            if direction == "scientific_to_common":
                results[name] = scientific_to_common(name, fallback_inat=False)
            else:
                results[name] = common_to_scientific(name, fallback_inat=False)
        except SpeciesError:
            results[name] = None
    return results


def validate_name(name: str) -> bool:
    """Return True if a species name is valid (found in eBird taxonomy)."""
    try:
        lookup(name, ebird_only=True)
        return True
    except SpeciesError:
        return False


def export_taxonomy(
    output_path: str | Path,
    format: str = "csv",
    taxa_filter: str | None = None,
) -> Path:
    """Export the loaded taxonomy to a file.

    Args:
        output_path: Path to save the file.
        format: Output format ("csv" or "json").
        taxa_filter: Optional order filter (e.g., "Aves" for birds only).

    Returns:
        Path to the exported file.

    Raises:
        SpeciesError: if the export fails.
    """
    try:
        _load_ebird_taxonomy()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        species: dict[str, dict[str, Any]] = {}
        for entry in _taxonomy_cache.values():
            sci_name = entry.get("scientific_name", "")
            if sci_name and sci_name not in species:
                if taxa_filter and taxa_filter.lower() not in entry.get("order", "").lower():
                    continue
                species[sci_name] = entry

        records = list(species.values())

        if format == "json":
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        else:
            if records:
                with output_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)

        return output_path
    except Exception as e:
        logger.error(f"Taxonomy export failed: {e}")
        raise SpeciesError(f"Export failed: {e}") from e


def find_species_name(category: str, all_categories: set) -> str:
    """Find the species name for a given (possibly subspecies) category.

    If ``category`` is a subspecies (e.g. "Lithobates sphenocephalus utricularius"),
    return the matching species name (e.g. "Lithobates sphenocephalus") when it
    exists in ``all_categories``; otherwise return ``category`` unchanged.
    """
    if not category:
        return category

    matching_species = [c for c in all_categories if category.startswith(c) and c != category]
    if matching_species:
        return min(matching_species, key=len)
    return category

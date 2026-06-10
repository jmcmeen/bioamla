"""Catalog integrations for bioacoustic data sources.

This package provides access to external bioacoustic databases and services,
each exposed as a submodule of plain functions that return data and raise
exceptions on failure:

- :mod:`bioamla.catalogs.inat` — iNaturalist citizen-science observations with audio
- :mod:`bioamla.catalogs.xeno_canto` — Xeno-canto bird sound archive
- :mod:`bioamla.catalogs.macaulay` — Cornell's Macaulay Library multimedia archive
- :mod:`bioamla.catalogs.ebird` — eBird observation data and taxonomy
- :mod:`bioamla.catalogs.species` — taxonomic name lookup / conversion
- :mod:`bioamla.catalogs.huggingface` — HuggingFace Hub model & dataset publishing

All failures raise a :class:`~bioamla.exceptions.CatalogError` (or its subclass
:class:`~bioamla.exceptions.SpeciesError`); bad arguments / missing API keys
raise :class:`~bioamla.exceptions.InvalidInputError`.

Example:
    >>> from bioamla.catalogs import species
    >>> species.scientific_to_common("Turdus migratorius")
    'American Robin'
"""

from bioamla.catalogs import ebird, huggingface, inat, macaulay, species, xeno_canto
from bioamla.catalogs._models import (
    CachedRepo,
    EBirdChecklist,
    EBirdHotspot,
    EBirdObservation,
    INaturalistDownloadResult,
    INaturalistSearchResult,
    MacaulayDownloadResult,
    MacaulaySearchResult,
    MLRecording,
    NearbyResult,
    ObservationInfo,
    ProjectStats,
    PullResult,
    PurgeResult,
    PushResult,
    RegionResult,
    SearchMatch,
    SpeciesInfo,
    TaxonInfo,
    ValidationResult,
    XCRecording,
    XenoCantoDownloadResult,
    XenoCantoSearchResult,
)
from bioamla.catalogs.ebird import EBirdService, match_detections_to_ebird
from bioamla.exceptions import CatalogError, DependencyError, InvalidInputError, SpeciesError

__all__ = [
    # Submodules
    "ebird",
    "inat",
    "macaulay",
    "xeno_canto",
    "species",
    "huggingface",
    # eBird
    "EBirdService",
    "match_detections_to_ebird",
    "EBirdObservation",
    "EBirdChecklist",
    "EBirdHotspot",
    "ValidationResult",
    "NearbyResult",
    "RegionResult",
    # Species
    "SpeciesInfo",
    "SearchMatch",
    # iNaturalist
    "INaturalistSearchResult",
    "INaturalistDownloadResult",
    "TaxonInfo",
    "ProjectStats",
    "ObservationInfo",
    # Macaulay
    "MLRecording",
    "MacaulaySearchResult",
    "MacaulayDownloadResult",
    # Xeno-canto
    "XCRecording",
    "XenoCantoSearchResult",
    "XenoCantoDownloadResult",
    # HuggingFace
    "PushResult",
    "PullResult",
    "CachedRepo",
    "PurgeResult",
    # Exceptions
    "CatalogError",
    "SpeciesError",
    "DependencyError",
    "InvalidInputError",
]

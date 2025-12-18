"""
API Integrations
================

This module provides unified interfaces for querying and downloading audio data
from various bioacoustic databases and citizen science platforms.

Supported APIs:
- Xeno-canto: Bird sounds from around the world
- iNaturalist: Citizen science observations with audio
- Macaulay Library: Cornell Lab of Ornithology's media archive
- Species: Name conversion between scientific and common names

Example:
    >>> from bioamla.api import xeno_canto, species
    >>>
    >>> # Search for bird recordings
    >>> results = xeno_canto.search(species="Turdus migratorius", quality="A")
    >>>
    >>> # Convert species names
    >>> common = species.scientific_to_common("Turdus migratorius")
    >>> print(common)  # "American Robin"
"""

from bioamla.api.base import (
    APICache,
    APIClient,
    RateLimiter,
    cached,
    rate_limited,
)

__all__ = [
    # Base utilities
    "APIClient",
    "APICache",
    "RateLimiter",
    "rate_limited",
    "cached",
]


def __getattr__(name):
    """Lazy import API modules."""
    import importlib

    if name == "xeno_canto":
        return importlib.import_module("bioamla.api.xeno_canto")
    if name == "macaulay":
        return importlib.import_module("bioamla.api.macaulay")
    if name == "species":
        return importlib.import_module("bioamla.api.species")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

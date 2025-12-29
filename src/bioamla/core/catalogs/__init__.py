# core/catalogs/__init__.py
"""
Catalogs Package (Deprecated)
=============================

This package previously contained external API integration code for:
- iNaturalist data access
- Xeno-canto audio search and download
- Macaulay Library integration
- eBird integration
- Species name lookup utilities

All functionality has been migrated to the services layer:
- bioamla.services.inaturalist.INaturalistService
- bioamla.services.xeno_canto.XenoCantoService
- bioamla.services.macaulay.MacaulayService
- bioamla.services.ebird.EBirdService
- bioamla.services.species.SpeciesService

HTTP client utilities have been moved to:
- bioamla.core.http (APIClient, RateLimiter, APICache)

This package is retained for backwards compatibility but may be removed
in a future version.
"""

__all__: list = []

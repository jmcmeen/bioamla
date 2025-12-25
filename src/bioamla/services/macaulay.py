# services/macaulay.py
"""
Service for Macaulay Library audio recording operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import BaseService, ServiceResult, ToDictMixin


@dataclass
class MLRecording(ToDictMixin):
    """Information about a Macaulay Library recording."""

    asset_id: str
    catalog_id: str
    species_code: str
    common_name: str
    scientific_name: str
    rating: int
    duration: Optional[float]
    location: str
    country: str
    user_display_name: str
    download_url: str


@dataclass
class SearchResult(ToDictMixin):
    """Result of a Macaulay Library search."""

    total_results: int
    recordings: List[MLRecording]
    query_params: Dict[str, Any]


@dataclass
class DownloadResult(ToDictMixin):
    """Result of a Macaulay Library download operation."""

    total: int
    downloaded: int
    failed: int
    skipped: int
    output_dir: str
    errors: List[str] = field(default_factory=list)


class MacaulayService(BaseService):
    """
    Service for Macaulay Library operations.

    Provides high-level methods for:
    - Searching for audio recordings
    - Downloading audio files with metadata
    - Managing API cache
    """

    def __init__(self) -> None:
        """Initialize Macaulay Library service."""
        super().__init__()

    def search(
        self,
        species_code: Optional[str] = None,
        scientific_name: Optional[str] = None,
        common_name: Optional[str] = None,
        region: Optional[str] = None,
        country: Optional[str] = None,
        taxon_code: Optional[str] = None,
        hotspot_code: Optional[str] = None,
        min_rating: int = 0,
        max_results: int = 10,
    ) -> ServiceResult[SearchResult]:
        """
        Search for Macaulay Library recordings.

        Args:
            species_code: eBird species code (e.g., amerob)
            scientific_name: Scientific name
            common_name: Common name
            region: Region code (e.g., US-NY)
            country: Country code (e.g., US)
            taxon_code: eBird taxon code for broader searches
            hotspot_code: eBird hotspot code
            min_rating: Minimum quality rating (1-5)
            max_results: Maximum number of results

        Returns:
            Result with search results
        """
        try:
            from bioamla.core.catalogs import macaulay

            results = macaulay.search(
                species_code=species_code,
                scientific_name=scientific_name,
                common_name=common_name,
                region=region,
                country=country,
                taxon_code=taxon_code,
                hotspot_code=hotspot_code,
                media_type="audio",
                min_rating=min_rating,
                count=max_results,
            )

            recordings = [
                MLRecording(
                    asset_id=r.asset_id,
                    catalog_id=r.catalog_id,
                    species_code=r.species_code,
                    common_name=r.common_name,
                    scientific_name=r.scientific_name,
                    rating=r.rating,
                    duration=r.duration,
                    location=r.location,
                    country=r.country,
                    user_display_name=r.user_display_name,
                    download_url=r.download_url,
                )
                for r in results
            ]

            result = SearchResult(
                total_results=len(recordings),
                recordings=recordings,
                query_params={
                    "species_code": species_code,
                    "scientific_name": scientific_name,
                    "common_name": common_name,
                    "region": region,
                    "country": country,
                    "taxon_code": taxon_code,
                    "hotspot_code": hotspot_code,
                    "min_rating": min_rating,
                },
            )

            return ServiceResult.ok(
                data=result,
                message=f"Found {len(recordings)} recordings",
            )
        except Exception as e:
            return ServiceResult.fail(f"Search failed: {e}")

    def download(
        self,
        species_code: Optional[str] = None,
        scientific_name: Optional[str] = None,
        common_name: Optional[str] = None,
        region: Optional[str] = None,
        country: Optional[str] = None,
        taxon_code: Optional[str] = None,
        hotspot_code: Optional[str] = None,
        min_rating: int = 3,
        max_recordings: int = 10,
        output_dir: str = "./ml_recordings",
    ) -> ServiceResult[DownloadResult]:
        """
        Download recordings from Macaulay Library.

        Args:
            species_code: eBird species code (e.g., amerob)
            scientific_name: Scientific name
            common_name: Common name
            region: Region code (e.g., US-NY)
            country: Country code (e.g., US)
            taxon_code: eBird taxon code for broader searches
            hotspot_code: eBird hotspot code
            min_rating: Minimum quality rating (default: 3)
            max_recordings: Maximum recordings to download
            output_dir: Output directory

        Returns:
            Result with download statistics
        """
        try:
            from bioamla.core.catalogs import macaulay

            # First search for recordings
            results = macaulay.search(
                species_code=species_code,
                scientific_name=scientific_name,
                common_name=common_name,
                region=region,
                country=country,
                taxon_code=taxon_code,
                hotspot_code=hotspot_code,
                media_type="audio",
                min_rating=min_rating,
                count=max_recordings,
            )

            if not results:
                return ServiceResult.ok(
                    data=DownloadResult(
                        total=0,
                        downloaded=0,
                        failed=0,
                        skipped=0,
                        output_dir=output_dir,
                    ),
                    message="No recordings found matching criteria",
                )

            # Download recordings
            stats = macaulay.download_assets(
                results,
                output_dir=output_dir,
                verbose=False,
            )

            result = DownloadResult(
                total=stats["total"],
                downloaded=stats["downloaded"],
                failed=stats["failed"],
                skipped=stats.get("skipped", 0),
                output_dir=output_dir,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Downloaded {result.downloaded}/{result.total} recordings",
            )
        except Exception as e:
            return ServiceResult.fail(f"Download failed: {e}")

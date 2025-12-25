# services/xeno_canto.py
"""
Service for Xeno-canto bird recording operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import BaseService, ServiceResult, ToDictMixin


@dataclass
class XCRecording(ToDictMixin):
    """Information about a Xeno-canto recording."""

    id: str
    scientific_name: str
    common_name: str
    quality: str
    sound_type: str
    length: str
    location: str
    country: str
    recordist: str
    url: str
    download_url: str
    license: str


@dataclass
class SearchResult(ToDictMixin):
    """Result of a Xeno-canto search."""

    total_results: int
    recordings: List[XCRecording]
    query_params: Dict[str, Any]


@dataclass
class DownloadResult(ToDictMixin):
    """Result of a Xeno-canto download operation."""

    total: int
    downloaded: int
    failed: int
    skipped: int
    output_dir: str
    errors: List[str] = field(default_factory=list)


class XenoCantoService(BaseService):
    """
    Service for Xeno-canto operations.

    Provides high-level methods for:
    - Searching for bird recordings
    - Downloading audio files with metadata
    - Managing API cache
    """

    def __init__(self) -> None:
        """Initialize Xeno-canto service."""
        super().__init__()

    def search(
        self,
        species: Optional[str] = None,
        genus: Optional[str] = None,
        country: Optional[str] = None,
        quality: Optional[str] = None,
        sound_type: Optional[str] = None,
        max_results: int = 10,
    ) -> ServiceResult[SearchResult]:
        """
        Search for Xeno-canto recordings.

        Args:
            species: Species name (scientific or common)
            genus: Genus name
            country: Country name
            quality: Recording quality (A, B, C, D, E)
            sound_type: Sound type (song, call, etc.)
            max_results: Maximum number of results

        Returns:
            Result with search results
        """
        try:
            from bioamla.core.catalogs import xeno_canto

            results = xeno_canto.search(
                species=species,
                genus=genus,
                country=country,
                quality=quality,
                sound_type=sound_type,
                max_results=max_results,
            )

            recordings = [
                XCRecording(
                    id=r.id,
                    scientific_name=r.scientific_name,
                    common_name=r.common_name,
                    quality=r.quality,
                    sound_type=r.sound_type,
                    length=r.length,
                    location=r.location,
                    country=r.country,
                    recordist=r.recordist,
                    url=r.url,
                    download_url=r.download_url,
                    license=r.license,
                )
                for r in results
            ]

            result = SearchResult(
                total_results=len(recordings),
                recordings=recordings,
                query_params={
                    "species": species,
                    "genus": genus,
                    "country": country,
                    "quality": quality,
                    "sound_type": sound_type,
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
        species: Optional[str] = None,
        genus: Optional[str] = None,
        country: Optional[str] = None,
        quality: Optional[str] = "A",
        max_recordings: int = 10,
        output_dir: str = "./xc_recordings",
        delay: float = 1.0,
    ) -> ServiceResult[DownloadResult]:
        """
        Download recordings from Xeno-canto.

        Args:
            species: Species name (scientific or common)
            genus: Genus name
            country: Country name
            quality: Recording quality filter (default: A)
            max_recordings: Maximum recordings to download
            output_dir: Output directory
            delay: Delay between downloads in seconds

        Returns:
            Result with download statistics
        """
        try:
            from bioamla.core.catalogs import xeno_canto

            # First search for recordings
            results = xeno_canto.search(
                species=species,
                genus=genus,
                country=country,
                quality=quality,
                max_results=max_recordings,
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
            stats = xeno_canto.download_recordings(
                results,
                output_dir=output_dir,
                delay=delay,
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

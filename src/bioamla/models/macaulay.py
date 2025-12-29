# models/macaulay.py
"""
Data models for Macaulay Library operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import ToDictMixin

# Macaulay Library API endpoints
ML_SEARCH_URL = "https://search.macaulaylibrary.org/api/v1/search"
ML_ASSET_URL = "https://cdn.download.ams.birds.cornell.edu/api/v1/asset"


@dataclass
class MLRecording(ToDictMixin):
    """
    Information about a Macaulay Library recording.

    Attributes:
        asset_id: Unique asset identifier.
        catalog_id: Catalog ID (ML number).
        species_code: eBird species code.
        common_name: Common species name.
        scientific_name: Scientific species name.
        media_type: Type of media (audio, video, photo).
        rating: Quality rating (1-5).
        location: Recording location.
        region: Geographic region.
        country: Country name.
        latitude: Latitude coordinate.
        longitude: Longitude coordinate.
        date: Recording date.
        user_display_name: Contributor name.
        download_url: URL to download the asset.
        preview_url: URL to preview the asset.
        duration: Duration in seconds (for audio/video).
    """

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
    media_type: str = "audio"
    region: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    date: str = ""
    preview_url: str = ""

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "MLRecording":
        """Create a recording from Macaulay Library API response data."""
        asset_id = str(data.get("assetId", data.get("catalogId", "")))
        catalog_id = str(data.get("catalogId", ""))
        download_url = data.get("downloadUrl", "")
        if not download_url and asset_id:
            download_url = f"{ML_ASSET_URL}/{asset_id}"

        return cls(
            asset_id=asset_id,
            catalog_id=catalog_id,
            species_code=data.get("speciesCode", ""),
            common_name=data.get("commonName", ""),
            scientific_name=data.get("sciName", ""),
            media_type=data.get("mediaType", "audio"),
            rating=int(float(data.get("rating", 0) or 0)),
            location=data.get("location", ""),
            region=data.get("region", ""),
            country=data.get("country", ""),
            latitude=data.get("latitude"),
            longitude=data.get("longitude"),
            date=data.get("obsDt", ""),
            user_display_name=data.get("userDisplayName", ""),
            download_url=download_url,
            preview_url=data.get("previewUrl", ""),
            duration=data.get("duration"),
        )

    def get_download_url(self) -> str:
        """Get the download URL for this recording."""
        if self.download_url:
            return self.download_url
        return f"{ML_ASSET_URL}/{self.asset_id}"


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

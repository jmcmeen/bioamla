"""
Macaulay Library API Integration
================================

Search and download audio recordings from the Macaulay Library at the
Cornell Lab of Ornithology (macaulaylibrary.org).

The Macaulay Library is one of the world's largest natural sound archives,
containing over 15 million audio, video, and photo specimens.

Note:
    The Macaulay Library uses the eBird API for metadata and a separate
    asset server for media files. Some features may require API keys.

Features:
- Search recordings by species, location, and media type
- Download audio with metadata
- Rate limiting and caching support
- Integration with eBird taxonomy

Example:
    >>> from bioamla.api import macaulay
    >>>
    >>> # Search for recordings
    >>> results = macaulay.search(species_code="amerob", media_type="audio")
    >>> print(f"Found {len(results)} recordings")
    >>>
    >>> # Download recordings
    >>> macaulay.download_assets(results[:5], output_dir="./recordings")
"""

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bioamla.core.base_api import APICache, APIClient, RateLimiter
from bioamla.core.files import TextFile, sanitize_filename

logger = logging.getLogger(__name__)

# Macaulay Library / eBird API endpoints
ML_SEARCH_URL = "https://search.macaulaylibrary.org/api/v1/search"
ML_ASSET_URL = "https://cdn.download.ams.birds.cornell.edu/api/v1/asset"
EBIRD_TAXONOMY_URL = "https://api.ebird.org/v2/ref/taxonomy/ebird"

# Default rate limit: 1 request per second
_rate_limiter = RateLimiter(requests_per_second=1.0, burst_size=2)
_cache = APICache(
    cache_dir=Path.home() / ".cache" / "bioamla" / "macaulay",
    default_ttl=3600,
)
_client = APIClient(
    rate_limiter=_rate_limiter,
    cache=_cache,
    user_agent="bioamla/1.0 (bioacoustics research tool)",
)


@dataclass
class MLAsset:
    """
    Represents a Macaulay Library media asset.

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
        width: Image/video width.
        height: Image/video height.
    """

    asset_id: str
    catalog_id: str = ""
    species_code: str = ""
    common_name: str = ""
    scientific_name: str = ""
    media_type: str = ""
    rating: int = 0
    location: str = ""
    region: str = ""
    country: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    date: str = ""
    user_display_name: str = ""
    download_url: str = ""
    preview_url: str = ""
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "MLAsset":
        """Create an asset from API response data."""
        return cls(
            asset_id=str(data.get("assetId", data.get("catalogId", ""))),
            catalog_id=str(data.get("catalogId", "")),
            species_code=data.get("speciesCode", ""),
            common_name=data.get("commonName", ""),
            scientific_name=data.get("sciName", ""),
            media_type=data.get("mediaType", ""),
            rating=int(data.get("rating", 0)),
            location=data.get("location", ""),
            region=data.get("region", ""),
            country=data.get("country", ""),
            latitude=data.get("latitude"),
            longitude=data.get("longitude"),
            date=data.get("obsDt", ""),
            user_display_name=data.get("userDisplayName", ""),
            download_url=data.get("downloadUrl", ""),
            preview_url=data.get("previewUrl", ""),
            duration=data.get("duration"),
            width=data.get("width"),
            height=data.get("height"),
            _raw=data,
        )

    def get_download_url(self) -> str:
        """Get the download URL for this asset."""
        if self.download_url:
            return self.download_url
        # Construct URL from asset ID
        return f"{ML_ASSET_URL}/{self.asset_id}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "asset_id": self.asset_id,
            "catalog_id": self.catalog_id,
            "species_code": self.species_code,
            "common_name": self.common_name,
            "scientific_name": self.scientific_name,
            "media_type": self.media_type,
            "rating": self.rating,
            "location": self.location,
            "region": self.region,
            "country": self.country,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "date": self.date,
            "user_display_name": self.user_display_name,
            "download_url": self.get_download_url(),
            "preview_url": self.preview_url,
            "duration": self.duration,
        }


def search(
    species_code: Optional[str] = None,
    scientific_name: Optional[str] = None,
    common_name: Optional[str] = None,
    media_type: str = "audio",
    region: Optional[str] = None,
    country: Optional[str] = None,
    min_rating: int = 0,
    taxon_code: Optional[str] = None,
    hotspot_code: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    sort: str = "rating_rank_desc",
    count: int = 100,
    use_cache: bool = True,
) -> List[MLAsset]:
    """
    Search the Macaulay Library for media assets.

    Args:
        species_code: eBird species code (e.g., "amerob" for American Robin).
        scientific_name: Scientific name to search.
        common_name: Common name to search.
        media_type: Media type filter ("audio", "video", "photo", or "all").
        region: Region code (e.g., "US-NY" for New York).
        country: Country code (e.g., "US").
        min_rating: Minimum quality rating (1-5).
        taxon_code: eBird taxon code for broader searches.
        hotspot_code: eBird hotspot code.
        year: Year filter.
        month: Month filter (1-12).
        sort: Sort order (rating_rank_desc, obs_dt_desc, upload_dt_desc).
        count: Maximum results to return.
        use_cache: Whether to use cached results.

    Returns:
        List of MLAsset objects.

    Example:
        >>> # Search by species code
        >>> results = search(species_code="amerob", media_type="audio", min_rating=4)
        >>>
        >>> # Search by scientific name
        >>> results = search(scientific_name="Turdus migratorius", media_type="audio")
        >>>
        >>> # Search by region
        >>> results = search(region="US-NY", media_type="audio", count=50)
    """
    params: Dict[str, Any] = {
        "mediaType": media_type if media_type != "all" else None,
        "sort": sort,
        "count": min(count, 100),  # API max is typically 100
    }

    if species_code:
        params["speciesCode"] = species_code
    if scientific_name:
        params["sciName"] = scientific_name
    if common_name:
        params["commonName"] = common_name
    if region:
        params["region"] = region
    if country:
        params["country"] = country
    if min_rating > 0:
        params["rating"] = min_rating
    if taxon_code:
        params["taxonCode"] = taxon_code
    if hotspot_code:
        params["hotspotCode"] = hotspot_code
    if year:
        params["year"] = year
    if month:
        params["month"] = month

    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}

    if not any(
        k in params
        for k in ["speciesCode", "sciName", "commonName", "region", "taxonCode", "hotspotCode"]
    ):
        raise ValueError(
            "At least one search filter is required (species_code, scientific_name, common_name, region, taxon_code, or hotspot_code)"
        )

    try:
        response = _client.get(ML_SEARCH_URL, params=params, use_cache=use_cache)
    except Exception as e:
        logger.error(f"Macaulay Library API error: {e}")
        raise

    results = response.get("results", {}).get("content", [])
    assets = [MLAsset.from_api_response(item) for item in results]

    return assets


def get_asset(asset_id: str) -> Optional[MLAsset]:
    """
    Get details for a specific asset.

    Args:
        asset_id: Macaulay Library asset ID.

    Returns:
        MLAsset object or None if not found.
    """
    try:
        # Search for specific asset
        response = _client.get(ML_SEARCH_URL, params={"catalogId": asset_id})
        results = response.get("results", {}).get("content", [])
        if results:
            return MLAsset.from_api_response(results[0])
    except Exception as e:
        logger.error(f"Failed to get asset {asset_id}: {e}")
    return None


def download_asset(
    asset: Union[MLAsset, str],
    output_dir: Union[str, Path],
    filename: Optional[str] = None,
    organize_by_species: bool = True,
) -> Optional[Path]:
    """
    Download a single asset.

    Args:
        asset: MLAsset object or asset ID.
        output_dir: Directory to save the file.
        filename: Custom filename (auto-generated if None).
        organize_by_species: Create subdirectory for each species.

    Returns:
        Path to downloaded file or None if failed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get asset details if only ID provided
    if isinstance(asset, str):
        asset = get_asset(asset)
        if asset is None:
            logger.error(f"Asset not found: {asset}")
            return None

    download_url = asset.get_download_url()
    if not download_url:
        logger.error(f"No download URL for asset {asset.asset_id}")
        return None

    # Determine output path
    if organize_by_species and asset.scientific_name:
        species_dir = output_dir / sanitize_filename(asset.scientific_name)
        species_dir.mkdir(exist_ok=True)
        save_dir = species_dir
    else:
        save_dir = output_dir

    # Determine file extension based on media type
    ext_map = {"audio": ".mp3", "video": ".mp4", "photo": ".jpg"}
    ext = ext_map.get(asset.media_type, ".mp3")

    if filename:
        filepath = save_dir / filename
    else:
        safe_name = sanitize_filename(asset.scientific_name or "unknown").replace(" ", "_")
        filepath = save_dir / f"ML{asset.catalog_id}_{safe_name}{ext}"

    try:
        _client.download(download_url, filepath)
        logger.info(f"Downloaded: {filepath.name}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to download {asset.asset_id}: {e}")
        return None


def download_assets(
    assets: List[MLAsset],
    output_dir: Union[str, Path],
    organize_by_species: bool = True,
    create_metadata: bool = True,
    delay: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Download multiple assets with metadata.

    Args:
        assets: List of MLAsset objects.
        output_dir: Directory to save files.
        organize_by_species: Create subdirectory for each species.
        create_metadata: Create metadata CSV file.
        delay: Delay between downloads in seconds.
        verbose: Print progress information.

    Returns:
        Dictionary with download statistics.

    Example:
        >>> results = search(species_code="amerob", media_type="audio", min_rating=4)
        >>> stats = download_assets(results[:10], "./recordings")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": len(assets),
        "downloaded": 0,
        "failed": 0,
        "skipped": 0,
        "metadata_file": None,
    }

    metadata_rows = []

    for i, asset in enumerate(assets, 1):
        if verbose:
            name = asset.scientific_name or asset.common_name or asset.asset_id
            print(f"[{i}/{len(assets)}] Downloading ML{asset.catalog_id} - {name}")

        result = download_asset(
            asset,
            output_dir,
            organize_by_species=organize_by_species,
        )

        if result:
            stats["downloaded"] += 1
            relative_path = result.relative_to(output_dir)

            metadata_rows.append(
                {
                    "file_name": str(relative_path),
                    "ml_id": asset.catalog_id,
                    "asset_id": asset.asset_id,
                    "scientific_name": asset.scientific_name,
                    "common_name": asset.common_name,
                    "species_code": asset.species_code,
                    "contributor": asset.user_display_name,
                    "country": asset.country,
                    "region": asset.region,
                    "location": asset.location,
                    "latitude": asset.latitude,
                    "longitude": asset.longitude,
                    "rating": asset.rating,
                    "media_type": asset.media_type,
                    "date": asset.date,
                    "duration": asset.duration,
                }
            )
        else:
            stats["failed"] += 1

        if i < len(assets):
            time.sleep(delay)

    # Write metadata
    if create_metadata and metadata_rows:
        metadata_path = output_dir / "metadata.csv"
        with TextFile(metadata_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f.handle, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)
        stats["metadata_file"] = str(metadata_path)

    if verbose:
        print("\nDownload complete!")
        print(f"  Downloaded: {stats['downloaded']}/{stats['total']}")
        print(f"  Failed: {stats['failed']}")
        if stats["metadata_file"]:
            print(f"  Metadata: {stats['metadata_file']}")

    return stats


def get_species_count(species_code: str, media_type: str = "audio") -> int:
    """
    Get the total number of assets for a species.

    Args:
        species_code: eBird species code.
        media_type: Media type filter.

    Returns:
        Number of assets available.
    """
    try:
        response = _client.get(
            ML_SEARCH_URL, params={"speciesCode": species_code, "mediaType": media_type, "count": 0}
        )
        return response.get("results", {}).get("count", 0)
    except Exception:
        return 0


def search_audio(
    species_code: Optional[str] = None,
    scientific_name: Optional[str] = None,
    region: Optional[str] = None,
    min_rating: int = 0,
    count: int = 100,
) -> List[MLAsset]:
    """
    Convenience function to search for audio recordings only.

    Args:
        species_code: eBird species code.
        scientific_name: Scientific name.
        region: Region code.
        min_rating: Minimum rating.
        count: Maximum results.

    Returns:
        List of audio assets.
    """
    return search(
        species_code=species_code,
        scientific_name=scientific_name,
        region=region,
        media_type="audio",
        min_rating=min_rating,
        count=count,
    )


def clear_cache() -> int:
    """
    Clear the Macaulay Library API cache.

    Returns:
        Number of cache entries cleared.
    """
    return _cache.clear()

"""
Xeno-canto API Integration
==========================

Search and download bird recordings from Xeno-canto (xeno-canto.org),
the world's largest collection of bird sounds.

Features:
- Search recordings by species, location, quality, and more
- Download recordings with metadata
- Rate limiting and caching support
- Batch download with progress tracking

Example:
    >>> from bioamla.api import xeno_canto
    >>>
    >>> # Search for recordings
    >>> results = xeno_canto.search(species="Turdus migratorius", quality="A")
    >>> print(f"Found {len(results)} recordings")
    >>>
    >>> # Download recordings
    >>> xeno_canto.download_recordings(results[:10], output_dir="./recordings")
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

# Xeno-canto API base URL (v3 requires API key)
XC_API_URL = "https://xeno-canto.org/api/3/recordings"

# API key for authentication (can be set via environment variable or set_api_key())
_api_key: str | None = None


def set_api_key(key: str) -> None:
    """
    Set the Xeno-canto API key for authentication.

    As of API v3, an API key is required for all requests.
    Get your API key from your Xeno-canto account page:
    https://xeno-canto.org/account

    Args:
        key: Your Xeno-canto API key.

    Example:
        >>> from bioamla.api import xeno_canto
        >>> xeno_canto.set_api_key("your-api-key-here")
    """
    global _api_key
    _api_key = key


def get_api_key() -> str | None:
    """
    Get the current API key.

    Returns the API key from (in order of priority):
    1. set_api_key() (runtime setting)
    2. XC_API_KEY environment variable
    3. bioamla config file ([api] xc_api_key)

    Returns:
        The API key or None if not set.
    """
    import os

    # Check runtime setting first
    if _api_key:
        return _api_key

    # Check environment variable
    env_key = os.environ.get("XC_API_KEY")
    if env_key:
        return env_key

    # Check config file
    try:
        from bioamla.core.config import get_config

        config = get_config()
        config_key = config.get("api", "xc_api_key")
        if config_key:
            return config_key
    except Exception:
        pass

    return None


# Default rate limit: 1 request per second (be respectful to the API)
_rate_limiter = RateLimiter(requests_per_second=1.0, burst_size=2)
_cache = APICache(
    cache_dir=Path.home() / ".cache" / "bioamla" / "xeno_canto",
    default_ttl=3600,  # 1 hour
)
_client = APIClient(
    rate_limiter=_rate_limiter,
    cache=_cache,
    user_agent="bioamla/1.0 (bioacoustics research tool)",
)


@dataclass
class XCRecording:
    """
    Represents a Xeno-canto recording.

    Attributes:
        id: Xeno-canto recording ID.
        genus: Genus name.
        species: Species name.
        subspecies: Subspecies name (if any).
        full_name: Full scientific name.
        common_name: English common name.
        recordist: Name of the recordist.
        country: Country where recorded.
        location: Specific location.
        latitude: Latitude coordinate.
        longitude: Longitude coordinate.
        sound_type: Type of vocalization (song, call, etc.).
        quality: Recording quality (A, B, C, D, E).
        length: Recording length in format "m:ss".
        date: Recording date.
        time: Recording time.
        url: URL to the recording page.
        file_url: Direct URL to the audio file.
        license: License code.
        remarks: Recordist remarks.
    """

    id: str
    genus: str = ""
    species: str = ""
    subspecies: str = ""
    full_name: str = ""
    common_name: str = ""
    recordist: str = ""
    country: str = ""
    location: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    sound_type: str = ""
    quality: str = ""
    length: str = ""
    date: str = ""
    time: str = ""
    url: str = ""
    file_url: str = ""
    license: str = ""
    remarks: str = ""
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "XCRecording":
        """Create a recording from API response data."""
        lat = data.get("lat")
        lng = data.get("lng")

        return cls(
            id=data.get("id", ""),
            genus=data.get("gen", ""),
            species=data.get("sp", ""),
            subspecies=data.get("ssp", ""),
            full_name=data.get("en", ""),
            common_name=data.get("en", ""),
            recordist=data.get("rec", ""),
            country=data.get("cnt", ""),
            location=data.get("loc", ""),
            latitude=float(lat) if lat else None,
            longitude=float(lng) if lng else None,
            sound_type=data.get("type", ""),
            quality=data.get("q", ""),
            length=data.get("length", ""),
            date=data.get("date", ""),
            time=data.get("time", ""),
            url=data.get("url", ""),
            file_url=data.get("file", ""),
            license=data.get("lic", ""),
            remarks=data.get("rmk", ""),
            _raw=data,
        )

    @property
    def scientific_name(self) -> str:
        """Full scientific name (Genus species subspecies)."""
        parts = [self.genus, self.species]
        if self.subspecies:
            parts.append(self.subspecies)
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "scientific_name": self.scientific_name,
            "common_name": self.common_name,
            "genus": self.genus,
            "species": self.species,
            "subspecies": self.subspecies,
            "recordist": self.recordist,
            "country": self.country,
            "location": self.location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "sound_type": self.sound_type,
            "quality": self.quality,
            "length": self.length,
            "date": self.date,
            "time": self.time,
            "url": self.url,
            "file_url": self.file_url,
            "license": self.license,
            "remarks": self.remarks,
        }


def search(
    species: Optional[str] = None,
    genus: Optional[str] = None,
    recordist: Optional[str] = None,
    country: Optional[str] = None,
    location: Optional[str] = None,
    quality: Optional[str] = None,
    sound_type: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    box: Optional[tuple] = None,
    since: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    query: Optional[str] = None,
    page: int = 1,
    max_results: Optional[int] = None,
    use_cache: bool = True,
) -> List[XCRecording]:
    """
    Search Xeno-canto for bird recordings.

    Args:
        species: Species name (scientific or common).
        genus: Genus name.
        recordist: Recordist name.
        country: Country name.
        location: Location string.
        quality: Recording quality (A, B, C, D, E or combinations like "A B").
        sound_type: Sound type (song, call, alarm call, flight call, etc.).
        latitude: Latitude for location-based search.
        longitude: Longitude for location-based search.
        box: Bounding box as (lat_min, lat_max, lon_min, lon_max).
        since: Return only recordings uploaded since this date (YYYY-MM-DD).
        year: Year of recording.
        month: Month of recording (1-12).
        query: Raw query string (overrides other parameters).
        page: Page number for pagination.
        max_results: Maximum number of results to return.
        use_cache: Whether to use cached results.

    Returns:
        List of XCRecording objects.

    Example:
        >>> # Search by species name
        >>> results = search(species="Turdus migratorius", quality="A")
        >>>
        >>> # Search by location
        >>> results = search(country="United States", genus="Turdus")
        >>>
        >>> # Search with bounding box
        >>> results = search(box=(35, 45, -90, -70), quality="A B")
    """
    # Build query string
    if query:
        query_str = query
    else:
        parts = []

        if species:
            parts.append(species)
        if genus:
            parts.append(f"gen:{genus}")
        if recordist:
            parts.append(f"rec:{recordist}")
        if country:
            parts.append(f"cnt:{country}")
        if location:
            parts.append(f"loc:{location}")
        if quality:
            parts.append(f"q:{quality}")
        if sound_type:
            parts.append(f"type:{sound_type}")
        if latitude is not None and longitude is not None:
            parts.append(f"lat:{latitude}")
            parts.append(f"lon:{longitude}")
        if box:
            lat_min, lat_max, lon_min, lon_max = box
            parts.append(f"box:{lat_min},{lon_min},{lat_max},{lon_max}")
        if since:
            parts.append(f"since:{since}")
        if year:
            parts.append(f"year:{year}")
        if month:
            parts.append(f"month:{month}")

        query_str = " ".join(parts) if parts else ""

    if not query_str:
        raise ValueError("At least one search parameter is required")

    # Check for API key (required for v3)
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "Xeno-canto API key required. Set via:\n"
            "  - Environment variable: XC_API_KEY\n"
            "  - Python: xeno_canto.set_api_key('your-key')\n"
            "  - CLI: bioamla config set xc_api_key <your-key>\n"
            "Get your API key at: https://xeno-canto.org/account"
        )

    all_recordings: List[XCRecording] = []
    current_page = page
    total_pages = 1

    while current_page <= total_pages:
        params = {"query": query_str, "page": current_page}
        headers = {"X-API-Key": api_key}

        try:
            response = _client.get(XC_API_URL, params=params, headers=headers, use_cache=use_cache)
        except Exception as e:
            logger.error(f"Xeno-canto API error: {e}")
            raise

        # Parse response
        recordings_data = response.get("recordings", [])
        num_recordings = int(response.get("numRecordings", 0))
        num_pages = int(response.get("numPages", 1))
        total_pages = num_pages

        for rec_data in recordings_data:
            recording = XCRecording.from_api_response(rec_data)
            all_recordings.append(recording)

            if max_results and len(all_recordings) >= max_results:
                return all_recordings[:max_results]

        current_page += 1

        # Don't hammer the API
        if current_page <= total_pages:
            time.sleep(0.5)

    return all_recordings


def get_recording(recording_id: str) -> Optional[XCRecording]:
    """
    Get details for a specific recording.

    Args:
        recording_id: Xeno-canto recording ID.

    Returns:
        XCRecording object or None if not found.
    """
    api_key = get_api_key()
    if not api_key:
        logger.error("Xeno-canto API key required. Set XC_API_KEY environment variable.")
        return None

    try:
        headers = {"X-API-Key": api_key}
        response = _client.get(XC_API_URL, params={"query": f"nr:{recording_id}"}, headers=headers)
        recordings = response.get("recordings", [])
        if recordings:
            return XCRecording.from_api_response(recordings[0])
    except Exception as e:
        logger.error(f"Failed to get recording {recording_id}: {e}")
    return None


def download_recording(
    recording: Union[XCRecording, str],
    output_dir: Union[str, Path],
    filename: Optional[str] = None,
    organize_by_species: bool = True,
) -> Optional[Path]:
    """
    Download a single recording.

    Args:
        recording: XCRecording object or recording ID.
        output_dir: Directory to save the file.
        filename: Custom filename (auto-generated if None).
        organize_by_species: Create subdirectory for each species.

    Returns:
        Path to downloaded file or None if failed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get recording details if only ID provided
    if isinstance(recording, str):
        recording = get_recording(recording)
        if recording is None:
            logger.error(f"Recording not found: {recording}")
            return None

    if not recording.file_url:
        logger.error(f"No file URL for recording {recording.id}")
        return None

    # Determine output path
    if organize_by_species:
        species_dir = output_dir / sanitize_filename(recording.scientific_name)
        species_dir.mkdir(exist_ok=True)
        save_dir = species_dir
    else:
        save_dir = output_dir

    if filename:
        filepath = save_dir / filename
    else:
        # Generate filename: XC{id}_{species}.mp3
        safe_name = sanitize_filename(recording.scientific_name).replace(" ", "_")
        filepath = save_dir / f"XC{recording.id}_{safe_name}.mp3"

    try:
        _client.download(recording.file_url, filepath)
        logger.info(f"Downloaded: {filepath.name}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to download {recording.id}: {e}")
        return None


def download_recordings(
    recordings: List[XCRecording],
    output_dir: Union[str, Path],
    organize_by_species: bool = True,
    create_metadata: bool = True,
    delay: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Download multiple recordings with metadata.

    Args:
        recordings: List of XCRecording objects.
        output_dir: Directory to save files.
        organize_by_species: Create subdirectory for each species.
        create_metadata: Create metadata CSV file.
        delay: Delay between downloads in seconds.
        verbose: Print progress information.

    Returns:
        Dictionary with download statistics:
            - total: Total recordings
            - downloaded: Successfully downloaded
            - failed: Failed downloads
            - skipped: Already existing files skipped
            - metadata_file: Path to metadata CSV

    Example:
        >>> results = search(species="Turdus migratorius", quality="A", max_results=10)
        >>> stats = download_recordings(results, "./recordings")
        >>> print(f"Downloaded {stats['downloaded']} of {stats['total']} recordings")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": len(recordings),
        "downloaded": 0,
        "failed": 0,
        "skipped": 0,
        "metadata_file": None,
    }

    metadata_rows = []

    for i, recording in enumerate(recordings, 1):
        if verbose:
            print(
                f"[{i}/{len(recordings)}] Downloading XC{recording.id} - {recording.scientific_name}"
            )

        result = download_recording(
            recording,
            output_dir,
            organize_by_species=organize_by_species,
        )

        if result:
            stats["downloaded"] += 1
            relative_path = result.relative_to(output_dir)

            metadata_rows.append(
                {
                    "file_name": str(relative_path),
                    "xc_id": recording.id,
                    "scientific_name": recording.scientific_name,
                    "common_name": recording.common_name,
                    "recordist": recording.recordist,
                    "country": recording.country,
                    "location": recording.location,
                    "latitude": recording.latitude,
                    "longitude": recording.longitude,
                    "quality": recording.quality,
                    "sound_type": recording.sound_type,
                    "date": recording.date,
                    "length": recording.length,
                    "license": recording.license,
                    "url": recording.url,
                    "remarks": recording.remarks,
                }
            )
        else:
            stats["failed"] += 1

        if i < len(recordings):
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


def get_species_recordings_count(species: str) -> int:
    """
    Get the total number of recordings for a species.

    Args:
        species: Species name (scientific or common).

    Returns:
        Number of recordings available.
    """
    api_key = get_api_key()
    if not api_key:
        return 0

    try:
        headers = {"X-API-Key": api_key}
        response = _client.get(XC_API_URL, params={"query": species}, headers=headers)
        return int(response.get("numRecordings", 0))
    except Exception:
        return 0


def search_by_location(
    latitude: float,
    longitude: float,
    radius_km: float = 50.0,
    quality: Optional[str] = None,
    max_results: Optional[int] = None,
) -> List[XCRecording]:
    """
    Search for recordings near a geographic location.

    Args:
        latitude: Center latitude.
        longitude: Center longitude.
        radius_km: Search radius in kilometers (approximate).
        quality: Recording quality filter.
        max_results: Maximum results to return.

    Returns:
        List of recordings near the location.
    """
    # Convert radius to approximate lat/lon degrees
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude varies by latitude
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * abs(latitude) if latitude != 0 else 111.0)

    box = (
        latitude - lat_delta,
        latitude + lat_delta,
        longitude - lon_delta,
        longitude + lon_delta,
    )

    return search(box=box, quality=quality, max_results=max_results)


def clear_cache() -> int:
    """
    Clear the Xeno-canto API cache.

    Returns:
        Number of cache entries cleared.
    """
    return _cache.clear()

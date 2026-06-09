"""Xeno-canto bird recording search and download.

Xeno-canto (https://xeno-canto.org) is the world's largest collection of bird
sounds. API v3 requires an API key, resolved (in priority order) from:

1. :func:`set_xc_api_key` (runtime override)
2. the ``XC_API_KEY`` environment variable
3. the bioamla config file (``[api] xc_api_key``)

Failures raise :class:`~bioamla.exceptions.CatalogError`; a missing API key or
empty query raises :class:`~bioamla.exceptions.InvalidInputError`.
"""

import logging
import math
import os
import time
from pathlib import Path

from bioamla.catalogs._models import (
    XC_API_URL,
    XCRecording,
    XenoCantoDownloadResult,
    XenoCantoSearchResult,
)
from bioamla.common.files import sanitize_filename
from bioamla.common.http import APIClient, RateLimiter
from bioamla.exceptions import CatalogError, InvalidInputError

logger = logging.getLogger(__name__)

_API_KEY_HELP = (
    "Xeno-canto API key required. Set via:\n"
    "  - Environment variable: XC_API_KEY\n"
    "  - Python: bioamla.catalogs.xeno_canto.set_xc_api_key('your-key')\n"
    "  - CLI: bioamla config set xc_api_key <your-key>\n"
    "Get your API key at: https://xeno-canto.org/account"
)

# Module-level rate limiter + client singletons (shared across calls).
_rate_limiter = RateLimiter(requests_per_second=1.0, burst_size=2)
_client = APIClient(
    rate_limiter=_rate_limiter,
    user_agent="bioamla/1.0 (bioacoustics research tool)",
)

# Runtime-set API key (highest priority).
_api_key: str | None = None


def set_xc_api_key(key: str) -> None:
    """Set the Xeno-canto API key at runtime (highest priority)."""
    global _api_key
    _api_key = key


def get_xc_api_key() -> str | None:
    """Resolve the Xeno-canto API key from runtime, env var, then config.

    Returns None if no key is configured anywhere.
    """
    if _api_key:
        return _api_key

    env_key = os.environ.get("XC_API_KEY")
    if env_key:
        return env_key

    try:
        from bioamla.common.config import get_config

        config = get_config()
        config_key = config.get("api", "xc_api_key")
        if config_key:
            return config_key
    except Exception:
        pass

    return None


def _tag(name: str, value: object) -> str:
    """Format a ``tag:value`` term, double-quoting values that contain spaces.

    The Xeno-canto API v3 rejects a bare space-separated word as an invalid
    free-text term (HTTP 400), so multi-word values like ``cnt:United States``
    must be sent as ``cnt:"United States"``.
    """
    text = str(value)
    if " " in text:
        return f'{name}:"{text}"'
    return f"{name}:{text}"


def _build_query_string(
    species: str | None = None,
    genus: str | None = None,
    recordist: str | None = None,
    country: str | None = None,
    location: str | None = None,
    quality: str | None = None,
    sound_type: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    box: tuple | None = None,
    since: str | None = None,
    year: int | None = None,
    month: int | None = None,
    query: str | None = None,
) -> str:
    """Build the tagged query string for the Xeno-canto API v3."""
    if query:
        return query

    parts: list[str] = []
    if species:
        species_parts = species.strip().split()
        if len(species_parts) >= 2:
            parts.append(_tag("gen", species_parts[0]))
            parts.append(_tag("sp", species_parts[1]))
        else:
            parts.append(_tag("en", species))
    if genus:
        parts.append(_tag("gen", genus))
    if recordist:
        parts.append(_tag("rec", recordist))
    if country:
        parts.append(_tag("cnt", country))
    if location:
        parts.append(_tag("loc", location))
    if quality:
        parts.append(_tag("q", quality))
    if sound_type:
        parts.append(_tag("type", sound_type))
    if latitude is not None and longitude is not None:
        parts.append(_tag("lat", latitude))
        parts.append(_tag("lon", longitude))
    if box:
        lat_min, lat_max, lon_min, lon_max = box
        parts.append(f"box:{lat_min},{lon_min},{lat_max},{lon_max}")
    if since:
        parts.append(_tag("since", since))
    if year:
        parts.append(_tag("year", year))
    if month:
        parts.append(_tag("month", month))

    return " ".join(parts) if parts else ""


def search(
    species: str | None = None,
    genus: str | None = None,
    recordist: str | None = None,
    country: str | None = None,
    location: str | None = None,
    quality: str | None = None,
    sound_type: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    box: tuple | None = None,
    since: str | None = None,
    year: int | None = None,
    month: int | None = None,
    query: str | None = None,
    page: int = 1,
    max_results: int | None = None,
) -> XenoCantoSearchResult:
    """Search Xeno-canto for bird recordings.

    Raises:
        InvalidInputError: if no search parameter is given or the API key is missing.
        CatalogError: on API failure.
    """
    query_str = _build_query_string(
        species=species,
        genus=genus,
        recordist=recordist,
        country=country,
        location=location,
        quality=quality,
        sound_type=sound_type,
        latitude=latitude,
        longitude=longitude,
        box=box,
        since=since,
        year=year,
        month=month,
        query=query,
    )
    if not query_str:
        raise InvalidInputError("At least one search parameter is required")

    api_key = get_xc_api_key()
    if not api_key:
        raise InvalidInputError(_API_KEY_HELP)

    try:
        all_recordings: list[XCRecording] = []
        current_page = page
        total_pages = 1

        while current_page <= total_pages:
            params = {"query": query_str, "page": current_page, "key": api_key}
            response = _client.get(XC_API_URL, params=params)

            recordings_data = response.get("recordings", [])
            total_pages = int(response.get("numPages", 1))

            for rec_data in recordings_data:
                all_recordings.append(XCRecording.from_api_response(rec_data))
                if max_results and len(all_recordings) >= max_results:
                    all_recordings = all_recordings[:max_results]
                    break

            if max_results and len(all_recordings) >= max_results:
                break

            current_page += 1
            if current_page <= total_pages:
                time.sleep(0.5)

        return XenoCantoSearchResult(
            total_results=len(all_recordings),
            recordings=all_recordings,
            query_params={
                "species": species,
                "genus": genus,
                "country": country,
                "quality": quality,
                "sound_type": sound_type,
            },
        )
    except Exception as e:
        logger.error(f"Xeno-canto search failed: {e}")
        raise CatalogError(f"Search failed: {e}") from e


def get_recording(recording_id: str) -> XCRecording:
    """Get details for a specific recording.

    Raises:
        InvalidInputError: if the API key is missing.
        CatalogError: if not found or on API failure.
    """
    api_key = get_xc_api_key()
    if not api_key:
        raise InvalidInputError(_API_KEY_HELP)

    try:
        response = _client.get(XC_API_URL, params={"query": f"nr:{recording_id}", "key": api_key})
        recordings = response.get("recordings", [])
        if recordings:
            return XCRecording.from_api_response(recordings[0])
        raise CatalogError(f"Recording {recording_id} not found")
    except CatalogError:
        raise
    except Exception as e:
        logger.error(f"Failed to get recording {recording_id}: {e}")
        raise CatalogError(f"Failed to get recording: {e}") from e


def download_recording(
    recording: XCRecording | str,
    output_dir: str | Path,
    filename: str | None = None,
    organize_by_species: bool = True,
) -> Path:
    """Download a single recording.

    Args:
        recording: XCRecording object or recording ID.
        output_dir: Directory to save the file.
        filename: Custom filename (auto-generated if None).
        organize_by_species: Create a subdirectory for each species.

    Raises:
        CatalogError: if there is no download URL or the download fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(recording, str):
        recording = get_recording(recording)

    if not recording.download_url:
        raise CatalogError(f"No download URL for recording {recording.id}")

    if organize_by_species:
        species_dir = output_dir / sanitize_filename(recording.scientific_name)
        species_dir.mkdir(exist_ok=True)
        save_dir = species_dir
    else:
        save_dir = output_dir

    if filename:
        filepath = save_dir / filename
    else:
        safe_name = sanitize_filename(recording.scientific_name).replace(" ", "_")
        filepath = save_dir / f"XC{recording.id}_{safe_name}.mp3"

    try:
        _client.download(recording.download_url, filepath)
        logger.info(f"Downloaded: {filepath.name}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to download {recording.id}: {e}")
        raise CatalogError(f"Download failed: {e}") from e


def download(
    species: str | None = None,
    genus: str | None = None,
    country: str | None = None,
    quality: str | None = "A",
    max_recordings: int = 10,
    output_dir: str = "./xc_recordings",
    organize_by_species: bool = True,
    create_metadata: bool = True,
    delay: float = 1.0,
) -> XenoCantoDownloadResult:
    """Search for and download recordings from Xeno-canto.

    Raises:
        InvalidInputError: if no search parameter is given or the API key is missing.
        CatalogError: on API/download failure.
    """
    try:
        search_result = search(
            species=species,
            genus=genus,
            country=country,
            quality=quality,
            max_results=max_recordings,
        )
        recordings = search_result.recordings

        if not recordings:
            return XenoCantoDownloadResult(
                total=0, downloaded=0, failed=0, skipped=0, output_dir=output_dir
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats = {"total": len(recordings), "downloaded": 0, "failed": 0, "skipped": 0}
        errors: list[str] = []
        metadata_rows = []

        for i, recording in enumerate(recordings, 1):
            try:
                filepath = download_recording(
                    recording, output_path, organize_by_species=organize_by_species
                )
                stats["downloaded"] += 1
                relative_path = filepath.relative_to(output_path)
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
            except CatalogError as e:
                stats["failed"] += 1
                errors.append(f"XC{recording.id}: {e}")

            if i < len(recordings):
                time.sleep(delay)

        if create_metadata and metadata_rows:
            import csv

            metadata_path = output_path / "metadata.csv"
            with metadata_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
                writer.writeheader()
                writer.writerows(metadata_rows)

        return XenoCantoDownloadResult(
            total=stats["total"],
            downloaded=stats["downloaded"],
            failed=stats["failed"],
            skipped=stats["skipped"],
            output_dir=output_dir,
            errors=errors,
        )
    except (CatalogError, InvalidInputError):
        raise
    except Exception as e:
        logger.error(f"Download operation failed: {e}")
        raise CatalogError(f"Download failed: {e}") from e


def get_species_recordings_count(species: str) -> int:
    """Get the total number of recordings for a species.

    Raises:
        InvalidInputError: if the API key is missing.
        CatalogError: on API failure.
    """
    api_key = get_xc_api_key()
    if not api_key:
        raise InvalidInputError(_API_KEY_HELP)

    try:
        species_parts = species.strip().split()
        if len(species_parts) >= 2:
            query = f"{_tag('gen', species_parts[0])} {_tag('sp', species_parts[1])}"
        else:
            query = _tag("en", species)

        response = _client.get(XC_API_URL, params={"query": query, "key": api_key})
        return int(response.get("numRecordings", 0))
    except Exception as e:
        logger.error(f"Failed to get recordings count: {e}")
        raise CatalogError(f"Failed to get count: {e}") from e


def search_by_location(
    latitude: float,
    longitude: float,
    radius_km: float = 50.0,
    quality: str | None = None,
    max_results: int | None = None,
) -> XenoCantoSearchResult:
    """Search for recordings near a geographic location.

    Args:
        latitude: Center latitude.
        longitude: Center longitude.
        radius_km: Approximate search radius in kilometers.
        quality: Recording quality filter.
        max_results: Maximum results to return.

    Raises:
        InvalidInputError: if the API key is missing.
        CatalogError: on API failure.
    """
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(latitude)))
    box = (
        latitude - lat_delta,
        latitude + lat_delta,
        longitude - lon_delta,
        longitude + lon_delta,
    )
    return search(box=box, quality=quality, max_results=max_results)

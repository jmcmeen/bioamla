"""Macaulay Library audio recording search and download.

The Macaulay Library (https://macaulaylibrary.org) at the Cornell Lab of
Ornithology is one of the world's largest natural sound archives. It uses eBird
species codes for species identification — use :mod:`bioamla.catalogs.species`
to look up codes.

Failures raise :class:`~bioamla.exceptions.CatalogError`; bad arguments raise
:class:`~bioamla.exceptions.InvalidInputError`.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any

from bioamla.catalogs._models import (
    ML_SEARCH_URL,
    MacaulayDownloadResult,
    MacaulaySearchResult,
    MLRecording,
)
from bioamla.common.files import sanitize_filename
from bioamla.common.http import APIClient, RateLimiter
from bioamla.exceptions import CatalogError, InvalidInputError

logger = logging.getLogger(__name__)

# Module-level rate limiter + client singletons (shared across calls).
_rate_limiter = RateLimiter(requests_per_second=1.0, burst_size=2)
_client = APIClient(
    rate_limiter=_rate_limiter,
    user_agent="bioamla/1.0 (bioacoustics research tool)",
)


def search(
    species_code: str | None = None,
    scientific_name: str | None = None,
    common_name: str | None = None,
    media_type: str = "audio",
    region: str | None = None,
    country: str | None = None,
    taxon_code: str | None = None,
    hotspot_code: str | None = None,
    min_rating: int = 0,
    year: int | None = None,
    month: int | None = None,
    sort: str = "rating_rank_desc",
    max_results: int = 100,
) -> MacaulaySearchResult:
    """Search the Macaulay Library for media assets.

    Requires at least one of: species_code, scientific_name, common_name,
    region, taxon_code, or hotspot_code.

    Raises:
        InvalidInputError: if no search filter is provided.
        CatalogError: on API failure.
    """
    try:
        params: dict[str, Any] = {
            "mediaType": media_type if media_type != "all" else None,
            "sort": sort,
            "count": min(max_results, 100),
        }
        if species_code:
            params["taxonCode"] = species_code
        if scientific_name:
            params["sciName"] = scientific_name
        if common_name:
            params["commonName"] = common_name
        if region:
            params["region"] = region
        if country:
            params["country"] = country
        if taxon_code:
            params["taxonCode"] = taxon_code
        if hotspot_code:
            params["hotspotCode"] = hotspot_code
        if min_rating > 0:
            params["rating"] = min_rating
        if year:
            params["year"] = year
        if month:
            params["month"] = month

        params = {k: v for k, v in params.items() if v is not None}

        if not any(
            k in params for k in ["taxonCode", "sciName", "commonName", "region", "hotspotCode"]
        ):
            raise InvalidInputError(
                "At least one search filter is required (species_code, scientific_name, "
                "common_name, region, taxon_code, or hotspot_code)"
            )

        response = _client.get(ML_SEARCH_URL, params=params)
        results_data = response.get("results", {}).get("content", [])
        recordings = [MLRecording.from_api_response(item) for item in results_data]

        return MacaulaySearchResult(
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
    except (CatalogError, InvalidInputError):
        raise
    except Exception as e:
        logger.error(f"Macaulay Library search failed: {e}")
        raise CatalogError(f"Search failed: {e}") from e


def get_recording(asset_id: str) -> MLRecording:
    """Get details for a specific recording.

    Raises:
        CatalogError: if not found or on API failure.
    """
    try:
        response = _client.get(ML_SEARCH_URL, params={"catalogId": asset_id})
        results = response.get("results", {}).get("content", [])
        if results:
            return MLRecording.from_api_response(results[0])
        raise CatalogError(f"Recording {asset_id} not found")
    except CatalogError:
        raise
    except Exception as e:
        logger.error(f"Failed to get recording {asset_id}: {e}")
        raise CatalogError(f"Failed to get recording: {e}") from e


def download_recording(
    recording: MLRecording | str,
    output_dir: str | Path,
    filename: str | None = None,
    organize_by_species: bool = True,
) -> Path:
    """Download a single recording.

    Args:
        recording: MLRecording object or asset ID.
        output_dir: Directory to save the file.
        filename: Custom filename (auto-generated if None).
        organize_by_species: Create a subdirectory for each species.

    Raises:
        CatalogError: if the recording has no download URL or the download fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(recording, str):
        recording = get_recording(recording)

    download_url = recording.get_download_url()
    if not download_url:
        raise CatalogError(f"No download URL for recording {recording.asset_id}")

    if organize_by_species and recording.scientific_name:
        species_dir = output_dir / sanitize_filename(recording.scientific_name)
        species_dir.mkdir(exist_ok=True)
        save_dir = species_dir
    else:
        save_dir = output_dir

    ext_map = {"audio": ".mp3", "video": ".mp4", "photo": ".jpg"}
    ext = ext_map.get(recording.media_type, ".mp3")

    if filename:
        filepath = save_dir / filename
    else:
        safe_name = sanitize_filename(recording.scientific_name or "unknown").replace(" ", "_")
        filepath = save_dir / f"ML{recording.catalog_id}_{safe_name}{ext}"

    try:
        _client.download(download_url, filepath)
        logger.info(f"Downloaded: {filepath.name}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to download {recording.asset_id}: {e}")
        raise CatalogError(f"Download failed: {e}") from e


def download(
    species_code: str | None = None,
    scientific_name: str | None = None,
    common_name: str | None = None,
    region: str | None = None,
    country: str | None = None,
    taxon_code: str | None = None,
    hotspot_code: str | None = None,
    min_rating: int = 3,
    max_recordings: int = 10,
    output_dir: str = "./ml_recordings",
    organize_by_species: bool = True,
    create_metadata: bool = True,
    delay: float = 1.0,
) -> MacaulayDownloadResult:
    """Search for and download recordings from the Macaulay Library.

    Raises:
        InvalidInputError: if no search filter is provided.
        CatalogError: on API/download failure.
    """
    try:
        search_result = search(
            species_code=species_code,
            scientific_name=scientific_name,
            common_name=common_name,
            region=region,
            country=country,
            taxon_code=taxon_code,
            hotspot_code=hotspot_code,
            min_rating=min_rating,
            max_results=max_recordings,
        )
        recordings = search_result.recordings

        if not recordings:
            return MacaulayDownloadResult(
                total=0, downloaded=0, failed=0, skipped=0, output_dir=output_dir
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats = {"total": len(recordings), "downloaded": 0, "failed": 0, "skipped": 0}
        errors: list[str] = []
        metadata_rows: list[dict[str, Any]] = []

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
                        "ml_id": recording.catalog_id,
                        "asset_id": recording.asset_id,
                        "scientific_name": recording.scientific_name,
                        "common_name": recording.common_name,
                        "species_code": recording.species_code,
                        "contributor": recording.user_display_name,
                        "country": recording.country,
                        "region": recording.region,
                        "location": recording.location,
                        "latitude": recording.latitude,
                        "longitude": recording.longitude,
                        "rating": recording.rating,
                        "media_type": recording.media_type,
                        "date": recording.date,
                        "duration": recording.duration,
                    }
                )
            except CatalogError as e:
                stats["failed"] += 1
                errors.append(f"ML{recording.catalog_id}: {e}")

            if i < len(recordings):
                time.sleep(delay)

        if create_metadata and metadata_rows:
            metadata_path = output_path / "metadata.csv"
            with metadata_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
                writer.writeheader()
                writer.writerows(metadata_rows)

        return MacaulayDownloadResult(
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


def get_species_count(species_code: str, media_type: str = "audio") -> int:
    """Get the total number of recordings for a species.

    Raises:
        CatalogError: on API failure.
    """
    try:
        response = _client.get(
            ML_SEARCH_URL,
            params={"speciesCode": species_code, "mediaType": media_type, "count": 0},
        )
        return response.get("results", {}).get("count", 0)
    except Exception as e:
        logger.error(f"Failed to get species count: {e}")
        raise CatalogError(f"Failed to get count: {e}") from e


def search_audio(
    species_code: str | None = None,
    scientific_name: str | None = None,
    region: str | None = None,
    min_rating: int = 0,
    max_results: int = 100,
) -> MacaulaySearchResult:
    """Convenience wrapper around :func:`search` for audio recordings only."""
    return search(
        species_code=species_code,
        scientific_name=scientific_name,
        region=region,
        media_type="audio",
        min_rating=min_rating,
        max_results=max_results,
    )

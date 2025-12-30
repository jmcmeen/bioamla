# services/macaulay.py
"""
Service for Macaulay Library audio recording operations.

The Macaulay Library (macaulaylibrary.org) at Cornell Lab of Ornithology
is one of the world's largest natural sound archives, containing over
15 million audio, video, and photo specimens.

Note:
    The Macaulay Library uses eBird species codes for species identification.
    Use SpeciesService to look up species codes.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bioamla.core.files import TextFile, sanitize_filename
from bioamla.core.constants import APIClient, RateLimiter
from bioamla.models.macaulay import (
    ML_SEARCH_URL,
    DownloadResult,
    MLRecording,
    SearchResult,
)

from .base import BaseService, ServiceResult

logger = logging.getLogger(__name__)


class MacaulayService(BaseService):
    """
    Service for Macaulay Library operations.

    Provides high-level methods for:
    - Searching for audio recordings
    - Downloading audio files with metadata
    - Managing API rate limiting

    Example:
        >>> service = MacaulayService()
        >>> result = service.search(species_code="amerob", min_rating=4)
        >>> if result.success:
        ...     print(f"Found {result.data.total_results} recordings")
    """

    def __init__(self) -> None:
        """Initialize Macaulay Library service with rate limiting."""
        super().__init__()
        self._rate_limiter = RateLimiter(requests_per_second=1.0, burst_size=2)
        self._client = APIClient(
            rate_limiter=self._rate_limiter,
            user_agent="bioamla/1.0 (bioacoustics research tool)",
        )

    def search(
        self,
        species_code: Optional[str] = None,
        scientific_name: Optional[str] = None,
        common_name: Optional[str] = None,
        media_type: str = "audio",
        region: Optional[str] = None,
        country: Optional[str] = None,
        taxon_code: Optional[str] = None,
        hotspot_code: Optional[str] = None,
        min_rating: int = 0,
        year: Optional[int] = None,
        month: Optional[int] = None,
        sort: str = "rating_rank_desc",
        max_results: int = 100,
    ) -> ServiceResult[SearchResult]:
        """
        Search the Macaulay Library for media assets.

        Args:
            species_code: eBird species code (e.g., "amerob" for American Robin).
            scientific_name: Scientific name to search.
            common_name: Common name to search.
            media_type: Media type filter ("audio", "video", "photo", or "all").
            region: Region code (e.g., "US-NY" for New York).
            country: Country code (e.g., "US").
            taxon_code: eBird taxon code for broader searches.
            hotspot_code: eBird hotspot code.
            min_rating: Minimum quality rating (1-5).
            year: Year filter.
            month: Month filter (1-12).
            sort: Sort order (rating_rank_desc, obs_dt_desc, upload_dt_desc).
            max_results: Maximum results to return.

        Returns:
            ServiceResult with SearchResult on success.
        """
        try:
            params: Dict[str, Any] = {
                "mediaType": media_type if media_type != "all" else None,
                "sort": sort,
                "count": min(max_results, 100),  # API max is typically 100
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

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            if not any(
                k in params
                for k in ["taxonCode", "sciName", "commonName", "region", "hotspotCode"]
            ):
                return ServiceResult.fail(
                    "At least one search filter is required (species_code, scientific_name, "
                    "common_name, region, taxon_code, or hotspot_code)"
                )

            response = self._client.get(ML_SEARCH_URL, params=params)

            results_data = response.get("results", {}).get("content", [])
            recordings = [MLRecording.from_api_response(item) for item in results_data]

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
            logger.error(f"Macaulay Library search failed: {e}")
            return ServiceResult.fail(f"Search failed: {e}")

    def get_recording(self, asset_id: str) -> ServiceResult[MLRecording]:
        """
        Get details for a specific recording.

        Args:
            asset_id: Macaulay Library asset ID.

        Returns:
            ServiceResult with MLRecording on success.
        """
        try:
            response = self._client.get(ML_SEARCH_URL, params={"catalogId": asset_id})
            results = response.get("results", {}).get("content", [])
            if results:
                recording = MLRecording.from_api_response(results[0])
                return ServiceResult.ok(data=recording)
            return ServiceResult.fail(f"Recording {asset_id} not found")
        except Exception as e:
            logger.error(f"Failed to get recording {asset_id}: {e}")
            return ServiceResult.fail(f"Failed to get recording: {e}")

    def download_recording(
        self,
        recording: Union[MLRecording, str],
        output_dir: Union[str, Path],
        filename: Optional[str] = None,
        organize_by_species: bool = True,
    ) -> ServiceResult[Path]:
        """
        Download a single recording.

        Args:
            recording: MLRecording object or asset ID.
            output_dir: Directory to save the file.
            filename: Custom filename (auto-generated if None).
            organize_by_species: Create subdirectory for each species.

        Returns:
            ServiceResult with Path to downloaded file on success.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get recording details if only ID provided
        if isinstance(recording, str):
            result = self.get_recording(recording)
            if not result.success:
                return ServiceResult.fail(f"Recording not found: {recording}")
            recording = result.data

        download_url = recording.get_download_url()
        if not download_url:
            return ServiceResult.fail(f"No download URL for recording {recording.asset_id}")

        # Determine output path
        if organize_by_species and recording.scientific_name:
            species_dir = output_dir / sanitize_filename(recording.scientific_name)
            species_dir.mkdir(exist_ok=True)
            save_dir = species_dir
        else:
            save_dir = output_dir

        # Determine file extension based on media type
        ext_map = {"audio": ".mp3", "video": ".mp4", "photo": ".jpg"}
        ext = ext_map.get(recording.media_type, ".mp3")

        if filename:
            filepath = save_dir / filename
        else:
            safe_name = sanitize_filename(recording.scientific_name or "unknown").replace(" ", "_")
            filepath = save_dir / f"ML{recording.catalog_id}_{safe_name}{ext}"

        try:
            self._client.download(download_url, filepath)
            logger.info(f"Downloaded: {filepath.name}")
            return ServiceResult.ok(data=filepath, message=f"Downloaded {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to download {recording.asset_id}: {e}")
            return ServiceResult.fail(f"Download failed: {e}")

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
        organize_by_species: bool = True,
        create_metadata: bool = True,
        delay: float = 1.0,
    ) -> ServiceResult[DownloadResult]:
        """
        Search and download recordings from Macaulay Library.

        Args:
            species_code: eBird species code (e.g., "amerob").
            scientific_name: Scientific name.
            common_name: Common name.
            region: Region code (e.g., "US-NY").
            country: Country code (e.g., "US").
            taxon_code: eBird taxon code for broader searches.
            hotspot_code: eBird hotspot code.
            min_rating: Minimum quality rating (default: 3).
            max_recordings: Maximum recordings to download.
            output_dir: Output directory.
            organize_by_species: Create subdirectory for each species.
            create_metadata: Create metadata CSV file.
            delay: Delay between downloads in seconds.

        Returns:
            ServiceResult with DownloadResult statistics.
        """
        try:
            # First search for recordings
            search_result = self.search(
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

            if not search_result.success:
                return ServiceResult.fail(f"Search failed: {search_result.error}")

            recordings = search_result.data.recordings

            if not recordings:
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

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            stats = {
                "total": len(recordings),
                "downloaded": 0,
                "failed": 0,
                "skipped": 0,
            }
            errors: List[str] = []
            metadata_rows: List[Dict[str, Any]] = []

            for i, recording in enumerate(recordings, 1):
                result = self.download_recording(
                    recording,
                    output_path,
                    organize_by_species=organize_by_species,
                )

                if result.success:
                    stats["downloaded"] += 1
                    filepath = result.data
                    relative_path = filepath.relative_to(output_path)

                    metadata_rows.append({
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
                    })
                else:
                    stats["failed"] += 1
                    errors.append(f"ML{recording.catalog_id}: {result.error}")

                if i < len(recordings):
                    time.sleep(delay)

            # Write metadata
            if create_metadata and metadata_rows:
                metadata_path = output_path / "metadata.csv"
                with TextFile(metadata_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f.handle, fieldnames=metadata_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(metadata_rows)

            download_result = DownloadResult(
                total=stats["total"],
                downloaded=stats["downloaded"],
                failed=stats["failed"],
                skipped=stats["skipped"],
                output_dir=output_dir,
                errors=errors,
            )

            return ServiceResult.ok(
                data=download_result,
                message=f"Downloaded {stats['downloaded']}/{stats['total']} recordings",
            )

        except Exception as e:
            logger.error(f"Download operation failed: {e}")
            return ServiceResult.fail(f"Download failed: {e}")

    def get_species_count(
        self,
        species_code: str,
        media_type: str = "audio",
    ) -> ServiceResult[int]:
        """
        Get the total number of recordings for a species.

        Args:
            species_code: eBird species code.
            media_type: Media type filter.

        Returns:
            ServiceResult with count on success.
        """
        try:
            response = self._client.get(
                ML_SEARCH_URL,
                params={"speciesCode": species_code, "mediaType": media_type, "count": 0},
            )
            count = response.get("results", {}).get("count", 0)
            return ServiceResult.ok(data=count)
        except Exception as e:
            logger.error(f"Failed to get species count: {e}")
            return ServiceResult.fail(f"Failed to get count: {e}")

    def search_audio(
        self,
        species_code: Optional[str] = None,
        scientific_name: Optional[str] = None,
        region: Optional[str] = None,
        min_rating: int = 0,
        max_results: int = 100,
    ) -> ServiceResult[SearchResult]:
        """
        Convenience method to search for audio recordings only.

        Args:
            species_code: eBird species code.
            scientific_name: Scientific name.
            region: Region code.
            min_rating: Minimum rating.
            max_results: Maximum results.

        Returns:
            ServiceResult with SearchResult on success.
        """
        return self.search(
            species_code=species_code,
            scientific_name=scientific_name,
            region=region,
            media_type="audio",
            min_rating=min_rating,
            max_results=max_results,
        )

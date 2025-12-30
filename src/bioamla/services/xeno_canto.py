# services/xeno_canto.py
"""
Service for Xeno-canto bird recording operations.

Xeno-canto (xeno-canto.org) is the world's largest collection of bird sounds.
This service provides high-level methods for searching and downloading recordings.
"""

import csv
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Union

from bioamla.core.files import TextFile, sanitize_filename
from bioamla.core.constants import APIClient, RateLimiter
from bioamla.models.xeno_canto import DownloadResult, SearchResult, XCRecording

from .base import BaseService, ServiceResult

logger = logging.getLogger(__name__)

# Xeno-canto API base URL (v3 requires API key)
XC_API_URL = "https://xeno-canto.org/api/3/recordings"


class XenoCantoService(BaseService):
    """
    Service for Xeno-canto operations.

    Provides high-level methods for:
    - Searching for bird recordings
    - Downloading audio files with metadata
    - Managing API cache and rate limiting

    Example:
        >>> service = XenoCantoService()
        >>> service.set_api_key("your-api-key")
        >>> result = service.search(species="Turdus migratorius", quality="A")
        >>> if result.success:
        ...     print(f"Found {result.data.total_results} recordings")
    """

    def __init__(self) -> None:
        """Initialize Xeno-canto service with rate limiting."""
        super().__init__()
        self._api_key: Optional[str] = None
        self._rate_limiter = RateLimiter(requests_per_second=1.0, burst_size=2)
        self._client = APIClient(
            rate_limiter=self._rate_limiter,
            user_agent="bioamla/1.0 (bioacoustics research tool)",
        )

    def set_api_key(self, key: str) -> None:
        """
        Set the Xeno-canto API key for authentication.

        As of API v3, an API key is required for all requests.
        Get your API key from your Xeno-canto account page:
        https://xeno-canto.org/account

        Args:
            key: Your Xeno-canto API key.
        """
        self._api_key = key

    def get_api_key(self) -> Optional[str]:
        """
        Get the current API key.

        Returns the API key from (in order of priority):
        1. set_api_key() (runtime setting)
        2. XC_API_KEY environment variable
        3. bioamla config file ([api] xc_api_key)

        Returns:
            The API key or None if not set.
        """
        # Check runtime setting first
        if self._api_key:
            return self._api_key

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

    def _build_query_string(
        self,
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
    ) -> str:
        """Build the query string for Xeno-canto API v3."""
        if query:
            return query

        parts = []

        if species:
            # Parse species name into genus and species parts for tagged query
            species_parts = species.strip().split()
            if len(species_parts) >= 2:
                # Scientific name format: "Genus species" -> gen:Genus sp:species
                parts.append(f"gen:{species_parts[0]}")
                parts.append(f"sp:{species_parts[1]}")
            else:
                # Single name - could be common name or genus, use en: tag
                parts.append(f"en:{species}")
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

        return " ".join(parts) if parts else ""

    def search(
        self,
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
    ) -> ServiceResult[SearchResult]:
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

        Returns:
            ServiceResult with SearchResult data on success.
        """
        try:
            # Build query string
            query_str = self._build_query_string(
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
                return ServiceResult.fail("At least one search parameter is required")

            # Check for API key (required for v3)
            api_key = self.get_api_key()
            if not api_key:
                return ServiceResult.fail(
                    "Xeno-canto API key required. Set via:\n"
                    "  - Environment variable: XC_API_KEY\n"
                    "  - Python: service.set_api_key('your-key')\n"
                    "  - CLI: bioamla config set xc_api_key <your-key>\n"
                    "Get your API key at: https://xeno-canto.org/account"
                )

            all_recordings: List[XCRecording] = []
            current_page = page
            total_pages = 1

            while current_page <= total_pages:
                params = {"query": query_str, "page": current_page, "key": api_key}

                response = self._client.get(XC_API_URL, params=params)

                # Parse response
                recordings_data = response.get("recordings", [])
                num_pages = int(response.get("numPages", 1))
                total_pages = num_pages

                for rec_data in recordings_data:
                    recording = XCRecording.from_api_response(rec_data)
                    all_recordings.append(recording)

                    if max_results and len(all_recordings) >= max_results:
                        all_recordings = all_recordings[:max_results]
                        break

                if max_results and len(all_recordings) >= max_results:
                    break

                current_page += 1

                # Don't hammer the API between pages
                if current_page <= total_pages:
                    time.sleep(0.5)

            result = SearchResult(
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

            return ServiceResult.ok(
                data=result,
                message=f"Found {len(all_recordings)} recordings",
            )

        except Exception as e:
            logger.error(f"Xeno-canto search failed: {e}")
            return ServiceResult.fail(f"Search failed: {e}")

    def get_recording(self, recording_id: str) -> ServiceResult[XCRecording]:
        """
        Get details for a specific recording.

        Args:
            recording_id: Xeno-canto recording ID.

        Returns:
            ServiceResult with XCRecording on success.
        """
        api_key = self.get_api_key()
        if not api_key:
            return ServiceResult.fail("Xeno-canto API key required")

        try:
            response = self._client.get(
                XC_API_URL, params={"query": f"nr:{recording_id}", "key": api_key}
            )
            recordings = response.get("recordings", [])
            if recordings:
                recording = XCRecording.from_api_response(recordings[0])
                return ServiceResult.ok(data=recording)
            return ServiceResult.fail(f"Recording {recording_id} not found")
        except Exception as e:
            logger.error(f"Failed to get recording {recording_id}: {e}")
            return ServiceResult.fail(f"Failed to get recording: {e}")

    def download_recording(
        self,
        recording: Union[XCRecording, str],
        output_dir: Union[str, Path],
        filename: Optional[str] = None,
        organize_by_species: bool = True,
    ) -> ServiceResult[Path]:
        """
        Download a single recording.

        Args:
            recording: XCRecording object or recording ID.
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

        if not recording.download_url:
            return ServiceResult.fail(f"No download URL for recording {recording.id}")

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
            self._client.download(recording.download_url, filepath)
            logger.info(f"Downloaded: {filepath.name}")
            return ServiceResult.ok(data=filepath, message=f"Downloaded {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to download {recording.id}: {e}")
            return ServiceResult.fail(f"Download failed: {e}")

    def download(
        self,
        species: Optional[str] = None,
        genus: Optional[str] = None,
        country: Optional[str] = None,
        quality: Optional[str] = "A",
        max_recordings: int = 10,
        output_dir: str = "./xc_recordings",
        organize_by_species: bool = True,
        create_metadata: bool = True,
        delay: float = 1.0,
    ) -> ServiceResult[DownloadResult]:
        """
        Search and download recordings from Xeno-canto.

        Args:
            species: Species name (scientific or common).
            genus: Genus name.
            country: Country name.
            quality: Recording quality filter (default: A).
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
                species=species,
                genus=genus,
                country=country,
                quality=quality,
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
            metadata_rows = []

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
                    errors.append(f"XC{recording.id}: {result.error}")

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

    def get_species_recordings_count(self, species: str) -> ServiceResult[int]:
        """
        Get the total number of recordings for a species.

        Args:
            species: Species name (scientific or common).

        Returns:
            ServiceResult with count on success.
        """
        api_key = self.get_api_key()
        if not api_key:
            return ServiceResult.fail("Xeno-canto API key required")

        try:
            # Parse species into tagged query format for API v3
            species_parts = species.strip().split()
            if len(species_parts) >= 2:
                query = f"gen:{species_parts[0]} sp:{species_parts[1]}"
            else:
                query = f"en:{species}"

            response = self._client.get(
                XC_API_URL, params={"query": query, "key": api_key}
            )
            count = int(response.get("numRecordings", 0))
            return ServiceResult.ok(data=count)
        except Exception as e:
            logger.error(f"Failed to get recordings count: {e}")
            return ServiceResult.fail(f"Failed to get count: {e}")

    def search_by_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 50.0,
        quality: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> ServiceResult[SearchResult]:
        """
        Search for recordings near a geographic location.

        Args:
            latitude: Center latitude.
            longitude: Center longitude.
            radius_km: Search radius in kilometers (approximate).
            quality: Recording quality filter.
            max_results: Maximum results to return.

        Returns:
            ServiceResult with SearchResult on success.
        """
        # Convert radius to approximate lat/lon degrees
        # 1 degree latitude ≈ 111 km
        lat_delta = radius_km / 111.0
        # 1 degree longitude varies by latitude
        import math

        lon_delta = radius_km / (111.0 * math.cos(math.radians(latitude)))

        box = (
            latitude - lat_delta,
            latitude + lat_delta,
            longitude - lon_delta,
            longitude + lon_delta,
        )

        return self.search(box=box, quality=quality, max_results=max_results)

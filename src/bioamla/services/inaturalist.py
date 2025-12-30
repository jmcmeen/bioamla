# services/inaturalist.py
"""
Service for iNaturalist data operations.

This service provides utilities for importing audio observation data from iNaturalist.
It uses the pyinaturalist library to query observations with sounds and download
audio files for use in bioacoustic machine learning workflows.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from pyinaturalist import get_observation_species_counts, get_observations

from bioamla.core.metadata import (
    get_existing_observation_ids,
    read_metadata_csv,
    write_metadata_csv,
)
from bioamla.core.files import (
    BinaryFile,
    TextFile,
    get_extension_from_content_type,
    get_extension_from_url,
    sanitize_filename,
)
from bioamla.models.inaturalist import (
    DownloadResult,
    ObservationInfo,
    ProjectStats,
    SearchResult,
    TaxonInfo,
)

from .base import BaseService, ServiceResult

logger = logging.getLogger(__name__)


class INaturalistService(BaseService):
    """
    Service for iNaturalist operations.

    Provides high-level methods for:
    - Searching for observations with sounds
    - Downloading audio files with metadata
    - Discovering taxa in projects/places
    - Getting project statistics

    Example:
        >>> service = INaturalistService()
        >>> result = service.search(taxon_name="Strix varia", per_page=5)
        >>> if result.success:
        ...     print(f"Found {result.data.total_results} observations")
    """

    def __init__(self) -> None:
        """Initialize iNaturalist service."""
        super().__init__()
        self._download_callback: Optional[Callable[[int, int, str], None]] = None

    def set_download_callback(
        self,
        callback: Callable[[int, int, str], None],
    ) -> None:
        """
        Set callback for download progress updates.

        Callback signature: (current, total, current_file) -> None
        """
        self._download_callback = callback

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _download_file(self, url: str, filepath: Path, verbose: bool = True) -> bool:
        """Download a file from URL to local path."""
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if content_type:
                ext = get_extension_from_content_type(content_type)
                if ext and not str(filepath).lower().endswith(ext):
                    filepath = filepath.with_suffix(ext)

            with BinaryFile(filepath, mode="wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if verbose:
                logger.debug(f"Downloaded: {filepath.name}")
            return True

        except requests.RequestException as e:
            if verbose:
                logger.warning(f"Failed to download {url}: {e}")
            return False

    def _discover_taxa_from_query(
        self,
        taxon_name: Optional[str] = None,
        place_id: Optional[int] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        quality_grade: Optional[str] = None,
        sound_license: Optional[List[str]] = None,
        d1: Optional[str] = None,
        d2: Optional[str] = None,
    ) -> List[int]:
        """
        Discover unique taxa from a query to enable proper per-taxon limits.

        Uses the species_counts API to efficiently get all taxa matching the query,
        then returns their IDs for individual per-taxon downloads.
        """
        try:
            all_taxa = []
            page = 1
            per_page = 500

            while True:
                response = get_observation_species_counts(
                    sounds=True,
                    taxon_name=taxon_name,
                    place_id=place_id,
                    user_id=user_id,
                    project_id=project_id,
                    quality_grade=quality_grade,
                    sound_license=sound_license,
                    d1=d1,
                    d2=d2,
                    page=page,
                    per_page=per_page,
                )

                results = response.get("results", [])
                if not results:
                    break

                for item in results:
                    taxon = item.get("taxon", {})
                    taxon_id = taxon.get("id")
                    if taxon_id:
                        all_taxa.append(taxon_id)

                if len(results) < per_page:
                    break

                page += 1

            return all_taxa

        except Exception as e:
            logger.warning(f"Failed to discover taxa: {e}")
            return []

    def _load_taxon_ids_from_csv(self, csv_path: Union[str, Path]) -> List[int]:
        """
        Load taxon IDs from a CSV file.

        The CSV file should have a column named 'taxon_id' containing integer taxon IDs.
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"Taxon CSV file not found: {csv_path}")

        taxon_ids: List[int] = []

        with TextFile(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f.handle)

            if reader.fieldnames is None or "taxon_id" not in reader.fieldnames:
                raise ValueError(
                    f"CSV file must have a 'taxon_id' column. Found columns: {reader.fieldnames}"
                )

            for row_num, row in enumerate(reader, start=2):
                taxon_id_str = row.get("taxon_id", "").strip()

                if not taxon_id_str:
                    logger.warning(f"Empty taxon_id at row {row_num}, skipping")
                    continue

                try:
                    taxon_id = int(taxon_id_str)
                    taxon_ids.append(taxon_id)
                except ValueError:
                    logger.warning(f"Invalid taxon_id '{taxon_id_str}' at row {row_num}, skipping")

        if not taxon_ids:
            raise ValueError(f"No valid taxon IDs found in {csv_path}")

        logger.info(f"Loaded {len(taxon_ids)} taxon IDs from {csv_path}")
        return taxon_ids

    # =========================================================================
    # Search
    # =========================================================================

    def search(
        self,
        taxon_id: Optional[int] = None,
        taxon_name: Optional[str] = None,
        place_id: Optional[int] = None,
        quality_grade: Optional[str] = "research",
        per_page: int = 30,
    ) -> ServiceResult[SearchResult]:
        """
        Search for iNaturalist observations with sounds.

        Args:
            taxon_id: Filter by taxon ID
            taxon_name: Filter by taxon name (e.g., "Strix varia")
            place_id: Filter by place ID
            quality_grade: Filter by quality grade ("research", "needs_id", "casual")
            per_page: Number of results to return

        Returns:
            Result with search results
        """
        try:
            response = get_observations(
                sounds=True,
                taxon_id=taxon_id,
                taxon_name=taxon_name,
                place_id=place_id,
                quality_grade=quality_grade,
                per_page=per_page,
            )

            observations = response.get("results", [])

            result = SearchResult(
                total_results=len(observations),
                observations=observations,
                query_params={
                    "taxon_id": taxon_id,
                    "taxon_name": taxon_name,
                    "place_id": place_id,
                    "quality_grade": quality_grade,
                },
            )

            return ServiceResult.ok(
                data=result,
                message=f"Found {len(observations)} observations with sounds",
            )
        except Exception as e:
            return ServiceResult.fail(f"Search failed: {e}")

    # =========================================================================
    # Download
    # =========================================================================

    def download(
        self,
        output_dir: str,
        taxon_ids: Optional[List[int]] = None,
        taxon_csv: Optional[str] = None,
        taxon_name: Optional[str] = None,
        place_id: Optional[int] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        quality_grade: Optional[str] = "research",
        sound_license: Optional[List[str]] = None,
        d1: Optional[str] = None,
        d2: Optional[str] = None,
        obs_per_taxon: int = 100,
        per_page: int = 30,
        delay_between_downloads: float = 1.0,
        organize_by_taxon: bool = True,
        include_metadata: bool = False,
    ) -> ServiceResult[DownloadResult]:
        """
        Download audio files from iNaturalist observations.

        Args:
            output_dir: Directory where audio files will be saved
            taxon_ids: Filter by taxon ID(s)
            taxon_csv: Path to a CSV file containing taxon IDs
            taxon_name: Filter by taxon name (e.g., "Aves" for birds)
            place_id: Filter by place ID (e.g., 1 for United States)
            user_id: Filter by observer username
            project_id: Filter by iNaturalist project ID or slug
            quality_grade: Filter by quality grade ("research", "needs_id", or "casual")
            sound_license: Filter by sound license(s) (e.g., ["cc-by", "cc-by-nc", "cc0"])
            d1: Start date for observation date range (YYYY-MM-DD format)
            d2: End date for observation date range (YYYY-MM-DD format)
            obs_per_taxon: Number of observations to download per taxon ID
            per_page: Number of results per API request (max 200)
            delay_between_downloads: Seconds to wait between file downloads
            organize_by_taxon: If True, organize files into subdirectories by species
            include_metadata: If True, include additional iNaturalist metadata fields

        Returns:
            ServiceResult with DownloadResult statistics
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Load taxon IDs from CSV if provided
            if taxon_csv:
                csv_taxon_ids = self._load_taxon_ids_from_csv(taxon_csv)
                if taxon_ids:
                    taxon_ids = list(taxon_ids) + csv_taxon_ids
                else:
                    taxon_ids = csv_taxon_ids

            stats = {
                "total_observations": 0,
                "total_sounds": 0,
                "observations_with_multiple_sounds": 0,
                "skipped_existing": 0,
                "failed_downloads": 0,
            }
            errors: List[str] = []
            metadata_rows: List[Dict[str, Any]] = []

            # Load existing metadata to skip already-downloaded files
            existing_files = get_existing_observation_ids(output_path / "metadata.csv")

            # Normalize sound_license to uppercase for pyinaturalist API compatibility
            normalized_license = None
            if sound_license:
                normalized_license = [lic.upper() for lic in sound_license]

            # Build list of taxon IDs to iterate over
            if taxon_ids:
                taxon_list = taxon_ids
            else:
                # Discover taxa from the query to properly apply obs_per_taxon to each
                discovered_taxa = self._discover_taxa_from_query(
                    taxon_name=taxon_name,
                    place_id=place_id,
                    user_id=user_id,
                    project_id=project_id,
                    quality_grade=quality_grade,
                    sound_license=normalized_license,
                    d1=d1,
                    d2=d2,
                )
                taxon_list = discovered_taxa if discovered_taxa else [None]

            for current_taxon_id in taxon_list:
                page = 1
                observations_processed = 0

                while observations_processed < obs_per_taxon:
                    remaining = obs_per_taxon - observations_processed
                    current_per_page = min(per_page, remaining)

                    response = get_observations(
                        sounds=True,
                        taxon_id=current_taxon_id,
                        taxon_name=taxon_name,
                        place_id=place_id,
                        user_id=user_id,
                        project_id=project_id,
                        quality_grade=quality_grade,
                        sound_license=normalized_license,
                        d1=d1,
                        d2=d2,
                        page=page,
                        per_page=current_per_page,
                    )

                    results = response.get("results", [])
                    if not results:
                        break

                    for obs in results:
                        if observations_processed >= obs_per_taxon:
                            break

                        obs_id = obs.get("id")
                        sounds = obs.get("sounds", [])

                        if not sounds:
                            continue

                        taxon = obs.get("taxon", {})
                        species_name = taxon.get("name", "unknown")
                        common_name = taxon.get("preferred_common_name", "")
                        taxon_id_val = taxon.get("id", "")
                        taxon_rank = taxon.get("rank", "")

                        # For subspecies/varieties, use the parent species name
                        if taxon_rank in ("subspecies", "variety", "form"):
                            ancestors = taxon.get("ancestors", [])
                            for ancestor in reversed(ancestors):
                                if ancestor.get("rank") == "species":
                                    species_name = ancestor.get("name", species_name)
                                    break
                            else:
                                name_parts = species_name.split()
                                if len(name_parts) >= 2:
                                    species_name = " ".join(name_parts[:2])

                        observed_on = obs.get("observed_on", "")
                        location = obs.get("location", "")
                        place_guess = obs.get("place_guess", "")
                        user = obs.get("user", {}).get("login", "")
                        quality = obs.get("quality_grade", "")

                        safe_species = sanitize_filename(species_name)

                        if organize_by_taxon:
                            species_dir = output_path / safe_species
                            species_dir.mkdir(exist_ok=True)
                        else:
                            species_dir = output_path

                        # Track if this observation has multiple sounds
                        sound_count_for_obs = 0

                        for sound in sounds:
                            sound_id = sound.get("id")
                            file_url = sound.get("file_url")
                            license_code = sound.get("license_code", "")

                            if not file_url:
                                continue

                            ext = get_extension_from_url(file_url)

                            # Skip files that already exist
                            if (obs_id, sound_id) in existing_files:
                                stats["skipped_existing"] += 1
                                continue

                            filename = f"inat_{obs_id}_sound_{sound_id}{ext}"
                            filepath = species_dir / filename

                            success = self._download_file(file_url, filepath, verbose=False)

                            if success:
                                stats["total_sounds"] += 1
                                sound_count_for_obs += 1
                                relative_path = filepath.relative_to(output_path)

                                row = {
                                    "file_name": str(relative_path),
                                    "split": "train",
                                    "target": "",
                                    "label": safe_species,
                                    "inat_obs_id": obs_id,
                                    "attr_id": user,
                                    "attr_lic": license_code,
                                    "attr_url": file_url,
                                    "attr_note": "",
                                }

                                if include_metadata:
                                    row.update({
                                        "observation_id": obs_id,
                                        "sound_id": sound_id,
                                        "common_name": common_name,
                                        "taxon_id": taxon_id_val,
                                        "observed_on": observed_on,
                                        "location": location,
                                        "place_guess": place_guess,
                                        "observer": user,
                                        "quality_grade": quality,
                                        "observation_url": f"https://www.inaturalist.org/observations/{obs_id}",
                                    })

                                metadata_rows.append(row)
                            else:
                                stats["failed_downloads"] += 1
                                errors.append(f"Failed to download sound {sound_id} from observation {obs_id}")

                            time.sleep(delay_between_downloads)

                        # Only count observation if at least one sound was downloaded
                        if sound_count_for_obs > 0:
                            observations_processed += 1
                            stats["total_observations"] += 1

                            # Track observations with multiple successfully downloaded sounds
                            if sound_count_for_obs > 1:
                                stats["observations_with_multiple_sounds"] += 1

                    page += 1

                    if len(results) < current_per_page:
                        break

            # Write metadata
            if metadata_rows:
                metadata_path = output_path / "metadata.csv"
                existing_rows, _ = read_metadata_csv(metadata_path)

                seen_files: set = set()
                all_rows = []
                for row in existing_rows:
                    file_name = row.get("file_name", "")
                    if file_name and file_name not in seen_files:
                        seen_files.add(file_name)
                        all_rows.append(row)
                for row in metadata_rows:
                    file_name = row.get("file_name", "")
                    if file_name and file_name not in seen_files:
                        seen_files.add(file_name)
                        all_rows.append(row)

                write_metadata_csv(metadata_path, all_rows, merge_existing=False)

            result = DownloadResult(
                total_observations=stats["total_observations"],
                total_sounds=stats["total_sounds"],
                observations_with_multiple_sounds=stats["observations_with_multiple_sounds"],
                skipped_existing=stats["skipped_existing"],
                failed_downloads=stats["failed_downloads"],
                output_dir=str(output_path.absolute()),
                metadata_file=str(output_path / "metadata.csv"),
                errors=errors,
            )

            message = f"Downloaded {result.total_sounds} audio files"
            if result.skipped_existing:
                message += f" (skipped {result.skipped_existing} existing)"

            return ServiceResult.ok(data=result, message=message)

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return ServiceResult.fail(f"Download failed: {e}")

    def download_from_observations(
        self,
        observation_ids: List[int],
        output_dir: str,
        organize_by_taxon: bool = True,
    ) -> ServiceResult[DownloadResult]:
        """
        Download audio from specific observation IDs.

        Args:
            observation_ids: List of observation IDs
            output_dir: Directory to save audio files
            organize_by_taxon: Organize by species

        Returns:
            Result with download statistics
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            total_sounds = 0
            failed = 0
            errors: List[str] = []

            for obs_id in observation_ids:
                try:
                    response = get_observations(id=obs_id)
                    results = response.get("results", [])

                    if not results:
                        errors.append(f"Observation {obs_id} not found")
                        continue

                    obs = results[0]
                    sounds = obs.get("sounds", [])
                    taxon = obs.get("taxon", {})
                    species_name = taxon.get("name", "unknown")
                    safe_species = sanitize_filename(species_name)

                    if organize_by_taxon:
                        species_dir = output_path / safe_species
                        species_dir.mkdir(exist_ok=True)
                    else:
                        species_dir = output_path

                    for sound in sounds:
                        sound_id = sound.get("id")
                        file_url = sound.get("file_url")

                        if not file_url:
                            continue

                        ext = Path(file_url).suffix or ".mp3"
                        filename = f"inat_{obs_id}_sound_{sound_id}{ext}"
                        filepath = species_dir / filename

                        if self._download_file(file_url, filepath, verbose=False):
                            total_sounds += 1
                        else:
                            failed += 1

                except Exception as e:
                    errors.append(f"Observation {obs_id}: {e}")
                    failed += 1

            result = DownloadResult(
                total_observations=len(observation_ids),
                total_sounds=total_sounds,
                observations_with_multiple_sounds=0,
                skipped_existing=0,
                failed_downloads=failed,
                output_dir=str(output_path),
                metadata_file="",
                errors=errors,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Downloaded {total_sounds} sounds from {len(observation_ids)} observations",
            )
        except Exception as e:
            return ServiceResult.fail(f"Download failed: {e}")

    # =========================================================================
    # Taxa Discovery
    # =========================================================================

    def get_taxa(
        self,
        place_id: Optional[int] = None,
        project_id: Optional[str] = None,
        quality_grade: Optional[str] = "research",
        parent_taxon_id: Optional[int] = None,
    ) -> ServiceResult[List[TaxonInfo]]:
        """
        Get taxa with observations in a place or project.

        Args:
            place_id: Filter by place ID
            project_id: Filter by project ID or slug
            quality_grade: Filter by quality grade
            parent_taxon_id: Filter by parent taxon (e.g., 20979 for Amphibia)

        Returns:
            Result with list of taxa
        """
        if not place_id and not project_id:
            return ServiceResult.fail("At least one of place_id or project_id must be provided")

        try:
            taxa_list: List[Dict[str, Any]] = []
            page = 1
            per_page = 500

            while True:
                response = get_observation_species_counts(
                    place_id=place_id,
                    project_id=project_id,
                    quality_grade=quality_grade,
                    taxon_id=parent_taxon_id,
                    page=page,
                    per_page=per_page,
                )

                results = response.get("results", [])
                if not results:
                    break

                for item in results:
                    taxon = item.get("taxon", {})
                    taxa_list.append({
                        "taxon_id": taxon.get("id"),
                        "name": taxon.get("name", "unknown"),
                        "common_name": taxon.get("preferred_common_name", ""),
                        "observation_count": item.get("count", 0),
                    })

                if len(results) < per_page:
                    break

                page += 1

            # Sort by observation count descending
            taxa_list.sort(key=lambda x: x["observation_count"], reverse=True)

            results_list = [
                TaxonInfo(
                    taxon_id=t["taxon_id"],
                    name=t["name"],
                    common_name=t["common_name"],
                    observation_count=t["observation_count"],
                )
                for t in taxa_list
            ]

            return ServiceResult.ok(
                data=results_list,
                message=f"Found {len(results_list)} taxa",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to get taxa: {e}")

    def export_taxa_csv(
        self,
        taxa: List[TaxonInfo],
        output_path: str,
    ) -> ServiceResult[str]:
        """
        Export taxa list to CSV file.

        Args:
            taxa: List of TaxonInfo objects
            output_path: Path to output CSV file

        Returns:
            Result with output path
        """
        try:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            with open(output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["taxon_id", "name", "common_name", "observation_count"],
                )
                writer.writeheader()
                for taxon in taxa:
                    writer.writerow({
                        "taxon_id": taxon.taxon_id,
                        "name": taxon.name,
                        "common_name": taxon.common_name,
                        "observation_count": taxon.observation_count,
                    })

            return ServiceResult.ok(
                data=str(output),
                message=f"Exported {len(taxa)} taxa to {output}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to export taxa: {e}")

    # =========================================================================
    # Project & Observation Info
    # =========================================================================

    def get_project_stats(
        self,
        project_id: str,
    ) -> ServiceResult[ProjectStats]:
        """
        Get statistics for an iNaturalist project.

        Args:
            project_id: Project ID or slug (e.g., "appalachia-bioacoustics")

        Returns:
            Result with project statistics
        """
        try:
            # Get project metadata
            project_url = f"https://api.inaturalist.org/v1/projects/{project_id}"
            response = requests.get(project_url, timeout=30)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            if not results:
                return ServiceResult.fail(f"Project '{project_id}' not found")

            project = results[0]

            # Get observation count
            obs_url = f"https://api.inaturalist.org/v1/observations?project_id={project_id}&per_page=0"
            obs_response = requests.get(obs_url, timeout=30)
            obs_response.raise_for_status()
            observation_count = obs_response.json().get("total_results", 0)

            # Get species count
            species_url = f"https://api.inaturalist.org/v1/observations/species_counts?project_id={project_id}&per_page=0"
            species_response = requests.get(species_url, timeout=30)
            species_response.raise_for_status()
            species_count = species_response.json().get("total_results", 0)

            # Get observers count
            observers_url = f"https://api.inaturalist.org/v1/observations/observers?project_id={project_id}&per_page=0"
            observers_response = requests.get(observers_url, timeout=30)
            observers_response.raise_for_status()
            observers_count = observers_response.json().get("total_results", 0)

            result = ProjectStats(
                id=project.get("id"),
                title=project.get("title", ""),
                description=project.get("description", ""),
                slug=project.get("slug", ""),
                observation_count=observation_count,
                species_count=species_count,
                observers_count=observers_count,
                created_at=project.get("created_at", ""),
                project_type=project.get("project_type", ""),
                place=project.get("place", {}).get("display_name", "") if project.get("place") else "",
                url=f"https://www.inaturalist.org/projects/{project.get('slug', project_id)}",
            )

            return ServiceResult.ok(
                data=result,
                message=f"Project: {result.title}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to get project stats: {e}")

    def get_observation(
        self,
        observation_id: int,
    ) -> ServiceResult[ObservationInfo]:
        """
        Get information about a specific observation.

        Args:
            observation_id: The iNaturalist observation ID

        Returns:
            Result with observation information
        """
        try:
            response = get_observations(id=observation_id)
            results = response.get("results", [])

            if not results:
                return ServiceResult.fail(f"Observation {observation_id} not found")

            obs = results[0]
            taxon = obs.get("taxon", {})

            result = ObservationInfo(
                id=obs.get("id"),
                taxon_name=taxon.get("name", "unknown"),
                common_name=taxon.get("preferred_common_name", ""),
                observed_on=obs.get("observed_on", ""),
                location=obs.get("location", ""),
                place_guess=obs.get("place_guess", ""),
                observer=obs.get("user", {}).get("login", ""),
                quality_grade=obs.get("quality_grade", ""),
                sounds=obs.get("sounds", []),
                url=f"https://www.inaturalist.org/observations/{observation_id}",
            )

            return ServiceResult.ok(
                data=result,
                message=f"Observation: {result.taxon_name}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to get observation: {e}")

    def get_observation_sounds(
        self,
        observation_id: int,
    ) -> ServiceResult[List[Dict[str, Any]]]:
        """
        Get all sounds from a specific observation.

        Args:
            observation_id: The iNaturalist observation ID

        Returns:
            Result with list of sound dictionaries
        """
        try:
            response = get_observations(id=observation_id)
            results = response.get("results", [])

            if results:
                sounds = results[0].get("sounds", [])
                return ServiceResult.ok(
                    data=sounds,
                    message=f"Found {len(sounds)} sounds",
                )
            return ServiceResult.ok(data=[], message="No sounds found")
        except Exception as e:
            return ServiceResult.fail(f"Failed to get sounds: {e}")

    # =========================================================================
    # Utilities
    # =========================================================================

    def load_taxon_ids(
        self,
        csv_path: str,
    ) -> ServiceResult[List[int]]:
        """
        Load taxon IDs from a CSV file.

        Args:
            csv_path: Path to CSV file with 'taxon_id' column

        Returns:
            Result with list of taxon IDs
        """
        try:
            taxon_ids = self._load_taxon_ids_from_csv(csv_path)
            return ServiceResult.ok(
                data=taxon_ids,
                message=f"Loaded {len(taxon_ids)} taxon IDs from {csv_path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to load taxon IDs: {e}")

    def get_common_taxa(self) -> ServiceResult[Dict[str, int]]:
        """
        Get a dictionary of common taxon IDs.

        Returns:
            Result with dictionary mapping names to taxon IDs
        """
        common_taxa = {
            # Major groups
            "Aves": 3,  # Birds
            "Amphibia": 20978,  # Amphibians
            "Mammalia": 40151,  # Mammals
            "Insecta": 47158,  # Insects
            "Reptilia": 26036,  # Reptiles
            # Bird orders
            "Passeriformes": 7251,  # Songbirds
            "Strigiformes": 19350,  # Owls
            "Piciformes": 19893,  # Woodpeckers
            "Caprimulgiformes": 67569,  # Nightjars
            # Amphibian orders
            "Anura": 20979,  # Frogs and toads
            "Caudata": 20980,  # Salamanders
            # Common families
            "Corvidae": 7823,  # Crows, jays
            "Strigidae": 19374,  # Typical owls
            "Hylidae": 21013,  # Tree frogs
            "Ranidae": 21136,  # True frogs
            "Bufonidae": 20981,  # True toads
            # Common species
            "Strix varia": 19893,  # Barred Owl
            "Bubo virginianus": 20949,  # Great Horned Owl
            "Lithobates catesbeianus": 24230,  # American Bullfrog
            "Pseudacris crucifer": 24234,  # Spring Peeper
            "Anaxyrus americanus": 24225,  # American Toad
        }

        return ServiceResult.ok(
            data=common_taxa,
            message=f"Found {len(common_taxa)} common taxa",
        )

"""iNaturalist observation search and audio download.

Uses the ``pyinaturalist`` library (a core dependency) to query observations
with sounds and download audio for bioacoustic ML workflows.

Failures raise :class:`~bioamla.exceptions.CatalogError`; bad arguments raise
:class:`~bioamla.exceptions.InvalidInputError`.
"""
import csv
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from pyinaturalist import get_observation_species_counts, get_observations

from bioamla.catalogs._metadata import (
    get_existing_observation_ids,
    read_metadata_csv,
    write_metadata_csv,
)
from bioamla.catalogs._models import (
    INaturalistDownloadResult,
    INaturalistSearchResult,
    ObservationInfo,
    ProjectStats,
    TaxonInfo,
)
from bioamla.common.files import (
    get_extension_from_content_type,
    get_extension_from_url,
    sanitize_filename,
)
from bioamla.exceptions import CatalogError, InvalidInputError

logger = logging.getLogger(__name__)

# Progress callback signature: (current, total, current_file) -> None
ProgressCallback = Callable[[int, int, str], None]


# =============================================================================
# Internal helpers
# =============================================================================


def _download_file(url: str, filepath: Path, verbose: bool = True) -> bool:
    """Download a file from URL to local path. Returns True on success."""
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if content_type:
            ext = get_extension_from_content_type(content_type)
            if ext and not str(filepath).lower().endswith(ext):
                filepath = filepath.with_suffix(ext)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as f:
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
    taxon_name: Optional[str] = None,
    place_id: Optional[int] = None,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    quality_grade: Optional[str] = None,
    sound_license: Optional[List[str]] = None,
    d1: Optional[str] = None,
    d2: Optional[str] = None,
) -> List[int]:
    """Discover unique taxa matching a query, for per-taxon download limits.

    Returns an empty list on failure (caller treats this as "no discovery").
    """
    try:
        all_taxa: List[int] = []
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
                taxon_id = item.get("taxon", {}).get("id")
                if taxon_id:
                    all_taxa.append(taxon_id)
            if len(results) < per_page:
                break
            page += 1
        return all_taxa
    except Exception as e:
        logger.warning(f"Failed to discover taxa: {e}")
        return []


def _load_taxon_ids_from_csv(csv_path: Union[str, Path]) -> List[int]:
    """Load taxon IDs from a CSV with a 'taxon_id' column.

    Raises:
        InvalidInputError: if the file is missing, malformed, or has no valid IDs.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise InvalidInputError(f"Taxon CSV file not found: {csv_path}")

    taxon_ids: List[int] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "taxon_id" not in reader.fieldnames:
            raise InvalidInputError(
                f"CSV file must have a 'taxon_id' column. Found columns: {reader.fieldnames}"
            )
        for row_num, row in enumerate(reader, start=2):
            taxon_id_str = row.get("taxon_id", "").strip()
            if not taxon_id_str:
                logger.warning(f"Empty taxon_id at row {row_num}, skipping")
                continue
            try:
                taxon_ids.append(int(taxon_id_str))
            except ValueError:
                logger.warning(f"Invalid taxon_id '{taxon_id_str}' at row {row_num}, skipping")

    if not taxon_ids:
        raise InvalidInputError(f"No valid taxon IDs found in {csv_path}")

    logger.info(f"Loaded {len(taxon_ids)} taxon IDs from {csv_path}")
    return taxon_ids


# =============================================================================
# Search
# =============================================================================


def search(
    taxon_id: Optional[int] = None,
    taxon_name: Optional[str] = None,
    place_id: Optional[int] = None,
    quality_grade: Optional[str] = "research",
    per_page: int = 30,
) -> INaturalistSearchResult:
    """Search for iNaturalist observations with sounds.

    Raises:
        CatalogError: on API failure.
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
        return INaturalistSearchResult(
            total_results=len(observations),
            observations=observations,
            query_params={
                "taxon_id": taxon_id,
                "taxon_name": taxon_name,
                "place_id": place_id,
                "quality_grade": quality_grade,
            },
        )
    except Exception as e:
        raise CatalogError(f"Search failed: {e}") from e


# =============================================================================
# Download
# =============================================================================


def download(
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
    progress_callback: Optional[ProgressCallback] = None,
) -> INaturalistDownloadResult:
    """Download audio files from iNaturalist observations.

    Args:
        output_dir: Directory where audio files will be saved.
        taxon_ids: Filter by taxon ID(s).
        taxon_csv: Path to a CSV file containing taxon IDs.
        taxon_name: Filter by taxon name (e.g., "Aves" for birds).
        place_id: Filter by place ID.
        user_id: Filter by observer username.
        project_id: Filter by iNaturalist project ID or slug.
        quality_grade: Filter by quality grade ("research", "needs_id", "casual").
        sound_license: Filter by sound license(s) (e.g., ["cc-by", "cc0"]).
        d1: Start date for observation range (YYYY-MM-DD).
        d2: End date for observation range (YYYY-MM-DD).
        obs_per_taxon: Number of observations to download per taxon ID.
        per_page: Number of results per API request (max 200).
        delay_between_downloads: Seconds to wait between file downloads.
        organize_by_taxon: If True, organize files into per-species subdirectories.
        include_metadata: If True, include additional iNaturalist metadata fields.
        progress_callback: Optional ``(current, total, current_file)`` callback
            invoked after each sound download.

    Raises:
        CatalogError: on download failure.
        InvalidInputError: if ``taxon_csv`` is invalid.
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if taxon_csv:
            csv_taxon_ids = _load_taxon_ids_from_csv(taxon_csv)
            taxon_ids = list(taxon_ids) + csv_taxon_ids if taxon_ids else csv_taxon_ids

        stats = {
            "total_observations": 0,
            "total_sounds": 0,
            "observations_with_multiple_sounds": 0,
            "skipped_existing": 0,
            "failed_downloads": 0,
        }
        errors: List[str] = []
        metadata_rows: List[Dict[str, Any]] = []

        existing_files = get_existing_observation_ids(output_path / "metadata.csv")

        normalized_license = None
        if sound_license:
            normalized_license = [lic.upper() for lic in sound_license]

        if taxon_ids:
            taxon_list: List[Optional[int]] = list(taxon_ids)
        else:
            discovered_taxa = _discover_taxa_from_query(
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

                    sound_count_for_obs = 0
                    for sound in sounds:
                        sound_id = sound.get("id")
                        file_url = sound.get("file_url")
                        license_code = sound.get("license_code", "")
                        if not file_url:
                            continue

                        ext = get_extension_from_url(file_url)
                        if (obs_id, sound_id) in existing_files:
                            stats["skipped_existing"] += 1
                            continue

                        filename = f"inat_{obs_id}_sound_{sound_id}{ext}"
                        filepath = species_dir / filename

                        success = _download_file(file_url, filepath, verbose=False)
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
                            if progress_callback:
                                progress_callback(
                                    stats["total_sounds"], len(sounds), str(filepath)
                                )
                        else:
                            stats["failed_downloads"] += 1
                            errors.append(
                                f"Failed to download sound {sound_id} from observation {obs_id}"
                            )

                        time.sleep(delay_between_downloads)

                    if sound_count_for_obs > 0:
                        observations_processed += 1
                        stats["total_observations"] += 1
                        if sound_count_for_obs > 1:
                            stats["observations_with_multiple_sounds"] += 1

                page += 1
                if len(results) < current_per_page:
                    break

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

        return INaturalistDownloadResult(
            total_observations=stats["total_observations"],
            total_sounds=stats["total_sounds"],
            observations_with_multiple_sounds=stats["observations_with_multiple_sounds"],
            skipped_existing=stats["skipped_existing"],
            failed_downloads=stats["failed_downloads"],
            output_dir=str(output_path.absolute()),
            metadata_file=str(output_path / "metadata.csv"),
            errors=errors,
        )
    except (CatalogError, InvalidInputError):
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise CatalogError(f"Download failed: {e}") from e


def download_from_observations(
    observation_ids: List[int],
    output_dir: str,
    organize_by_taxon: bool = True,
) -> INaturalistDownloadResult:
    """Download audio from specific observation IDs.

    Raises:
        CatalogError: on download failure.
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
                    if _download_file(file_url, filepath, verbose=False):
                        total_sounds += 1
                    else:
                        failed += 1
            except Exception as e:
                errors.append(f"Observation {obs_id}: {e}")
                failed += 1

        return INaturalistDownloadResult(
            total_observations=len(observation_ids),
            total_sounds=total_sounds,
            observations_with_multiple_sounds=0,
            skipped_existing=0,
            failed_downloads=failed,
            output_dir=str(output_path),
            metadata_file="",
            errors=errors,
        )
    except Exception as e:
        raise CatalogError(f"Download failed: {e}") from e


# =============================================================================
# Taxa discovery
# =============================================================================


def get_taxa(
    place_id: Optional[int] = None,
    project_id: Optional[str] = None,
    quality_grade: Optional[str] = "research",
    parent_taxon_id: Optional[int] = None,
) -> List[TaxonInfo]:
    """Get taxa with observations in a place or project, sorted by count desc.

    Raises:
        InvalidInputError: if neither place_id nor project_id is provided.
        CatalogError: on API failure.
    """
    if not place_id and not project_id:
        raise InvalidInputError("At least one of place_id or project_id must be provided")

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

        taxa_list.sort(key=lambda x: x["observation_count"], reverse=True)
        return [
            TaxonInfo(
                taxon_id=t["taxon_id"],
                name=t["name"],
                common_name=t["common_name"],
                observation_count=t["observation_count"],
            )
            for t in taxa_list
        ]
    except Exception as e:
        raise CatalogError(f"Failed to get taxa: {e}") from e


# =============================================================================
# Project & observation info
# =============================================================================


def get_project_stats(project_id: str) -> ProjectStats:
    """Get statistics for an iNaturalist project.

    Raises:
        CatalogError: if the project is not found or an API call fails.
    """
    try:
        project_url = f"https://api.inaturalist.org/v1/projects/{project_id}"
        response = requests.get(project_url, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            raise CatalogError(f"Project '{project_id}' not found")
        project = results[0]

        obs_url = (
            f"https://api.inaturalist.org/v1/observations?project_id={project_id}&per_page=0"
        )
        obs_response = requests.get(obs_url, timeout=30)
        obs_response.raise_for_status()
        observation_count = obs_response.json().get("total_results", 0)

        species_url = (
            f"https://api.inaturalist.org/v1/observations/species_counts"
            f"?project_id={project_id}&per_page=0"
        )
        species_response = requests.get(species_url, timeout=30)
        species_response.raise_for_status()
        species_count = species_response.json().get("total_results", 0)

        observers_url = (
            f"https://api.inaturalist.org/v1/observations/observers"
            f"?project_id={project_id}&per_page=0"
        )
        observers_response = requests.get(observers_url, timeout=30)
        observers_response.raise_for_status()
        observers_count = observers_response.json().get("total_results", 0)

        return ProjectStats(
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
    except CatalogError:
        raise
    except Exception as e:
        raise CatalogError(f"Failed to get project stats: {e}") from e


def get_observation_info(observation_id: int) -> ObservationInfo:
    """Get information about a specific observation.

    Raises:
        CatalogError: if the observation is not found or an API call fails.
    """
    try:
        response = get_observations(id=observation_id)
        results = response.get("results", [])
        if not results:
            raise CatalogError(f"Observation {observation_id} not found")

        obs = results[0]
        taxon = obs.get("taxon", {})
        return ObservationInfo(
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
    except CatalogError:
        raise
    except Exception as e:
        raise CatalogError(f"Failed to get observation: {e}") from e

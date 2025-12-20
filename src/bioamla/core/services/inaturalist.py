"""
iNaturalist Audio Data Importer
===============================

This module provides utilities for importing audio observation data from iNaturalist.
It uses the pyinaturalist library to query observations with sounds and download
audio files for use in bioacoustic machine learning workflows.

Example usage:
    from bioamla.inat import download_inat_audio

    # Download bird sounds from a specific place
    download_inat_audio(
        output_dir="./bird_sounds",
        taxon_id=3,  # Birds (Aves)
        place_id=1,  # United States
        quality_grade="research",
        max_observations=100
    )
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Union

import requests
from pyinaturalist import get_observation_species_counts, get_observations

from bioamla.core.files import (
    BinaryFile,
    TextFile,
    get_extension_from_content_type,
    get_extension_from_url,
    sanitize_filename,
)
from bioamla.core.metadata import (
    get_existing_observation_ids,
    read_metadata_csv,
    write_metadata_csv,
)

logger = logging.getLogger(__name__)


def load_taxon_ids_from_csv(csv_path: Union[str, Path]) -> List[int]:
    """
    Load taxon IDs from a CSV file.

    The CSV file should have a column named 'taxon_id' containing integer taxon IDs.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of taxon IDs as integers

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV doesn't have a 'taxon_id' column or contains invalid data

    Example CSV format:
        taxon_id
        65489
        23456
        78901
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

        for row_num, row in enumerate(reader, start=2):  # start=2 accounts for header
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


def download_inat_audio(
    output_dir: str,
    taxon_ids: Optional[List[int]] = None,
    taxon_csv: Optional[Union[str, Path]] = None,
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
    include_inat_metadata: bool = False,
    file_extensions: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """
    Download audio files from iNaturalist observations.

    This function queries the iNaturalist API for observations containing sounds
    and downloads the audio files to a local directory. It also creates a metadata
    CSV file with information about each downloaded audio file.

    Args:
        output_dir: Directory where audio files will be saved
        taxon_ids: Filter by taxon ID(s) (e.g., [3] for birds/Aves, [3, 20978] for multiple taxa)
        taxon_csv: Path to a CSV file containing taxon IDs (must have a 'taxon_id' column).
            If provided along with taxon_ids, the CSV IDs are appended to taxon_ids.
        taxon_name: Filter by taxon name (e.g., "Aves" for birds)
        place_id: Filter by place ID (e.g., 1 for United States)
        user_id: Filter by observer username
        project_id: Filter by iNaturalist project ID or slug (e.g., "appalachia-bioacoustics")
        quality_grade: Filter by quality grade ("research", "needs_id", or "casual")
        sound_license: Filter by sound license(s) (e.g., ["cc-by", "cc-by-nc", "cc0"])
        d1: Start date for observation date range (YYYY-MM-DD format)
        d2: End date for observation date range (YYYY-MM-DD format)
        obs_per_taxon: Number of observations to download per taxon ID
        per_page: Number of results per API request (max 200)
        delay_between_downloads: Seconds to wait between file downloads (rate limiting)
        organize_by_taxon: If True, organize files into subdirectories by species
        include_inat_metadata: If True, include additional iNaturalist metadata fields
            (observation_id, sound_id, common_name, taxon_id, observed_on, location,
            place_guess, observer, quality_grade, observation_url)
        file_extensions: List of file extensions to filter by (e.g., ["wav", "mp3"]).
            If None, all audio formats are downloaded.
        verbose: If True, print progress information

    Returns:
        dict: Summary statistics including:
            - total_observations: Number of observations processed
            - total_sounds: Number of audio files downloaded
            - skipped_existing: Number of files skipped (already in collection)
            - failed_downloads: Number of failed downloads
            - output_dir: Path to output directory
            - metadata_file: Path to metadata CSV file

    Raises:
        ValueError: If output_dir cannot be created
        FileNotFoundError: If taxon_csv is provided but the file doesn't exist
        ValueError: If taxon_csv doesn't have a 'taxon_id' column

    Example:
        >>> stats = download_inat_audio(
        ...     output_dir="./frog_sounds",
        ...     taxon_name="Anura",
        ...     place_id=1,
        ...     max_observations=50
        ... )
        >>> print(f"Downloaded {stats['total_sounds']} audio files")

        >>> # Using a CSV file with taxon IDs
        >>> stats = download_inat_audio(
        ...     output_dir="./sounds",
        ...     taxon_csv="taxa.csv",
        ...     obs_per_taxon=10
        ... )
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load taxon IDs from CSV if provided
    if taxon_csv:
        csv_taxon_ids = load_taxon_ids_from_csv(taxon_csv)
        if taxon_ids:
            # Combine provided taxon_ids with CSV taxon_ids
            taxon_ids = list(taxon_ids) + csv_taxon_ids
        else:
            taxon_ids = csv_taxon_ids
        if verbose:
            print(f"Loaded {len(csv_taxon_ids)} taxon IDs from CSV file")

    stats = {
        "total_observations": 0,
        "total_sounds": 0,
        "skipped_existing": 0,
        "failed_downloads": 0,
        "output_dir": str(output_path.absolute()),
        "metadata_file": str(output_path / "metadata.csv"),
    }

    metadata_rows = []

    # Load existing metadata to skip already-downloaded files
    existing_files = get_existing_observation_ids(output_path / "metadata.csv")
    if verbose and existing_files:
        print(f"Found {len(existing_files)} existing files in collection, will skip duplicates.")

    # Normalize sound_license to uppercase for pyinaturalist API compatibility
    normalized_license = None
    if sound_license:
        normalized_license = [lic.upper() for lic in sound_license]

    # Normalize file extensions (ensure they start with a dot and are lowercase)
    normalized_extensions = None
    if file_extensions:
        normalized_extensions = [
            ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in file_extensions
        ]

    # Build list of taxon IDs to iterate over
    # If taxon_ids provided, iterate over each; otherwise use None (single query)
    taxon_list = taxon_ids if taxon_ids else [None]

    for current_taxon_id in taxon_list:
        page = 1
        observations_processed = 0

        if verbose:
            if current_taxon_id:
                print(f"Querying iNaturalist for taxon ID {current_taxon_id}...")
            else:
                print("Querying iNaturalist for observations with sounds...")

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
                if verbose:
                    print("No more observations found.")
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

                # For subspecies/varieties, use the parent species name for grouping
                # The rank field tells us if this is a subspecies, variety, etc.
                if taxon_rank in ("subspecies", "variety", "form"):
                    # Try to get the species-level name from ancestors or by truncating
                    ancestors = taxon.get("ancestors", [])
                    # Find the species-level ancestor
                    for ancestor in reversed(ancestors):
                        if ancestor.get("rank") == "species":
                            species_name = ancestor.get("name", species_name)
                            break
                    else:
                        # Fallback: take first two words (genus + species) from subspecies name
                        name_parts = species_name.split()
                        if len(name_parts) >= 2:
                            species_name = " ".join(name_parts[:2])

                observed_on = obs.get("observed_on", "")
                location = obs.get("location", "")
                place_guess = obs.get("place_guess", "")
                user = obs.get("user", {}).get("login", "")
                quality = obs.get("quality_grade", "")

                # Sanitize species name for folder and category consistency
                safe_species = sanitize_filename(species_name)

                if organize_by_taxon:
                    species_dir = output_path / safe_species
                    species_dir.mkdir(exist_ok=True)
                else:
                    species_dir = output_path

                for sound in sounds:
                    sound_id = sound.get("id")
                    file_url = sound.get("file_url")
                    license_code = sound.get("license_code", "")

                    if not file_url:
                        continue

                    ext = get_extension_from_url(file_url)

                    # Skip files that don't match the requested extensions
                    if normalized_extensions and ext.lower() not in normalized_extensions:
                        if verbose:
                            print(f"  Skipped: {ext} file (filtering for {normalized_extensions})")
                        continue

                    # Skip files that already exist in the collection
                    # (metadata for existing files is preserved from the read at the start)
                    if (obs_id, sound_id) in existing_files:
                        if verbose:
                            print(f"  Skipped: inat_{obs_id}_sound_{sound_id} (already exists)")
                        stats["skipped_existing"] += 1
                        continue

                    filename = f"inat_{obs_id}_sound_{sound_id}{ext}"
                    filepath = species_dir / filename

                    success = _download_file(file_url, filepath, verbose)

                    if success:
                        stats["total_sounds"] += 1

                        relative_path = filepath.relative_to(output_path)

                        # Default metadata headers
                        row = {
                            "file_name": str(relative_path),
                            "split": "train",
                            "target": taxon_id_val,
                            "label": safe_species,
                            "attr_id": user,
                            "attr_lic": license_code,
                            "attr_url": file_url,
                            "attr_note": "",
                        }

                        # Optional iNaturalist metadata
                        if include_inat_metadata:
                            row.update(
                                {
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
                                }
                            )

                        metadata_rows.append(row)
                    else:
                        stats["failed_downloads"] += 1

                    time.sleep(delay_between_downloads)

                observations_processed += 1
                stats["total_observations"] += 1

                if verbose and observations_processed % 10 == 0:
                    print(
                        f"Processed {observations_processed}/{obs_per_taxon} observations for current taxon..."
                    )

            page += 1

            if len(results) < current_per_page:
                break

    if metadata_rows:
        metadata_path = output_path / "metadata.csv"
        # Read existing metadata and merge with new rows for consistent file
        existing_rows, existing_fieldnames = read_metadata_csv(metadata_path)

        # Combine existing and new rows, deduplicating by file_name
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

        # Rewrite entire file for consistency
        write_metadata_csv(metadata_path, all_rows, merge_existing=False)

    if verbose:
        print("\nDownload complete!")
        print(f"  Observations processed: {stats['total_observations']}")
        print(f"  Audio files downloaded: {stats['total_sounds']}")
        print(f"  Skipped (already exist): {stats['skipped_existing']}")
        print(f"  Failed downloads: {stats['failed_downloads']}")
        print(f"  Output directory: {stats['output_dir']}")
        print(f"  Metadata file: {stats['metadata_file']}")

    return stats


def search_inat_sounds(
    taxon_id: Optional[int] = None,
    taxon_name: Optional[str] = None,
    place_id: Optional[int] = None,
    quality_grade: Optional[str] = "research",
    per_page: int = 30,
) -> list:
    """
    Search for iNaturalist observations with sounds without downloading.

    This function queries the iNaturalist API and returns observation data
    for preview before downloading.

    Args:
        taxon_id: Filter by taxon ID
        taxon_name: Filter by taxon name
        place_id: Filter by place ID
        quality_grade: Filter by quality grade
        per_page: Number of results to return

    Returns:
        list: List of observation dictionaries containing sound information

    Example:
        >>> results = search_inat_sounds(taxon_name="Strix varia", per_page=5)
        >>> for obs in results:
        ...     print(f"{obs['taxon']['name']}: {len(obs['sounds'])} sounds")
    """
    response = get_observations(
        sounds=True,
        taxon_id=taxon_id,
        taxon_name=taxon_name,
        place_id=place_id,
        quality_grade=quality_grade,
        per_page=per_page,
    )

    return response.get("results", [])


def get_observation_sounds(observation_id: int) -> list:
    """
    Get all sounds from a specific iNaturalist observation.

    Args:
        observation_id: The iNaturalist observation ID

    Returns:
        list: List of sound dictionaries with file_url, license, etc.

    Example:
        >>> sounds = get_observation_sounds(12345678)
        >>> for sound in sounds:
        ...     print(sound['file_url'])
    """
    response = get_observations(id=observation_id)
    results = response.get("results", [])

    if results:
        return results[0].get("sounds", [])
    return []


def get_taxa(
    place_id: Optional[int] = None,
    project_id: Optional[str] = None,
    quality_grade: Optional[str] = "research",
    taxon_id: Optional[int] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Find all taxa from observations in a place or project.

    Uses the iNaturalist species_counts API for efficient retrieval.

    Args:
        place_id: Filter by place ID (e.g., 1 for United States)
        project_id: Filter by iNaturalist project ID or slug
        quality_grade: Filter by quality grade ("research", "needs_id", or "casual")
        taxon_id: Filter by parent taxon ID (e.g., 20979 for Amphibia)
        verbose: If True, print progress information

    Returns:
        List of dictionaries containing taxon information:
            - taxon_id: The iNaturalist taxon ID
            - name: Scientific name
            - common_name: Common name (if available)
            - observation_count: Number of observations found

    Example:
        >>> taxa = get_taxa(project_id="appalachia-bioacoustics", taxon_id=20979)
        >>> for t in taxa:
        ...     print(f"{t['name']} ({t['common_name']}): {t['observation_count']} observations")
    """
    if not place_id and not project_id:
        raise ValueError("At least one of place_id or project_id must be provided")

    if verbose:
        print("Fetching species counts from iNaturalist...")

    taxa_list = []
    page = 1
    per_page = 500

    while True:
        response = get_observation_species_counts(
            place_id=place_id,
            project_id=project_id,
            quality_grade=quality_grade,
            taxon_id=taxon_id,
            page=page,
            per_page=per_page,
        )

        results = response.get("results", [])

        if not results:
            break

        for item in results:
            taxon = item.get("taxon", {})
            taxa_list.append(
                {
                    "taxon_id": taxon.get("id"),
                    "name": taxon.get("name", "unknown"),
                    "common_name": taxon.get("preferred_common_name", ""),
                    "observation_count": item.get("count", 0),
                }
            )

        if verbose:
            print(f"  Fetched {len(taxa_list)} taxa...")

        if len(results) < per_page:
            break

        page += 1

    # Sort by observation count descending
    taxa_list.sort(key=lambda x: x["observation_count"], reverse=True)

    if verbose:
        print(f"Found {len(taxa_list)} unique taxa")

    return taxa_list


def get_project_stats(project_id: str, verbose: bool = True) -> dict[str, Any]:
    """
    Get statistics for an iNaturalist project.

    Queries the iNaturalist API for project information including
    observation counts, species counts, and observer information.

    Args:
        project_id: iNaturalist project ID or slug (e.g., "appalachia-bioacoustics")
        verbose: If True, print progress information

    Returns:
        Dictionary containing project statistics:
            - id: Project ID
            - title: Project title
            - description: Project description
            - slug: Project slug
            - observation_count: Total number of observations
            - species_count: Number of species observed
            - observers_count: Number of observers
            - created_at: Project creation date
            - project_type: Type of project
            - place: Place information (if applicable)
            - url: URL to the project page

    Example:
        >>> stats = get_project_stats("appalachia-bioacoustics")
        >>> print(f"{stats['title']}: {stats['observation_count']} observations")
    """
    if verbose:
        print(f"Fetching project stats for '{project_id}'...")

    # Get project metadata
    project_url = f"https://api.inaturalist.org/v1/projects/{project_id}"
    response = requests.get(project_url, timeout=30)
    response.raise_for_status()

    data = response.json()
    results = data.get("results", [])

    if not results:
        raise ValueError(f"Project '{project_id}' not found")

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
    observers_url = (
        f"https://api.inaturalist.org/v1/observations/observers?project_id={project_id}&per_page=0"
    )
    observers_response = requests.get(observers_url, timeout=30)
    observers_response.raise_for_status()
    observers_count = observers_response.json().get("total_results", 0)

    stats = {
        "id": project.get("id"),
        "title": project.get("title", ""),
        "description": project.get("description", ""),
        "slug": project.get("slug", ""),
        "observation_count": observation_count,
        "species_count": species_count,
        "observers_count": observers_count,
        "created_at": project.get("created_at", ""),
        "project_type": project.get("project_type", ""),
        "place": project.get("place", {}).get("display_name", "") if project.get("place") else "",
        "url": f"https://www.inaturalist.org/projects/{project.get('slug', project_id)}",
    }

    if verbose:
        print(f"  Title: {stats['title']}")
        print(f"  Observations: {stats['observation_count']}")
        print(f"  Species: {stats['species_count']}")
        print(f"  Observers: {stats['observers_count']}")

    return stats


def _download_file(url: str, filepath: Path, verbose: bool = True) -> bool:
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
            print(f"  Downloaded: {filepath.name}")
        return True

    except requests.RequestException as e:
        if verbose:
            print(f"  Failed to download {url}: {e}")
        logger.warning(f"Failed to download {url}: {e}")
        return False

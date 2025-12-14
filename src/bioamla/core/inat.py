"""
iNaturalist Audio Data Importer
===============================

This module provides utilities for importing audio observation data from iNaturalist.
It uses the pyinaturalist library to query observations with sounds and download
audio files for use in bioacoustic machine learning workflows.

Example usage:
    from bioamla.core.wrappers.inat import download_inat_audio

    # Download bird sounds from a specific place
    download_inat_audio(
        output_dir="./bird_sounds",
        taxon_id=3,  # Birds (Aves)
        place_id=1,  # United States
        quality_grade="research",
        max_observations=100
    )
"""

import time
import csv
import warnings
from typing import Any, Optional, List, Set
from pathlib import Path

import requests
from pyinaturalist import get_observations, get_observation_species_counts


# Required metadata fields that must always be present
_REQUIRED_METADATA_FIELDS = [
    "filename", "split", "target", "category",
    "attr_id", "attr_lic", "attr_url", "attr_note"
]

# Optional iNaturalist metadata fields
_OPTIONAL_METADATA_FIELDS = [
    "observation_id", "sound_id", "common_name", "taxon_id",
    "observed_on", "location", "place_guess", "observer",
    "quality_grade", "observation_url"
]


def download_inat_audio(
    output_dir: str,
    taxon_ids: Optional[List[int]] = None,
    taxon_name: Optional[str] = None,
    place_id: Optional[int] = None,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    quality_grade: Optional[str] = "research",
    sound_license: Optional[str] = None,
    d1: Optional[str] = None,
    d2: Optional[str] = None,
    obs_per_taxon: int = 100,
    per_page: int = 30,
    delay_between_downloads: float = 1.0,
    organize_by_taxon: bool = True,
    include_inat_metadata: bool = False,
    file_extensions: Optional[List[str]] = None,
    verbose: bool = True
) -> dict:
    """
    Download audio files from iNaturalist observations.

    This function queries the iNaturalist API for observations containing sounds
    and downloads the audio files to a local directory. It also creates a metadata
    CSV file with information about each downloaded audio file.

    Args:
        output_dir: Directory where audio files will be saved
        taxon_ids: Filter by taxon ID(s) (e.g., [3] for birds/Aves, [3, 20978] for multiple taxa)
        taxon_name: Filter by taxon name (e.g., "Aves" for birds)
        place_id: Filter by place ID (e.g., 1 for United States)
        user_id: Filter by observer username
        project_id: Filter by iNaturalist project ID or slug (e.g., "appalachia-bioacoustics")
        quality_grade: Filter by quality grade ("research", "needs_id", or "casual")
        sound_license: Filter by sound license (e.g., "cc-by", "cc-by-nc", "cc0")
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

    Example:
        >>> stats = download_inat_audio(
        ...     output_dir="./frog_sounds",
        ...     taxon_name="Anura",
        ...     place_id=1,
        ...     max_observations=50
        ... )
        >>> print(f"Downloaded {stats['total_sounds']} audio files")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_observations": 0,
        "total_sounds": 0,
        "skipped_existing": 0,
        "failed_downloads": 0,
        "output_dir": str(output_path.absolute()),
        "metadata_file": str(output_path / "metadata.csv")
    }

    metadata_rows = []

    # Load existing metadata to skip already-downloaded files
    existing_files = _get_existing_files(output_path / "metadata.csv")
    if verbose and existing_files:
        print(f"Found {len(existing_files)} existing files in collection, will skip duplicates.")

    # Normalize sound_license to uppercase for pyinaturalist API compatibility
    normalized_license = sound_license.upper() if sound_license else None

    # Normalize file extensions (ensure they start with a dot and are lowercase)
    normalized_extensions = None
    if file_extensions:
        normalized_extensions = [
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in file_extensions
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
                print(f"Querying iNaturalist for observations with sounds...")

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
                per_page=current_per_page
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

                observed_on = obs.get("observed_on", "")
                location = obs.get("location", "")
                place_guess = obs.get("place_guess", "")
                user = obs.get("user", {}).get("login", "")
                quality = obs.get("quality_grade", "")

                if organize_by_taxon:
                    safe_species = _sanitize_filename(species_name)
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

                    ext = _get_extension_from_url(file_url)

                    # Skip files that don't match the requested extensions
                    if normalized_extensions and ext.lower() not in normalized_extensions:
                        if verbose:
                            print(f"  Skipped: {ext} file (filtering for {normalized_extensions})")
                        continue

                    # Skip files that already exist in the collection
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
                            "filename": str(relative_path),
                            "split": "train",
                            "target": taxon_id_val,
                            "category": species_name,
                            "attr_id": user,
                            "attr_lic": license_code,
                            "attr_url": file_url,
                            "attr_note": ""
                        }

                        # Optional iNaturalist metadata
                        if include_inat_metadata:
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
                                "observation_url": f"https://www.inaturalist.org/observations/{obs_id}"
                            })

                        metadata_rows.append(row)
                    else:
                        stats["failed_downloads"] += 1

                    time.sleep(delay_between_downloads)

                observations_processed += 1
                stats["total_observations"] += 1

                if verbose and observations_processed % 10 == 0:
                    print(f"Processed {observations_processed}/{obs_per_taxon} observations for current taxon...")

            page += 1

            if len(results) < current_per_page:
                break

    if metadata_rows:
        _write_metadata_csv(output_path / "metadata.csv", metadata_rows, verbose)

    if verbose:
        print(f"\nDownload complete!")
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
    per_page: int = 30
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
        per_page=per_page
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
    verbose: bool = True
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
            per_page=per_page
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
                "observation_count": item.get("count", 0)
            })

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


def get_project_stats(
    project_id: str,
    verbose: bool = True
) -> dict[str, Any]:
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
    observers_url = f"https://api.inaturalist.org/v1/observations/observers?project_id={project_id}&per_page=0"
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
        "url": f"https://www.inaturalist.org/projects/{project.get('slug', project_id)}"
    }

    if verbose:
        print(f"  Title: {stats['title']}")
        print(f"  Observations: {stats['observation_count']}")
        print(f"  Species: {stats['species_count']}")
        print(f"  Observers: {stats['observers_count']}")

    return stats


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename or directory name."""
    invalid_chars = '<>:"/\\|?*'
    sanitized = name.lower()
    sanitized = sanitized.replace(" ", "_")
    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")
    sanitized = sanitized.strip(". ")
    return sanitized if sanitized else "unknown"


def _get_extension_from_url(url: str) -> str:
    """Extract file extension from URL or return default."""
    url_lower = url.lower()
    if ".wav" in url_lower:
        return ".wav"
    elif ".m4a" in url_lower:
        return ".m4a"
    elif ".mp3" in url_lower:
        return ".mp3"
    elif ".ogg" in url_lower:
        return ".ogg"
    elif ".flac" in url_lower:
        return ".flac"
    return ".mp3"


def _download_file(url: str, filepath: Path, verbose: bool = True) -> bool:
    """Download a file from URL to local path."""
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if content_type:
            ext = _get_extension_from_content_type(content_type)
            if ext and not str(filepath).lower().endswith(ext):
                filepath = filepath.with_suffix(ext)

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if verbose:
            print(f"  Downloaded: {filepath.name}")
        return True

    except requests.RequestException as e:
        if verbose:
            print(f"  Failed to download {url}: {e}")
        return False


def _get_extension_from_content_type(content_type: str) -> str:
    """Map content type to file extension."""
    content_type = content_type.lower().split(";")[0].strip()
    mapping = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/m4a": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/mp4": ".m4a",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "audio/x-flac": ".flac",
    }
    return mapping.get(content_type, "")


def _get_existing_files(metadata_path: Path) -> Set[tuple[int, int]]:
    """
    Read existing metadata CSV and extract (observation_id, sound_id) pairs.

    This is used to skip downloading files that already exist in the collection.
    The function parses filenames in the format 'inat_{obs_id}_sound_{sound_id}.ext'
    to extract the IDs.

    Args:
        metadata_path: Path to the metadata.csv file

    Returns:
        Set of (observation_id, sound_id) tuples for existing files
    """
    existing: Set[tuple[int, int]] = set()

    if not metadata_path.exists():
        return existing

    try:
        with open(metadata_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "")
                # Extract obs_id and sound_id from filename pattern: inat_{obs_id}_sound_{sound_id}.ext
                # The filename may include a subdirectory (e.g., "species_name/inat_123_sound_456.mp3")
                basename = Path(filename).name
                if basename.startswith("inat_") and "_sound_" in basename:
                    try:
                        # Parse: inat_{obs_id}_sound_{sound_id}.ext
                        parts = basename.replace("inat_", "").split("_sound_")
                        obs_id = int(parts[0])
                        sound_id = int(parts[1].split(".")[0])
                        existing.add((obs_id, sound_id))
                    except (ValueError, IndexError):
                        # Skip malformed filenames
                        continue
    except (OSError, csv.Error):
        # If we can't read the file, return empty set (will download everything)
        pass

    return existing


def _read_existing_metadata(filepath: Path) -> tuple[list[dict], Set[str]]:
    """Read existing metadata CSV and return rows and fieldnames."""
    rows = []
    fieldnames: Set[str] = set()

    if not filepath.exists():
        return rows, fieldnames

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    return rows, fieldnames


def _write_metadata_csv(filepath: Path, rows: list, verbose: bool = True) -> None:
    """Write metadata rows to a CSV file, merging with existing data if present."""
    if not rows:
        return

    new_fieldnames = set(rows[0].keys())
    existing_rows, existing_fieldnames = _read_existing_metadata(filepath)

    if existing_rows:
        # Check for optional metadata mismatch
        existing_optional = existing_fieldnames & set(_OPTIONAL_METADATA_FIELDS)
        new_optional = new_fieldnames & set(_OPTIONAL_METADATA_FIELDS)

        if existing_optional != new_optional:
            # TODO: Handle optional metadata mismatch more gracefully by allowing
            # users to choose whether to keep, drop, or fill with defaults
            warnings.warn(
                f"Optional metadata mismatch when merging datasets. "
                f"Existing has: {existing_optional or 'none'}, "
                f"New has: {new_optional or 'none'}. "
                f"Dropping optional metadata columns to maintain consistency.",
                UserWarning
            )
            # Drop optional metadata from both existing and new rows
            for row in existing_rows:
                for field in _OPTIONAL_METADATA_FIELDS:
                    row.pop(field, None)
            for row in rows:
                for field in _OPTIONAL_METADATA_FIELDS:
                    row.pop(field, None)

            # Update fieldnames to required only
            final_fieldnames = _REQUIRED_METADATA_FIELDS
        else:
            # Fieldnames match, use required order plus any optional fields
            final_fieldnames = list(_REQUIRED_METADATA_FIELDS)
            optional_in_rows = [f for f in _OPTIONAL_METADATA_FIELDS if f in existing_fieldnames]
            final_fieldnames.extend(optional_in_rows)

        # Get existing file names to avoid duplicates
        seen_files = {row.get("filename") for row in existing_rows}

        # Filter out duplicates from new rows (against existing and within new rows)
        new_unique_rows = []
        for row in rows:
            filename = row.get("filename")
            if filename not in seen_files:
                seen_files.add(filename)
                new_unique_rows.append(row)

        if verbose and len(new_unique_rows) < len(rows):
            skipped = len(rows) - len(new_unique_rows)
            print(f"  Skipped {skipped} duplicate entries during merge")

        # Merge rows
        all_rows = existing_rows + new_unique_rows
    else:
        # Use required fields in defined order, then any optional fields
        final_fieldnames = list(_REQUIRED_METADATA_FIELDS)
        optional_in_rows = [f for f in _OPTIONAL_METADATA_FIELDS if f in rows[0]]
        final_fieldnames.extend(optional_in_rows)
        # Deduplicate rows by filename
        seen_files: Set[str] = set()
        all_rows = []
        for row in rows:
            filename = row.get("filename")
            if filename not in seen_files:
                seen_files.add(filename)
                all_rows.append(row)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

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
from typing import Optional, List, Set
from pathlib import Path

import requests
from pyinaturalist import get_observations


# Required metadata fields that must always be present
_REQUIRED_METADATA_FIELDS = [
    "file_name", "split", "target", "category",
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
    project_id: Optional[int] = None,
    quality_grade: Optional[str] = "research",
    sound_license: Optional[str] = None,
    d1: Optional[str] = None,
    d2: Optional[str] = None,
    max_observations: int = 100,
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
        project_id: Filter by iNaturalist project ID
        quality_grade: Filter by quality grade ("research", "needs_id", or "casual")
        sound_license: Filter by sound license (e.g., "cc-by", "cc-by-nc", "cc0")
        d1: Start date for observation date range (YYYY-MM-DD format)
        d2: End date for observation date range (YYYY-MM-DD format)
        max_observations: Maximum number of observations to download
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
        "failed_downloads": 0,
        "output_dir": str(output_path.absolute()),
        "metadata_file": str(output_path / "metadata.csv")
    }

    metadata_rows = []

    page = 1
    observations_processed = 0

    # Normalize sound_license to uppercase for pyinaturalist API compatibility
    normalized_license = sound_license.upper() if sound_license else None

    # Normalize file extensions (ensure they start with a dot and are lowercase)
    normalized_extensions = None
    if file_extensions:
        normalized_extensions = [
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in file_extensions
        ]

    if verbose:
        print(f"Querying iNaturalist for observations with sounds...")

    while observations_processed < max_observations:
        remaining = max_observations - observations_processed
        current_per_page = min(per_page, remaining)

        response = get_observations(
            sounds=True,
            taxon_id=taxon_ids,
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
            if observations_processed >= max_observations:
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
                print(f"Processed {observations_processed}/{max_observations} observations...")

        page += 1

        if len(results) < current_per_page:
            break

    if metadata_rows:
        _write_metadata_csv(output_path / "metadata.csv", metadata_rows, verbose)

    if verbose:
        print(f"\nDownload complete!")
        print(f"  Observations processed: {stats['total_observations']}")
        print(f"  Audio files downloaded: {stats['total_sounds']}")
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
            # Fieldnames match, use existing order
            final_fieldnames = list(existing_fieldnames)

        # Get existing file names to avoid duplicates
        existing_files = {row.get("file_name") for row in existing_rows}

        # Filter out duplicates from new rows
        new_unique_rows = [row for row in rows if row.get("file_name") not in existing_files]

        if verbose and len(new_unique_rows) < len(rows):
            skipped = len(rows) - len(new_unique_rows)
            print(f"  Skipped {skipped} duplicate entries during merge")

        # Merge rows
        all_rows = existing_rows + new_unique_rows
    else:
        final_fieldnames = list(rows[0].keys())
        all_rows = rows

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

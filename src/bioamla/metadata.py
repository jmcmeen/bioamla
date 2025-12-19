"""
Metadata Management
===================

This module provides centralized functionality for managing audio dataset metadata.
It consolidates metadata field definitions and CSV I/O operations used across
the bioamla package, ensuring consistency in metadata handling.

The module defines standard metadata fields for audio datasets and provides
utilities for reading, writing, and validating metadata CSV files.
"""

import csv
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Set, Tuple

from bioamla.core.files import TextFile

logger = logging.getLogger(__name__)


# Required metadata fields that must always be present in dataset metadata
REQUIRED_FIELDS = [
    "file_name",    # Relative path to the audio file
    "split",        # Dataset split (train, test, val)
    "target",       # Numeric target/label ID
    "label",        # Label/class name (e.g., species name)
    "attr_id",      # Attribution identifier (e.g., observer username)
    "attr_lic",     # License code for the audio
    "attr_url",     # URL to the original source
    "attr_note"     # Additional attribution notes
]

# Optional iNaturalist metadata fields that may be included
OPTIONAL_INAT_FIELDS = [
    "observation_id",    # iNaturalist observation ID
    "sound_id",          # iNaturalist sound ID
    "common_name",       # Common name of the species
    "taxon_id",          # iNaturalist taxon ID
    "observed_on",       # Date of observation
    "location",          # GPS coordinates
    "place_guess",       # Human-readable location description
    "observer",          # Observer username
    "quality_grade",     # iNaturalist quality grade
    "observation_url"    # URL to the observation page
]


def read_metadata_csv(filepath: Path) -> Tuple[List[dict], Set[str]]:
    """
    Read metadata from a CSV file.

    Args:
        filepath: Path to the metadata CSV file

    Returns:
        Tuple containing:
            - List of row dictionaries
            - Set of fieldnames found in the CSV

    Note:
        Returns empty list and empty set if file doesn't exist.
    """
    rows: List[dict] = []
    fieldnames: Set[str] = set()

    if not filepath.exists():
        logger.debug(f"Metadata file not found: {filepath}")
        return rows, fieldnames

    try:
        with TextFile(filepath, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f.handle)
            fieldnames = set(reader.fieldnames or [])
            rows = list(reader)
        logger.debug(f"Read {len(rows)} rows from {filepath}")
    except (OSError, csv.Error) as e:
        logger.warning(f"Error reading metadata file {filepath}: {e}")

    return rows, fieldnames


def write_metadata_csv(
    filepath: Path,
    rows: List[dict],
    fieldnames: Optional[Set[str]] = None,
    merge_existing: bool = True
) -> int:
    """
    Write metadata rows to a CSV file.

    Args:
        filepath: Path to the output CSV file
        rows: List of row dictionaries to write
        fieldnames: Optional set of field names. If None, derived from rows.
        merge_existing: If True, merge with existing data in file

    Returns:
        Number of rows written

    Note:
        When merge_existing is True, rows are deduplicated by file_name.
        Required fields are ordered first, followed by optional fields.
    """
    # Handle empty rows case
    if not rows:
        if merge_existing:
            # When merging, no rows means nothing to add, keep existing
            logger.debug("No rows to write")
            return 0
        else:
            # When not merging, empty rows means clear the file
            logger.debug("Writing empty metadata file (clearing existing)")
            with TextFile(filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f.handle, fieldnames=REQUIRED_FIELDS)
                writer.writeheader()
            return 0

    # Derive fieldnames from rows if not provided
    if fieldnames is None:
        fieldnames = set()
        for row in rows:
            fieldnames.update(row.keys())

    # Handle merging with existing data
    all_rows = rows
    if merge_existing and filepath.exists():
        existing_rows, existing_fieldnames = read_metadata_csv(filepath)
        fieldnames = fieldnames.union(existing_fieldnames)

        # Check for optional metadata mismatch
        existing_optional = existing_fieldnames & set(OPTIONAL_INAT_FIELDS)
        new_optional = set(rows[0].keys()) & set(OPTIONAL_INAT_FIELDS)

        if existing_optional != new_optional and existing_rows:
            warnings.warn(
                f"Optional metadata mismatch when merging datasets. "
                f"Existing has: {existing_optional or 'none'}, "
                f"New has: {new_optional or 'none'}. "
                f"Dropping optional metadata columns to maintain consistency.",
                UserWarning, stacklevel=2
            )
            # Drop optional metadata from both existing and new rows
            for row in existing_rows:
                for field in OPTIONAL_INAT_FIELDS:
                    row.pop(field, None)
            for row in rows:
                for field in OPTIONAL_INAT_FIELDS:
                    row.pop(field, None)
            # Update fieldnames to required only
            fieldnames = set(REQUIRED_FIELDS)

        # Deduplicate by file_name
        seen_files: Set[str] = set()
        deduplicated_rows: List[dict] = []

        for row in existing_rows:
            file_name = row.get("file_name", "")
            if file_name and file_name not in seen_files:
                seen_files.add(file_name)
                deduplicated_rows.append(row)

        skipped = 0
        for row in rows:
            file_name = row.get("file_name", "")
            if file_name and file_name not in seen_files:
                seen_files.add(file_name)
                deduplicated_rows.append(row)
            elif file_name:
                skipped += 1

        if skipped > 0:
            logger.info(f"Skipped {skipped} duplicate entries during merge")

        all_rows = deduplicated_rows

    # Build ordered fieldnames: required fields first, then optional, then rest
    final_fieldnames = []
    for field in REQUIRED_FIELDS:
        if field in fieldnames:
            final_fieldnames.append(field)
            fieldnames.discard(field)

    # Add optional iNat fields in defined order
    for field in OPTIONAL_INAT_FIELDS:
        if field in fieldnames:
            final_fieldnames.append(field)
            fieldnames.discard(field)

    # Add remaining fields in sorted order
    final_fieldnames.extend(sorted(fieldnames))

    # Normalize rows to ensure all fields are present
    normalized_rows = []
    for row in all_rows:
        normalized_row = {field: row.get(field, "") for field in final_fieldnames}
        normalized_rows.append(normalized_row)

    # Write to file
    with TextFile(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f.handle, fieldnames=final_fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    logger.info(f"Wrote {len(normalized_rows)} entries to {filepath}")
    return len(normalized_rows)


def get_existing_observation_ids(metadata_path: Path) -> Set[Tuple[int, int]]:
    """
    Extract (observation_id, sound_id) pairs from existing metadata.

    This is used to skip downloading files that already exist in the collection.
    The function parses filenames in the format 'inat_{obs_id}_sound_{sound_id}.ext'.

    Args:
        metadata_path: Path to the metadata.csv file

    Returns:
        Set of (observation_id, sound_id) tuples for existing files
    """
    existing: Set[Tuple[int, int]] = set()

    if not metadata_path.exists():
        return existing

    try:
        rows, _ = read_metadata_csv(metadata_path)
        for row in rows:
            file_name = row.get("file_name", "")
            # Extract obs_id and sound_id from file_name pattern: inat_{obs_id}_sound_{sound_id}.ext
            basename = Path(file_name).name
            if basename.startswith("inat_") and "_sound_" in basename:
                try:
                    parts = basename.replace("inat_", "").split("_sound_")
                    obs_id = int(parts[0])
                    sound_id = int(parts[1].split(".")[0])
                    existing.add((obs_id, sound_id))
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        logger.warning(f"Error reading existing observation IDs from {metadata_path}: {e}")

    return existing


def validate_required_fields(rows: List[dict]) -> Tuple[bool, List[str]]:
    """
    Validate that all rows contain required metadata fields.

    Args:
        rows: List of metadata row dictionaries

    Returns:
        Tuple containing:
            - True if all rows are valid, False otherwise
            - List of error messages for invalid rows
    """
    errors: List[str] = []

    for i, row in enumerate(rows):
        missing = [field for field in REQUIRED_FIELDS if field not in row or not row[field]]
        if missing:
            # file_name can be empty for some operations, but should warn
            critical_missing = [f for f in missing if f != "attr_note"]
            if critical_missing:
                errors.append(f"Row {i+1}: missing required fields {critical_missing}")

    return len(errors) == 0, errors

"""
Metadata Management
===================

This module provides centralized functionality for managing audio dataset metadata.
It consolidates metadata field definitions and CSV I/O operations used across
the bioamla package, ensuring consistency in metadata handling.

The module defines standard metadata fields for audio datasets and provides
utilities for reading, writing, and validating metadata CSV files.

Field Schema Registry
---------------------
Use the MetadataRegistry to define and compose field schemas from different sources:

    from bioamla.core.metadata import MetadataRegistry

    # Get fields for a specific source
    fields = MetadataRegistry.get_fields("inaturalist")

    # Compose fields from multiple sources
    fields = MetadataRegistry.compose("base", "inaturalist", "acoustic")

    # Register custom fields
    MetadataRegistry.register("custom", [
        FieldDef("my_field", "Description", required=True),
    ])
"""

import csv
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from bioamla.core.files import TextFile

logger = logging.getLogger(__name__)


# =============================================================================
# Field Definition and Registry
# =============================================================================


@dataclass
class FieldDef:
    """Definition of a metadata field."""

    name: str
    description: str = ""
    required: bool = False
    default: Any = ""
    dtype: str = "str"  # str, int, float, bool, list

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "dtype": self.dtype,
        }


class MetadataRegistry:
    """
    Registry for composable metadata field schemas.

    Allows defining field sets for different data sources (iNaturalist, Xeno-canto,
    Macaulay, etc.) and composing them dynamically.

    Example:
        # Get all fields for iNaturalist data
        fields = MetadataRegistry.get_fields("inaturalist")

        # Compose base + source-specific + acoustic analysis fields
        fields = MetadataRegistry.compose("base", "inaturalist", "acoustic")

        # Get ordered field names for CSV header
        headers = MetadataRegistry.get_headers("base", "inaturalist")

        # Register custom fields
        MetadataRegistry.register("my_source", [
            FieldDef("custom_id", "Custom identifier", required=True),
            FieldDef("custom_score", "Score value", dtype="float"),
        ])
    """

    _registry: Dict[str, List[FieldDef]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Initialize built-in field schemas."""
        if cls._initialized:
            return

        # Base fields - always required for any audio dataset
        cls.register(
            "base",
            [
                FieldDef("file_name", "Relative path to audio file", required=True),
                FieldDef(
                    "split", "Dataset split (train, test, val)", required=True, default="train"
                ),
                FieldDef("target", "Numeric target/label ID", required=True),
                FieldDef("label", "Label/class name (e.g., species)", required=True),
            ],
        )

        # Attribution fields - for proper crediting
        cls.register(
            "attribution",
            [
                FieldDef("attr_id", "Attribution identifier (observer/recordist)"),
                FieldDef("attr_lic", "License code"),
                FieldDef("attr_url", "URL to original source"),
                FieldDef("attr_note", "Additional attribution notes"),
            ],
        )

        # iNaturalist-specific fields
        cls.register(
            "inaturalist",
            [
                FieldDef("observation_id", "iNaturalist observation ID", dtype="int"),
                FieldDef("sound_id", "iNaturalist sound ID", dtype="int"),
                FieldDef("common_name", "Common name of species"),
                FieldDef("taxon_id", "iNaturalist taxon ID", dtype="int"),
                FieldDef("observed_on", "Date of observation"),
                FieldDef("location", "GPS coordinates (lat,lon)"),
                FieldDef("place_guess", "Human-readable location"),
                FieldDef("observer", "Observer username"),
                FieldDef("quality_grade", "Quality grade (research, needs_id, casual)"),
                FieldDef("observation_url", "URL to observation page"),
            ],
        )

        # Xeno-canto-specific fields
        cls.register(
            "xeno_canto",
            [
                FieldDef("xc_id", "Xeno-canto recording ID", dtype="int"),
                FieldDef("common_name", "Common name of species"),
                FieldDef("genus", "Genus name"),
                FieldDef("species", "Species epithet"),
                FieldDef("subspecies", "Subspecies name"),
                FieldDef("recordist", "Recordist name"),
                FieldDef("country", "Country of recording"),
                FieldDef("locality", "Recording locality"),
                FieldDef("latitude", "Latitude", dtype="float"),
                FieldDef("longitude", "Longitude", dtype="float"),
                FieldDef("elevation", "Elevation in meters", dtype="int"),
                FieldDef("vocalization_type", "Type of vocalization (song, call, etc.)"),
                FieldDef("quality", "Quality rating (A, B, C, D, E)"),
                FieldDef("length", "Recording length in seconds", dtype="float"),
                FieldDef("sample_rate", "Sample rate in Hz", dtype="int"),
                FieldDef("bitrate", "Bitrate in kbps", dtype="int"),
                FieldDef("xc_url", "URL to Xeno-canto page"),
                FieldDef("download_url", "Direct download URL"),
                FieldDef("recording_date", "Date of recording"),
                FieldDef("recording_time", "Time of recording"),
                FieldDef("remarks", "Recordist remarks"),
            ],
        )

        # Macaulay Library-specific fields
        cls.register(
            "macaulay",
            [
                FieldDef("ml_catalog_id", "Macaulay Library catalog ID"),
                FieldDef("common_name", "Common name of species"),
                FieldDef("scientific_name", "Scientific name"),
                FieldDef("recordist", "Recordist name"),
                FieldDef("country", "Country of recording"),
                FieldDef("state_province", "State or province"),
                FieldDef("county", "County or district"),
                FieldDef("locality", "Specific locality"),
                FieldDef("latitude", "Latitude", dtype="float"),
                FieldDef("longitude", "Longitude", dtype="float"),
                FieldDef("recording_date", "Date of recording"),
                FieldDef("duration", "Duration in seconds", dtype="float"),
                FieldDef("ml_url", "URL to Macaulay Library page"),
            ],
        )

        # Acoustic analysis fields
        cls.register(
            "acoustic",
            [
                FieldDef("duration", "Duration in seconds", dtype="float"),
                FieldDef("sample_rate", "Sample rate in Hz", dtype="int"),
                FieldDef("channels", "Number of channels", dtype="int"),
                FieldDef("bit_depth", "Bit depth", dtype="int"),
                FieldDef("aci", "Acoustic Complexity Index", dtype="float"),
                FieldDef("adi", "Acoustic Diversity Index", dtype="float"),
                FieldDef("aei", "Acoustic Evenness Index", dtype="float"),
                FieldDef("bio", "Bioacoustic Index", dtype="float"),
                FieldDef("ndsi", "Normalized Difference Soundscape Index", dtype="float"),
                FieldDef("h_spectral", "Spectral entropy", dtype="float"),
                FieldDef("h_temporal", "Temporal entropy", dtype="float"),
            ],
        )

        # Detection/inference fields
        cls.register(
            "detection",
            [
                FieldDef("start_time", "Detection start time in seconds", dtype="float"),
                FieldDef("end_time", "Detection end time in seconds", dtype="float"),
                FieldDef("confidence", "Detection confidence score", dtype="float"),
                FieldDef("predicted_label", "Model prediction"),
                FieldDef("model_name", "Name of detection model"),
                FieldDef("model_version", "Version of detection model"),
            ],
        )

        # Embedding fields
        cls.register(
            "embedding",
            [
                FieldDef("embedding_model", "Model used for embedding"),
                FieldDef("embedding_dim", "Embedding dimensionality", dtype="int"),
                FieldDef("embedding_path", "Path to embedding file"),
                FieldDef("cluster_id", "Cluster assignment", dtype="int"),
                FieldDef("cluster_distance", "Distance to cluster center", dtype="float"),
            ],
        )

        # Annotation fields
        cls.register(
            "annotation",
            [
                FieldDef("annotation_id", "Unique annotation ID"),
                FieldDef("annotator", "Name or ID of annotator"),
                FieldDef("annotation_date", "Date of annotation"),
                FieldDef("annotation_type", "Type (bbox, segment, point, tag)"),
                FieldDef("start_time", "Start time in seconds", dtype="float"),
                FieldDef("end_time", "End time in seconds", dtype="float"),
                FieldDef("low_freq", "Low frequency in Hz", dtype="float"),
                FieldDef("high_freq", "High frequency in Hz", dtype="float"),
                FieldDef("notes", "Annotation notes"),
            ],
        )

        cls._initialized = True

    @classmethod
    def register(
        cls,
        source: str,
        fields: List[FieldDef],
        replace: bool = False,
    ) -> None:
        """
        Register a field schema for a source.

        Args:
            source: Source name (e.g., "inaturalist", "xeno_canto")
            fields: List of FieldDef objects
            replace: If True, replace existing; if False, extend
        """
        if source in cls._registry and not replace:
            cls._registry[source].extend(fields)
        else:
            cls._registry[source] = list(fields)

    @classmethod
    def get_fields(cls, source: str) -> List[FieldDef]:
        """Get field definitions for a source."""
        cls._ensure_initialized()
        return cls._registry.get(source, [])

    @classmethod
    def get_field_names(cls, source: str) -> List[str]:
        """Get field names for a source."""
        return [f.name for f in cls.get_fields(source)]

    @classmethod
    def get_required_fields(cls, source: str) -> List[str]:
        """Get required field names for a source."""
        return [f.name for f in cls.get_fields(source) if f.required]

    @classmethod
    def compose(cls, *sources: str) -> List[FieldDef]:
        """
        Compose fields from multiple sources.

        Args:
            *sources: Source names to compose

        Returns:
            Combined list of FieldDef objects (deduplicated by name)
        """
        cls._ensure_initialized()
        seen: Set[str] = set()
        result: List[FieldDef] = []

        for source in sources:
            for field_def in cls.get_fields(source):
                if field_def.name not in seen:
                    seen.add(field_def.name)
                    result.append(field_def)

        return result

    @classmethod
    def get_headers(cls, *sources: str) -> List[str]:
        """
        Get ordered header names for CSV output.

        Args:
            *sources: Source names to include

        Returns:
            List of field names in order
        """
        return [f.name for f in cls.compose(*sources)]

    @classmethod
    def list_sources(cls) -> List[str]:
        """List all registered sources."""
        cls._ensure_initialized()
        return list(cls._registry.keys())

    @classmethod
    def describe_source(cls, source: str) -> Dict[str, Any]:
        """Get description of a source's fields."""
        fields = cls.get_fields(source)
        return {
            "source": source,
            "num_fields": len(fields),
            "required": [f.name for f in fields if f.required],
            "optional": [f.name for f in fields if not f.required],
            "fields": [f.to_dict() for f in fields],
        }

    @classmethod
    def validate_row(
        cls,
        row: Dict[str, Any],
        *sources: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a row against field schema.

        Args:
            row: Row dictionary to validate
            *sources: Sources to validate against

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: List[str] = []
        fields = cls.compose(*sources)

        for field_def in fields:
            if field_def.required and field_def.name not in row:
                errors.append(f"Missing required field: {field_def.name}")
            elif field_def.required and not row.get(field_def.name):
                errors.append(f"Empty required field: {field_def.name}")

        return len(errors) == 0, errors

    @classmethod
    def normalize_row(
        cls,
        row: Dict[str, Any],
        *sources: str,
    ) -> Dict[str, Any]:
        """
        Normalize a row to include all fields with defaults.

        Args:
            row: Row dictionary to normalize
            *sources: Sources to include fields from

        Returns:
            Normalized row with all fields
        """
        fields = cls.compose(*sources)
        result = {}

        for field_def in fields:
            if field_def.name in row:
                result[field_def.name] = row[field_def.name]
            else:
                result[field_def.name] = field_def.default

        return result


# =============================================================================
# Legacy compatibility - map to registry
# =============================================================================

# Required metadata fields that must always be present in dataset metadata
REQUIRED_FIELDS = [
    "file_name",  # Relative path to the audio file
    "split",  # Dataset split (train, test, val)
    "target",  # Numeric target/label ID
    "label",  # Label/class name (e.g., species name)
    "attr_id",  # Attribution identifier (e.g., observer username)
    "attr_lic",  # License code for the audio
    "attr_url",  # URL to the original source
    "attr_note",  # Additional attribution notes
]

# Optional iNaturalist metadata fields that may be included
OPTIONAL_INAT_FIELDS = [
    "observation_id",  # iNaturalist observation ID
    "sound_id",  # iNaturalist sound ID
    "common_name",  # Common name of the species
    "taxon_id",  # iNaturalist taxon ID
    "observed_on",  # Date of observation
    "location",  # GPS coordinates
    "place_guess",  # Human-readable location description
    "observer",  # Observer username
    "quality_grade",  # iNaturalist quality grade
    "observation_url",  # URL to the observation page
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
    merge_existing: bool = True,
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
                UserWarning,
                stacklevel=2,
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
                errors.append(f"Row {i + 1}: missing required fields {critical_missing}")

    return len(errors) == 0, errors

"""Metadata CSV helpers for dataset merging.

Direct pathlib/csv implementations of the metadata read/write utilities the
dataset merge pipeline needs. De-layered from the legacy core metadata
module so the datasets package owns its own I/O.
"""

import csv
import logging
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

# Standard dataset metadata fields (required first, then optional iNat fields).
REQUIRED_FIELDS = [
    "file_name",
    "split",
    "target",
    "label",
    "attr_id",
    "attr_lic",
    "attr_url",
    "attr_note",
]

OPTIONAL_INAT_FIELDS = [
    "observation_id",
    "sound_id",
    "common_name",
    "taxon_id",
    "observed_on",
    "location",
    "place_guess",
    "observer",
    "quality_grade",
    "observation_url",
]


def read_metadata_csv(filepath: Path) -> tuple[list[dict], set[str]]:
    """Read metadata rows and field names from a CSV file.

    Returns an empty list and empty set if the file does not exist.
    """
    rows: list[dict] = []
    fieldnames: set[str] = set()

    if not filepath.exists():
        return rows, fieldnames

    try:
        with filepath.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, csv.Error) as e:
        logger.warning(f"Error reading metadata file {filepath}: {e}")

    return rows, fieldnames


def write_metadata_csv(
    filepath: Path,
    rows: list[dict],
    fieldnames: set[str] | None = None,
    merge_existing: bool = True,
) -> int:
    """Write metadata rows to a CSV file.

    When ``merge_existing`` is True, rows are merged with and deduplicated
    against existing data by ``file_name``. Required fields are ordered first,
    followed by optional iNaturalist fields, then any remaining fields.
    """
    if not rows:
        if merge_existing:
            return 0
        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
        return 0

    if fieldnames is None:
        fieldnames = set()
        for row in rows:
            fieldnames.update(row.keys())

    all_rows = rows
    if merge_existing and filepath.exists():
        existing_rows, existing_fieldnames = read_metadata_csv(filepath)
        fieldnames = fieldnames.union(existing_fieldnames)

        existing_optional = existing_fieldnames & set(OPTIONAL_INAT_FIELDS)
        new_optional = set(rows[0].keys()) & set(OPTIONAL_INAT_FIELDS)

        if existing_optional != new_optional and existing_rows:
            warnings.warn(
                "Optional metadata mismatch when merging datasets. "
                f"Existing has: {existing_optional or 'none'}, "
                f"New has: {new_optional or 'none'}. "
                "Dropping optional metadata columns to maintain consistency.",
                UserWarning,
                stacklevel=2,
            )
            for row in existing_rows:
                for fld in OPTIONAL_INAT_FIELDS:
                    row.pop(fld, None)
            for row in rows:
                for fld in OPTIONAL_INAT_FIELDS:
                    row.pop(fld, None)
            fieldnames = set(REQUIRED_FIELDS)

        seen_files: set[str] = set()
        deduplicated_rows: list[dict] = []
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
        if skipped:
            logger.info(f"Skipped {skipped} duplicate entries during merge")
        all_rows = deduplicated_rows

    final_fieldnames: list[str] = []
    for fld in REQUIRED_FIELDS:
        if fld in fieldnames:
            final_fieldnames.append(fld)
            fieldnames.discard(fld)
    for fld in OPTIONAL_INAT_FIELDS:
        if fld in fieldnames:
            final_fieldnames.append(fld)
            fieldnames.discard(fld)
    final_fieldnames.extend(sorted(fieldnames))

    normalized_rows = [{fld: row.get(fld, "") for fld in final_fieldnames} for row in all_rows]

    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    return len(normalized_rows)


__all__ = [
    "read_metadata_csv",
    "write_metadata_csv",
    "REQUIRED_FIELDS",
    "OPTIONAL_INAT_FIELDS",
]

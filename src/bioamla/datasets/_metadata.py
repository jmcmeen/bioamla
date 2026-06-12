"""Canonical metadata-CSV schema and helpers (group/folder level).

This is the single source of truth for the ``metadata.csv`` that describes a
folder of audio files — emitted by catalog downloads and by dataset extraction,
and consumed by merge/stats/manifest. The catalog package re-exports these via a
thin shim (``bioamla.catalogs._metadata``).

Field ordering for any written file is: :data:`CORE_FIELDS`, then
:data:`ATTRIBUTION_FIELDS`, then :data:`OPTIONAL_INAT_FIELDS`, then any remaining
columns sorted alphabetically. Source-specific extras (``xc_id``, ``ml_id``,
``quality``, ``rating``, ...) survive in that sorted remainder, so normalizing a
catalog row never loses data.
"""

import csv
import logging
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical columns every metadata.csv should lead with. Present-only: a column
# is written only when at least one row carries it, so a sparse iNat row stays
# compact.
CORE_FIELDS = [
    "file_name",
    "label",
    "target",
    "split",
    "source",
    "scientific_name",
    "common_name",
    "license",
    "attribution",
    "latitude",
    "longitude",
    "date",
    "duration",
]

# Attribution block (iNaturalist-style provenance), ordered after the core.
ATTRIBUTION_FIELDS = [
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

# Back-compat alias for the historical iNat-centric "required" set. Used as the
# header for an empty file and still importable by older callers.
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

# Precedence used when ordering columns in a written file.
_FIELD_ORDER = CORE_FIELDS + ATTRIBUTION_FIELDS + OPTIONAL_INAT_FIELDS

# Per-source remaps from a catalog's idiosyncratic column onto a canonical one.
# Applied only when the canonical column isn't already set, so no value is lost.
_SOURCE_REMAPS: dict[str, dict[str, str]] = {
    "xeno_canto": {"recordist": "attribution", "url": "attr_url"},
    "macaulay": {"contributor": "attribution"},
    "inaturalist": {},
}


def normalize_catalog_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    """Map a catalog-specific metadata row onto the canonical schema.

    Sets ``source``, derives ``label`` from the scientific/common name when
    absent, and renames known source-specific columns (e.g. ``recordist`` ->
    ``attribution``) onto canonical names. Unmapped columns are preserved as-is
    and end up in the sorted remainder when written, so nothing is dropped.

    Args:
        row: The catalog's row dict.
        source: Source identifier (e.g. ``"xeno_canto"``, ``"macaulay"``,
            ``"inaturalist"``).

    Returns:
        A new dict with canonical columns populated.
    """
    out = dict(row)
    out["source"] = source

    for src_key, dst_key in _SOURCE_REMAPS.get(source, {}).items():
        if src_key in out and not out.get(dst_key):
            out[dst_key] = out.pop(src_key)

    if not out.get("label"):
        out["label"] = out.get("scientific_name") or out.get("common_name") or ""

    return out


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
    against existing data by ``file_name``. Columns are ordered :data:`CORE_FIELDS`
    -> :data:`ATTRIBUTION_FIELDS` -> :data:`OPTIONAL_INAT_FIELDS` -> sorted
    remainder.
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
            # Drop only the mismatched optional columns; keep core/extra columns.
            fieldnames = fieldnames - set(OPTIONAL_INAT_FIELDS)

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
    for fld in _FIELD_ORDER:
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


def get_existing_observation_ids(metadata_path: Path) -> set[tuple[int, int]]:
    """Extract ``(observation_id, sound_id)`` pairs from existing metadata.

    Parses filenames of the form ``inat_{obs_id}_sound_{sound_id}.ext`` to skip
    files that have already been downloaded.
    """
    existing: set[tuple[int, int]] = set()

    if not metadata_path.exists():
        return existing

    try:
        rows, _ = read_metadata_csv(metadata_path)
        for row in rows:
            file_name = row.get("file_name", "")
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


__all__ = [
    "read_metadata_csv",
    "write_metadata_csv",
    "normalize_catalog_row",
    "get_existing_observation_ids",
    "CORE_FIELDS",
    "ATTRIBUTION_FIELDS",
    "OPTIONAL_INAT_FIELDS",
    "REQUIRED_FIELDS",
]

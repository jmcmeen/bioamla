"""Merge multiple audio datasets (files + metadata) into a single dataset."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from bioamla.exceptions import InvalidInputError, MergeError, NotFoundError

logger = logging.getLogger(__name__)

# Supported audio formats for conversion during merge
_SUPPORTED_FORMATS = {"wav", "mp3", "m4a", "aac", "flac", "ogg", "wma"}


def _get_converter(source_ext: str, target_format: str):
    """Return a ``<src>_to_<dst>`` converter function, or None if unavailable.

    No named per-format converters are provided in the current layout, so this
    always returns None and callers fall back to a plain file copy.
    """
    return None


def find_species_name(category: str, all_categories: set[str]) -> str:
    """Return the most general matching species name for a (sub)species category.

    If ``category`` is a subspecies (e.g. "Genus species subsp"), return the
    shortest known category that is a strict prefix of it; otherwise return
    ``category`` unchanged.
    """
    if not category:
        return category

    matching = [c for c in all_categories if category.startswith(c) and c != category]
    if matching:
        return min(matching, key=len)
    return category


def merge_datasets(
    dataset_paths: list[str],
    output_dir: str,
    metadata_filename: str = "metadata.csv",
    skip_existing: bool = True,
    organize_by_category: bool = True,
    target_format: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Merge multiple audio datasets into a single output dataset.

    Combines audio files and metadata from several dataset directories into one
    output directory, optionally organizing by category and converting formats.

    Args:
        dataset_paths: Paths to dataset directories to merge.
        output_dir: Output directory for the merged dataset.
        metadata_filename: Name of the metadata CSV in each dataset.
        skip_existing: Skip files that already exist in the output.
        organize_by_category: Organize files into per-category subdirectories.
        target_format: Convert all audio to this format during merge.
        verbose: Print progress information.

    Returns:
        Dict of summary stats: ``datasets_merged``, ``total_files``,
        ``files_copied``, ``files_skipped``, ``files_converted``, ``output_dir``,
        ``metadata_file``.

    Raises:
        InvalidInputError: If no paths given or target_format is unsupported.
        NotFoundError: If a source dataset path does not exist.
        MergeError: If reading/writing metadata fails.
    """
    from bioamla.common.files import sanitize_filename
    from bioamla.datasets._metadata import read_metadata_csv, write_metadata_csv

    if not dataset_paths:
        raise InvalidInputError("At least one dataset path must be provided")

    if target_format:
        target_format = target_format.lower().lstrip(".")
        if target_format not in _SUPPORTED_FORMATS:
            raise InvalidInputError(
                f"Unsupported target format: {target_format}. "
                f"Supported formats: {', '.join(sorted(_SUPPORTED_FORMATS))}"
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "total_files": 0,
        "files_copied": 0,
        "files_skipped": 0,
        "files_converted": 0,
        "datasets_merged": 0,
        "output_dir": str(output_path.absolute()),
        "metadata_file": str(output_path / metadata_filename),
    }

    all_metadata_rows: list[dict] = []
    existing_filenames: set[str] = set()
    all_fieldnames: set[str] = set()
    all_categories: set[str] = set()

    output_metadata_path = output_path / metadata_filename

    try:
        if output_metadata_path.exists():
            existing_rows, existing_fieldnames = read_metadata_csv(output_metadata_path)
            all_metadata_rows.extend(existing_rows)
            all_fieldnames.update(existing_fieldnames)
            existing_filenames.update(row.get("file_name", "") for row in existing_rows)
            for row in existing_rows:
                category = row.get("category", "")
                if category:
                    all_categories.add(category)
            if verbose:
                print(f"Found {len(existing_rows)} existing entries in output metadata.")

        for dataset_path in dataset_paths:
            source_path = Path(dataset_path)

            if not source_path.exists():
                raise NotFoundError(f"Dataset path does not exist: {dataset_path}")

            source_metadata_path = source_path / metadata_filename
            if not source_metadata_path.exists():
                if verbose:
                    print(f"Warning: No {metadata_filename} found in {dataset_path}, skipping.")
                continue

            if verbose:
                print(f"Processing dataset: {dataset_path}")

            source_rows, source_fieldnames = read_metadata_csv(source_metadata_path)
            all_fieldnames.update(source_fieldnames)

            for row in source_rows:
                category = row.get("category", "")
                if category:
                    all_categories.add(category)

            files_copied_from_source = 0
            files_skipped_from_source = 0
            files_converted_from_source = 0

            for row in source_rows:
                source_filename = row.get("file_name", "")
                if not source_filename:
                    continue

                source_file_path = source_path / source_filename

                if organize_by_category:
                    category = row.get("category", "")
                    dir_category = find_species_name(category, all_categories)
                    category_dir = sanitize_filename(dir_category) if dir_category else "unknown"
                    base_filename = Path(source_filename).name
                    dest_filename = f"{category_dir}/{base_filename}"
                else:
                    dest_filename = source_filename

                source_ext = Path(dest_filename).suffix.lower().lstrip(".")
                needs_conversion = bool(target_format) and source_ext != target_format

                if needs_conversion:
                    dest_filename = str(Path(dest_filename).with_suffix(f".{target_format}"))

                dest_file_path = output_path / dest_filename

                if dest_filename in existing_filenames and skip_existing:
                    files_skipped_from_source += 1
                    continue

                dest_file_path.parent.mkdir(parents=True, exist_ok=True)

                if not source_file_path.exists():
                    if verbose:
                        print(f"  Warning: Source file not found: {source_file_path}")
                    continue

                if needs_conversion:
                    converter = _get_converter(source_ext, target_format)
                    if converter:
                        try:
                            converter(str(source_file_path), str(dest_file_path))
                            files_converted_from_source += 1
                            if dest_filename not in existing_filenames:
                                updated_row = row.copy()
                                updated_row["file_name"] = dest_filename
                                updated_row["attr_note"] = "modified clip from original source"
                                all_metadata_rows.append(updated_row)
                                existing_filenames.add(dest_filename)
                            if verbose:
                                print(f"  Converted: {source_filename} -> {dest_filename}")
                        except Exception as e:  # noqa: BLE001 - per-file conversion failure
                            if verbose:
                                print(f"  Error converting {source_filename}: {e}")
                    else:
                        if verbose:
                            print(
                                f"  Warning: No converter for {source_ext} -> "
                                f"{target_format}, copying: {source_filename}"
                            )
                        shutil.copy2(source_file_path, dest_file_path)
                        files_copied_from_source += 1
                        if dest_filename not in existing_filenames:
                            updated_row = row.copy()
                            updated_row["file_name"] = dest_filename
                            all_metadata_rows.append(updated_row)
                            existing_filenames.add(dest_filename)
                else:
                    shutil.copy2(source_file_path, dest_file_path)
                    files_copied_from_source += 1
                    if dest_filename not in existing_filenames:
                        updated_row = row.copy()
                        updated_row["file_name"] = dest_filename
                        all_metadata_rows.append(updated_row)
                        existing_filenames.add(dest_filename)

            stats["files_copied"] += files_copied_from_source
            stats["files_skipped"] += files_skipped_from_source
            stats["files_converted"] += files_converted_from_source
            stats["datasets_merged"] += 1

            if verbose:
                msg = f"  Copied: {files_copied_from_source}, Skipped: {files_skipped_from_source}"
                if target_format:
                    msg += f", Converted: {files_converted_from_source}"
                print(msg)

        if all_metadata_rows:
            write_metadata_csv(
                output_metadata_path,
                all_metadata_rows,
                all_fieldnames,
                merge_existing=False,
            )
    except (NotFoundError, InvalidInputError):
        raise
    except OSError as e:
        raise MergeError(f"Dataset merge failed: {e}") from e

    stats["total_files"] = len(all_metadata_rows)

    if verbose:
        print("\nMerge complete!")
        print(f"  Datasets merged: {stats['datasets_merged']}")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Files copied: {stats['files_copied']}")
        if target_format:
            print(f"  Files converted: {stats['files_converted']}")
        print(f"  Files skipped: {stats['files_skipped']}")
        print(f"  Output directory: {stats['output_dir']}")
        print(f"  Metadata file: {stats['metadata_file']}")

    return stats

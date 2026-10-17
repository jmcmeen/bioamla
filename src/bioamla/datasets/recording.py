"""Recorder metadata for dataset rows.

Field recorders stamp each WAV with when and where it was made and by which unit
(GUANO, AudioMoth, Wildlife Acoustics WAMD, ``bext``). :func:`add_recording_metadata`
copies those details into a dataset's ``metadata.csv`` as the columns in
:data:`bioamla.audio.RECORDING_METADATA_FIELDS`, so a raw deployment can be
described from the files themselves instead of a hand-written README, and
:func:`bioamla.datasets.build_manifest_from_metadata` can summarize the span,
recorders and sites.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bioamla.datasets._metadata import read_metadata_csv, write_metadata_csv
from bioamla.exceptions import AudioLoadError, NotFoundError

logger = logging.getLogger(__name__)


def add_recording_metadata(
    dataset_dir: str,
    metadata_filename: str = "metadata.csv",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fill the recorder-metadata columns of a dataset's ``metadata.csv`` in place.

    Each row's ``file_name`` is resolved relative to ``dataset_dir`` (absolute
    paths are used as-is). WAV files are read with
    :func:`bioamla.audio.get_recording_metadata`; rows for other formats, missing
    files or unreadable WAVs are left untouched and counted as skipped.

    Args:
        dataset_dir: Dataset directory containing the metadata CSV.
        metadata_filename: Name of the metadata CSV.
        overwrite: Re-read files whose row already has ``metadata_sources`` set.
            By default those rows are kept as they are.

    Returns:
        Counts: ``rows``, ``updated``, ``with_metadata`` (updated rows whose file
        carried at least one standard), ``skipped``, and ``errors`` (list of
        ``"file: reason"`` strings for unreadable WAVs).

    Raises:
        NotFoundError: If the metadata CSV is missing.
    """
    from bioamla.audio._riffy import is_wav_path
    from bioamla.audio.metadata import RECORDING_METADATA_FIELDS, get_recording_metadata

    base = Path(dataset_dir)
    metadata_path = base / metadata_filename
    if not metadata_path.exists():
        raise NotFoundError(f"Metadata file not found: {metadata_path}")

    rows, fieldnames = read_metadata_csv(metadata_path)
    counts: dict[str, Any] = {
        "rows": len(rows),
        "updated": 0,
        "with_metadata": 0,
        "skipped": 0,
        "errors": [],
    }

    for row in rows:
        file_name = row.get("file_name") or ""
        path = Path(file_name)
        if not path.is_absolute():
            path = base / path
        if not file_name or not is_wav_path(path) or not path.exists():
            counts["skipped"] += 1
            continue
        if not overwrite and (row.get("metadata_sources") or "").strip():
            counts["skipped"] += 1
            continue
        try:
            meta = get_recording_metadata(str(path))
        except AudioLoadError as e:
            counts["errors"].append(f"{file_name}: {e}")
            counts["skipped"] += 1
            continue
        row.update(meta.to_dict())
        counts["updated"] += 1
        if not meta.is_empty:
            counts["with_metadata"] += 1

    if counts["updated"]:
        fieldnames = set(fieldnames) | set(RECORDING_METADATA_FIELDS)
        write_metadata_csv(metadata_path, rows, fieldnames=fieldnames, merge_existing=False)
        logger.info(
            f"Recorder metadata: {counts['updated']} rows updated, "
            f"{counts['with_metadata']} with metadata, {counts['skipped']} skipped"
        )
    return counts


__all__ = ["add_recording_metadata"]

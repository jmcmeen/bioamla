"""Shared annotation-format dispatch (raven / csv / bioamla).

A single place to resolve an annotation file's format from its extension and to
load/save through the right reader/writer. Used by both the ``annotation`` CLI
group and the dataset-extraction routines so format handling stays consistent.
"""

from pathlib import Path
from typing import Any

from bioamla.datasets.annotations import (
    Annotation,
    load_bioamla_annotations,
    load_csv_annotations,
    load_raven_selection_table,
    save_bioamla_annotations,
    save_csv_annotations,
    save_raven_selection_table,
)

# Supported annotation file formats.
ANNOTATION_FORMATS = ["raven", "csv", "bioamla"]


def detect_annotation_format(path: Path, explicit: str | None = None) -> str:
    """Resolve an annotation format from an explicit flag or the file extension.

    ``.txt`` -> raven selection table, ``.json`` -> bioamla format, anything
    else -> flat CSV.
    """
    if explicit:
        return explicit
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return "raven"
    if suffix == ".json":
        return "bioamla"
    return "csv"


def load_annotations(
    path: Path, fmt: str, label_column: str | None = None
) -> tuple[list[Annotation], dict[str, Any]]:
    """Load annotations in the given format, returning ``(annotations, metadata)``.

    Only the bioamla format carries file-level metadata; the others return an
    empty metadata dict so callers can treat every format uniformly.
    """
    if fmt == "raven":
        return load_raven_selection_table(str(path), label_column=label_column), {}
    if fmt == "bioamla":
        return load_bioamla_annotations(str(path))
    return load_csv_annotations(str(path)), {}


def save_annotations(
    annotations: list[Annotation], path: Path, fmt: str, metadata: dict | None = None
) -> None:
    """Save annotations in the given format, preserving metadata for bioamla."""
    if fmt == "raven":
        save_raven_selection_table(annotations, str(path))
    elif fmt == "bioamla":
        save_bioamla_annotations(annotations, str(path), metadata=metadata)
    else:
        save_csv_annotations(annotations, str(path))

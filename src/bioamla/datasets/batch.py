"""Batch operations for the datasets domain.

Thin wrappers around :func:`bioamla.batch.run_batch` so the batch CLI can be
wired to the datasets domain at cut-over time. Functions raise
:class:`~bioamla.exceptions.BioamlaError` subclasses on bad input.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from bioamla.batch import BatchResult, discover_files, run_batch
from bioamla.datasets.annotations import (
    load_csv_annotations,
    load_raven_selection_table,
    save_csv_annotations,
    save_raven_selection_table,
)
from bioamla.exceptions import InvalidInputError, NotFoundError

logger = logging.getLogger(__name__)

_VALID_FORMATS = {"raven", "csv"}


def _format_for_path(path: Path) -> str:
    """Infer annotation format from a path's extension (.txt -> raven, else csv)."""
    return "raven" if path.suffix.lower() == ".txt" else "csv"


def batch_convert_annotations(
    input_dir: str,
    output_dir: str,
    to_format: str,
    from_format: str | None = None,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """Convert every annotation file in a directory to another format.

    Args:
        input_dir: Directory containing annotation files.
        output_dir: Directory for converted output (created per-file).
        to_format: Target format ('raven' or 'csv').
        from_format: Source format ('raven' or 'csv'); auto-detected if None.
        recursive: Recurse into subdirectories.
        max_workers: Worker processes for parallel conversion.
        continue_on_error: Keep going past per-file failures.
        on_progress: Optional ``(completed, total)`` progress callback.

    Returns:
        A :class:`BatchResult` summarizing the run.

    Raises:
        InvalidInputError: If a format is invalid.
        NotFoundError: If the input directory doesn't exist.
    """
    if to_format not in _VALID_FORMATS:
        raise InvalidInputError(f"Invalid to_format: {to_format} (expected 'raven' or 'csv')")
    if from_format is not None and from_format not in _VALID_FORMATS:
        raise InvalidInputError(f"Invalid from_format: {from_format} (expected 'raven' or 'csv')")

    in_path = Path(input_dir)
    if not in_path.exists():
        raise NotFoundError(f"Input directory not found: {input_dir}")

    out_path = Path(output_dir)
    out_ext = ".txt" if to_format == "raven" else ".csv"

    extensions = {".txt", ".csv"}
    files: list[Path] = discover_files(
        in_path, recursive=recursive, file_filter=lambda p: p.suffix.lower() in extensions
    )

    def _convert(src: Path) -> str:
        src_format = from_format or _format_for_path(src)
        if src_format == "raven":
            annotations = load_raven_selection_table(str(src))
        else:
            annotations = load_csv_annotations(str(src))

        try:
            rel = src.relative_to(in_path)
        except ValueError:
            rel = Path(src.name)
        dest = out_path / rel.with_suffix(out_ext)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if to_format == "raven":
            return save_raven_selection_table(annotations, str(dest))
        return save_csv_annotations(annotations, str(dest))

    return run_batch(
        files,
        _convert,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
    )

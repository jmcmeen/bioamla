"""
Indices Batch Processing
========================

Directory-level wrapper for acoustic indices built on
:func:`bioamla.batch.run_batch`.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from bioamla.audio.discovery import list_audio_files
from bioamla.batch import BatchResult, run_batch, write_rows_csv
from bioamla.exceptions import NotFoundError
from bioamla.indices.compute import _failed_indices_row, _indices_row, check_index_names


def batch_index_files(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    indices: list[str] | None = None,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    **kwargs: Any,
) -> BatchResult:
    """
    Compute acoustic indices for every audio file in a directory.

    Args:
        input_dir: Directory containing audio files.
        output_dir: If given, ``indices.csv`` is written here.
        indices: Indices to keep; None keeps everything computed (see
            :func:`bioamla.indices.indices_row`).
        recursive: Search subdirectories.
        max_workers: Number of worker threads (1 = sequential).
        continue_on_error: Collect per-file errors and keep going if True.
        on_progress: Optional ``(completed, total)`` progress callback.
        **kwargs: Passed to :func:`bioamla.indices.compute_indices_from_file`.

    Returns:
        A :class:`bioamla.batch.BatchResult`. Its ``metadata["rows"]`` holds one
        row per file in discovery order: ``filepath``, the index values and
        ``success``, or ``filepath``, ``success`` (False) and ``error`` for a
        file that failed. ``output_files`` holds the CSV path when one was
        written.

    Raises:
        NotFoundError: If the input directory does not exist.
        InvalidInputError: If ``indices`` names an unknown index.
    """
    if indices is not None:
        check_index_names(indices)
    in_dir = Path(input_dir)
    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")

    files = [str(f) for f in list_audio_files(str(in_dir), recursive=recursive)]
    by_path: dict[str, dict[str, Any]] = {}

    def _index(path: str) -> None:
        try:
            by_path[path] = _indices_row(path, indices, **kwargs)
        except Exception as e:
            by_path[path] = _failed_indices_row(path, e)
            raise

    result = run_batch(
        files,
        _index,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
    )
    rows = [by_path[f] for f in files if f in by_path]
    result.metadata["rows"] = rows
    if output_dir is not None and rows:
        result.output_files = [
            str(write_rows_csv(Path(output_dir) / "indices.csv", rows, _columns(rows)))
        ]
    return result


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    """``filepath``, then index columns in first-seen order, then ``success``/``error``."""
    bookkeeping = ("filepath", "success", "error")
    values: list[str] = []
    for row in rows:
        values.extend(k for k in row if k not in bookkeeping and k not in values)
    tail = ["success", "error"] if any("error" in row for row in rows) else ["success"]
    return ["filepath", *values, *tail]


__all__ = ["batch_index_files"]

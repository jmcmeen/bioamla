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
from bioamla.indices.compute import compute_indices_from_file


def batch_index_files(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
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
    """
    in_dir = Path(input_dir)
    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")

    files = [str(f) for f in list_audio_files(str(in_dir), recursive=recursive)]
    by_path: dict[str, dict[str, Any]] = {}

    def _index(path: str) -> None:
        try:
            indices = compute_indices_from_file(path, **kwargs)
        except Exception as e:
            by_path[path] = {"filepath": path, "success": False, "error": str(e)}
            raise
        # filepath first, success last, for a readable CSV column order
        row: dict[str, Any] = {"filepath": path}
        row.update(indices.to_dict())
        row["success"] = True
        by_path[path] = row

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
        result.output_files = [str(write_rows_csv(Path(output_dir) / "indices.csv", rows))]
    return result


__all__ = ["batch_index_files"]

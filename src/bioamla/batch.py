"""
Generic Batch Engine
====================

Module-level batch machinery ported from the old ``BatchServiceBase`` and
``BatchCSVHandler``. Plain functions + dataclasses, direct file I/O via
:mod:`pathlib`, and raising on error.

Public surface:
- :func:`run_batch` — sequential or ProcessPool-parallel item processing.
  Per-item results are collected from ``future.result()`` return values so
  parallel mode does not lose data.
- :func:`run_csv_batch` — drive a per-row processor over a loaded CSV context
  (the engine half of the old ``process_batch_csv``).
- :func:`discover_files` — glob-based file discovery.
- CSV helpers: :func:`load_csv`, :func:`write_csv`, :func:`resolve_file_path`,
  :func:`resolve_output_path`, :func:`update_row_path`,
  :func:`merge_analysis_results`, :func:`expand_row_for_segments`.
- Types: :class:`BatchConfig`, :class:`BatchResult`, :class:`SegmentInfo`,
  :class:`MetadataRow`, and :class:`CSVBatchContext`.
"""

import csv
import io
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from bioamla.exceptions import InvalidInputError, NotFoundError

T = TypeVar("T")
I = TypeVar("I")  # noqa: E741 - item type


# =============================================================================
# Batch configuration and result types
# =============================================================================


@dataclass
class BatchConfig:
    """Configuration for batch operations.

    Supports two input modes (mutually exclusive):
    - Directory mode: provide ``input_dir`` to process all files in a directory.
    - CSV metadata mode: provide ``input_file`` pointing to a CSV with a
      ``file_name`` column.

    For programmatic usage, validation can be bypassed by setting
    ``_skip_validation=True`` (useful for testing or advanced use cases).
    """

    input_dir: str | None = None
    input_file: str | None = None
    output_dir: str = ""
    recursive: bool = True
    max_workers: int = 1
    continue_on_error: bool = True
    quiet: bool = False
    output_template: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    _skip_validation: bool = False  # Internal flag for testing/advanced usage

    def __post_init__(self) -> None:
        """Validate mutual exclusivity of ``input_dir`` and ``input_file``."""
        if self._skip_validation:
            return

        if self.input_dir is None and self.input_file is None:
            raise ValueError(
                "Either input_dir or input_file must be specified. "
                "For testing or advanced usage, set _skip_validation=True to bypass this check."
            )
        if self.input_dir is not None and self.input_file is not None:
            raise ValueError(
                "input_dir and input_file are mutually exclusive. "
                "For testing or advanced usage, set _skip_validation=True to bypass this check."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)


@dataclass
class SegmentInfo:
    """Information about a created audio segment."""

    segment_path: Path
    segment_id: int
    start_time: float
    end_time: float
    duration: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary (paths stringified)."""
        data = asdict(self)
        data["segment_path"] = str(self.segment_path)
        return data


@dataclass
class BatchResult:
    """Generic result of batch processing."""

    total_files: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    output_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)


# =============================================================================
# File discovery
# =============================================================================


def discover_files(
    input_dir: str | Path,
    *,
    recursive: bool = True,
    file_filter: Callable[[Path], bool] | None = None,
) -> list[Path]:
    """
    Discover files under a directory using stdlib globbing.

    Args:
        input_dir: Directory to search.
        recursive: If True, recurse into subdirectories.
        file_filter: Optional predicate applied to each file path.

    Returns:
        Sorted list of matching file paths (empty if the directory is missing).
    """
    base = Path(input_dir)
    if not base.exists():
        return []

    glob = base.rglob("*") if recursive else base.glob("*")
    files = [p for p in glob if p.is_file()]
    if file_filter:
        files = [p for p in files if file_filter(p)]
    return sorted(files)


# =============================================================================
# Generic batch runner
# =============================================================================


def run_batch(
    items: list[I],
    process_fn: Callable[[I], T],
    *,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """
    Run ``process_fn`` over ``items`` sequentially or in parallel.

    In parallel mode (``max_workers > 1``) a :class:`ProcessPoolExecutor` is
    used and per-item return values are collected via ``future.result()`` so no
    data is lost. Successful return values are appended to
    ``BatchResult.output_files`` as strings when they are not None.

    Args:
        items: Items to process.
        process_fn: Callable applied to each item. Must be picklable for
            parallel execution.
        max_workers: Number of worker processes; 1 runs sequentially.
        continue_on_error: If True, collect errors and keep going; if False,
            re-raise the first exception encountered.
        on_progress: Optional callback ``(completed, total)`` invoked after each
            item completes.

    Returns:
        A :class:`BatchResult` summarizing the run.
    """
    start_time = datetime.now()
    result = BatchResult(start_time=start_time.isoformat())
    result.total_files = len(items)

    def _record_success(value: Any) -> None:
        result.successful += 1
        if value is not None:
            result.output_files.append(str(value))

    def _record_failure(item: Any, exc: Exception) -> None:
        result.failed += 1
        result.errors.append(f"{item}: {exc}")

    completed = 0
    total = len(items)

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_fn, item): item for item in items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    _record_success(future.result())
                except Exception as e:
                    _record_failure(item, e)
                    if not continue_on_error:
                        raise
                completed += 1
                if on_progress:
                    on_progress(completed, total)
        sys.stdout.flush()
        sys.stderr.flush()
    else:
        for item in items:
            try:
                _record_success(process_fn(item))
            except Exception as e:
                _record_failure(item, e)
                if not continue_on_error:
                    raise
            completed += 1
            if on_progress:
                on_progress(completed, total)

    end_time = datetime.now()
    result.end_time = end_time.isoformat()
    result.duration_seconds = (end_time - start_time).total_seconds()
    return result


# =============================================================================
# CSV metadata handling
# =============================================================================


@dataclass
class MetadataRow:
    """Single row from a metadata CSV with file path and arbitrary fields."""

    file_name: str  # Original relative path from CSV
    file_path: Path  # Resolved absolute path for processing
    metadata_fields: dict[str, Any] = field(default_factory=dict)  # All other CSV columns
    output_path: Path | None = None  # Updated path after processing


@dataclass
class CSVBatchContext:
    """Context for CSV-based batch processing."""

    csv_path: Path  # Input CSV location
    csv_dir: Path  # CSV directory (base for relative paths)
    output_dir: Path | None  # Output directory if specified
    rows: list[MetadataRow] = field(default_factory=list)  # All CSV rows
    fieldnames: list[str] = field(default_factory=list)  # CSV column names (preserved order)
    new_fieldnames: list[str] = field(default_factory=list)  # New columns added during processing


def resolve_file_path(file_name: str, csv_dir: Path) -> Path:
    """
    Resolve a file path relative to the CSV directory.

    Args:
        file_name: Relative or absolute path from the CSV.
        csv_dir: Directory containing the CSV file.

    Returns:
        Resolved absolute path.
    """
    file_path = Path(file_name)
    if file_path.is_absolute():
        return file_path
    return (csv_dir / file_path).resolve()


def load_csv(csv_path: str | Path, output_dir: str | None = None) -> CSVBatchContext:
    """
    Load a metadata CSV and resolve file paths relative to the CSV directory.

    Args:
        csv_path: Path to the metadata CSV file.
        output_dir: Optional output directory for processed files.

    Returns:
        A :class:`CSVBatchContext` with all rows and resolved paths.

    Raises:
        NotFoundError: If the CSV file does not exist.
        InvalidInputError: If the CSV lacks a ``file_name`` column.
    """
    csv_path_obj = Path(csv_path).resolve()
    if not csv_path_obj.exists():
        raise NotFoundError(f"CSV file not found: {csv_path}")

    csv_dir = csv_path_obj.parent
    output_dir_obj = Path(output_dir).resolve() if output_dir else None

    csv_content = csv_path_obj.read_text(encoding="utf-8")
    reader = csv.DictReader(csv_content.splitlines())

    fieldnames = reader.fieldnames or []
    if "file_name" not in fieldnames:
        raise InvalidInputError(f"CSV must have 'file_name' column. Found: {fieldnames}")

    rows: list[MetadataRow] = []
    for row_dict in reader:
        file_name = row_dict["file_name"]
        file_path = resolve_file_path(file_name, csv_dir)
        metadata_fields = {k: v for k, v in row_dict.items() if k != "file_name"}
        rows.append(
            MetadataRow(file_name=file_name, file_path=file_path, metadata_fields=metadata_fields)
        )

    return CSVBatchContext(
        csv_path=csv_path_obj,
        csv_dir=csv_dir,
        output_dir=output_dir_obj,
        rows=rows,
        fieldnames=list(fieldnames),
    )


def resolve_output_path(
    input_path: Path,
    csv_context: CSVBatchContext,
    new_extension: str | None = None,
) -> Path:
    """
    Calculate the output path for a processed file.

    WITH output_dir: ``output_dir / relative_structure / filename``.
    WITHOUT output_dir: same location as input (in-place).

    Args:
        input_path: Original input file path.
        csv_context: CSV batch context with output directory info.
        new_extension: New file extension (e.g. ``.wav``) if the format changes.

    Returns:
        Output file path.
    """
    if new_extension:
        output_stem = input_path.stem
        output_ext = new_extension if new_extension.startswith(".") else f".{new_extension}"
        output_name = f"{output_stem}{output_ext}"
    else:
        output_name = input_path.name

    if csv_context.output_dir:
        try:
            rel_to_csv = input_path.relative_to(csv_context.csv_dir)
            output_path = csv_context.output_dir / rel_to_csv.parent / output_name
        except ValueError:
            output_path = csv_context.output_dir / output_name
    else:
        output_path = input_path.parent / output_name

    return output_path


def update_row_path(row: MetadataRow, new_path: Path, csv_context: CSVBatchContext) -> None:
    """
    Update a row's ``file_name`` to ``new_path`` (relative to the CSV if possible).

    Args:
        row: Metadata row to update.
        new_path: New absolute path after processing.
        csv_context: CSV batch context.
    """
    row.output_path = new_path

    if csv_context.output_dir:
        try:
            row.file_name = str(new_path.relative_to(csv_context.output_dir))
        except ValueError:
            row.file_name = str(new_path)
    else:
        try:
            row.file_name = str(new_path.relative_to(csv_context.csv_dir))
        except ValueError:
            row.file_name = str(new_path)


def merge_analysis_results(row: MetadataRow, results: dict[str, Any]) -> None:
    """
    Merge analysis results into a row's metadata fields.

    Args:
        row: Metadata row to update.
        results: Result columns to add (e.g. ``{'aci': 0.85, 'adi': 0.72}``).
    """
    row.metadata_fields.update(results)


def expand_row_for_segments(
    parent_row: MetadataRow,
    segments: list[Any],
    csv_context: CSVBatchContext,
) -> list[MetadataRow]:
    """
    Create multiple output rows from one input row (for the segment operation).

    Args:
        parent_row: Original input row with parent file metadata.
        segments: List of :class:`SegmentInfo`-like objects.
        csv_context: CSV batch context.

    Returns:
        List of new :class:`MetadataRow` objects (one per segment).
    """
    new_rows: list[MetadataRow] = []

    for seg_info in segments:
        try:
            if csv_context.output_dir:
                rel_path = seg_info.segment_path.relative_to(csv_context.output_dir)
            else:
                rel_path = seg_info.segment_path.relative_to(csv_context.csv_dir)
            file_name = str(rel_path)
        except ValueError:
            file_name = str(seg_info.segment_path)

        segment_metadata = parent_row.metadata_fields.copy()
        segment_metadata.update(
            {
                "parent_file": parent_row.file_name,
                "segment_id": seg_info.segment_id,
                "start_time": seg_info.start_time,
                "end_time": seg_info.end_time,
                "duration": seg_info.duration,
            }
        )

        new_rows.append(
            MetadataRow(
                file_name=file_name,
                file_path=seg_info.segment_path,
                metadata_fields=segment_metadata,
                output_path=seg_info.segment_path,
            )
        )

    return new_rows


def run_csv_batch(
    context: CSVBatchContext,
    process_row: Callable[["MetadataRow"], Any],
    *,
    max_workers: int = 1,
    continue_on_error: bool = True,
    quiet: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """Run ``process_row`` over every row of a loaded CSV context.

    Existence of each row's resolved ``file_path`` is checked up-front (a missing
    file is recorded as a failure, matching the old ``process_batch_csv``
    behaviour). ``process_row`` is invoked with the :class:`MetadataRow` and may
    mutate it in place (e.g. merge result columns via
    :func:`merge_analysis_results`, update its path via :func:`update_row_path`,
    or stash segment info for later expansion). Its non-None return value is
    appended to :attr:`BatchResult.output_files`.

    The caller is responsible for calling :func:`write_csv` (and any row
    expansion) afterwards so it controls the final CSV shape.

    Args:
        context: Loaded :class:`CSVBatchContext`.
        process_row: Callable applied to each existing row. Runs in-process
            (sequentially) so it may close over and mutate ``context``.
        max_workers: Reserved for parity; CSV rows are processed sequentially
            because ``process_row`` typically mutates shared state.
        continue_on_error: Collect per-row errors and keep going if True.
        quiet: Suppress per-error stderr prints.
        on_progress: Optional ``(completed, total)`` progress callback.

    Returns:
        A :class:`BatchResult` summarizing the run.
    """
    start_time = datetime.now()
    result = BatchResult(start_time=start_time.isoformat())

    rows = context.rows
    result.total_files = len(rows)

    completed = 0
    total = len(rows)
    for row in rows:
        try:
            if not row.file_path.exists():
                raise NotFoundError(f"File not found: {row.file_path}")
            value = process_row(row)
            result.successful += 1
            if value is not None:
                result.output_files.append(str(value))
        except Exception as e:
            result.failed += 1
            result.errors.append(f"{row.file_path}: {e}")
            if not continue_on_error:
                raise
            if not quiet:
                print(f"Error processing {row.file_path}: {e}", file=sys.stderr)
        completed += 1
        if on_progress:
            on_progress(completed, total)

    end_time = datetime.now()
    result.end_time = end_time.isoformat()
    result.duration_seconds = (end_time - start_time).total_seconds()
    return result


def write_csv(context: CSVBatchContext) -> Path:
    """
    Write the updated metadata CSV to its output location.

    Preserves all original columns, adds new columns from analysis results
    (column union), updates ``file_name`` paths, and writes to ``output_dir``
    if specified else in-place.

    Args:
        context: CSV batch context with all rows.

    Returns:
        Path to the written CSV file.
    """
    if context.output_dir:
        output_csv_path = context.output_dir / context.csv_path.name
        context.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_csv_path = context.csv_path

    # Collect all fieldnames (original + new from analysis results), file_name first.
    all_fieldnames = ["file_name"]
    all_fieldnames.extend([f for f in context.fieldnames if f != "file_name"])

    new_fields = set()
    for row in context.rows:
        for key in row.metadata_fields.keys():
            if key not in all_fieldnames:
                new_fields.add(key)

    all_fieldnames.extend(sorted(new_fields))

    rows_data = []
    for row in context.rows:
        row_dict = {"file_name": row.file_name}
        row_dict.update(row.metadata_fields)
        rows_data.append(row_dict)

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=all_fieldnames)
    writer.writeheader()
    writer.writerows(rows_data)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.write_text(csv_buffer.getvalue(), encoding="utf-8")

    return output_csv_path


__all__ = [
    # Types
    "BatchConfig",
    "BatchResult",
    "SegmentInfo",
    "MetadataRow",
    "CSVBatchContext",
    # Discovery / runner
    "discover_files",
    "run_batch",
    "run_csv_batch",
    # CSV helpers
    "load_csv",
    "write_csv",
    "resolve_file_path",
    "resolve_output_path",
    "update_row_path",
    "merge_analysis_results",
    "expand_row_for_segments",
]

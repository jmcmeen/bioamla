"""
Generic Batch Engine
====================

Module-level batch machinery ported from the old ``BatchServiceBase`` and
``BatchCSVHandler``. Plain functions + dataclasses, direct file I/O via
:mod:`pathlib`, and raising on error.

Public surface:
- :func:`run_batch` — sequential or thread-parallel item processing.
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
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from bioamla.common.config import config_dataclass
from bioamla.common.log import log_fields
from bioamla.exceptions import InvalidInputError, NotFoundError
from bioamla.operations import PathStr
from bioamla.protocols import OnProgress

logger = logging.getLogger(__name__)

T = TypeVar("T")
I = TypeVar("I")  # noqa: E741 - item type


# =============================================================================
# Batch configuration and result types
# =============================================================================


@config_dataclass
class BatchConfig:
    """Configuration for batch operations.

    Supports two input modes (mutually exclusive):
    - Directory mode: provide ``input_dir`` to process all files in a directory.
    - CSV metadata mode: provide ``input_file`` pointing to a CSV with a
      ``file_name`` column.

    For programmatic usage, validation can be bypassed by setting
    ``_skip_validation=True`` (useful for testing or advanced use cases).
    """

    input_dir: str | Path | None = None
    input_file: str | Path | None = None
    output_dir: str | Path = ""
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
            raise InvalidInputError(
                "Either input_dir or input_file must be specified. "
                "For testing or advanced usage, set _skip_validation=True to bypass this check."
            )
        if self.input_dir is not None and self.input_file is not None:
            raise InvalidInputError(
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


def write_rows_csv(
    path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> Path:
    """
    Write dict rows to a CSV.

    Args:
        path: Destination CSV path (parent directories are created).
        rows: Rows to write; they may have different keys.
        fieldnames: Column order. Defaults to every key, in first-seen order.

    Returns:
        The written path.
    """
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def output_stems(files: list[str], root: str | Path) -> dict[str, Path]:
    """
    Map each input file to its output stem: its path under ``root`` minus the suffix.

    Inputs that differ only in extension (``a.wav`` and ``a.mp3``) would write
    to the same output, so those keep the extension in the stem (``a_wav``,
    ``a_mp3``). Files outside ``root`` use just their name.

    Args:
        files: Input file paths.
        root: Directory the output tree mirrors.

    Returns:
        Dict mapping each input path to a relative stem with no suffix. Build an
        output path with ``stem.parent / f"{stem.name}.wav"``.
    """
    root = Path(root)
    rel: dict[str, Path] = {}
    for f in files:
        path = Path(f)
        try:
            rel[f] = path.relative_to(root)
        except ValueError:
            rel[f] = Path(path.name)

    # Compared case-insensitively: A.wav and a.mp3 would write to the same file on
    # a case-insensitive filesystem (the macOS and Windows defaults).
    def key(stem: Path) -> str:
        return str(stem).casefold()

    plain = {f: r.parent / r.stem for f, r in rel.items()}
    counts: dict[str, int] = {}
    for stem in plain.values():
        counts[key(stem)] = counts.get(key(stem), 0) + 1

    # Unique stems keep their plain name; colliding ones take their extension,
    # plus a counter if that name is taken too (a.wav + a.mp3 + a_wav.flac).
    stems: dict[str, Path] = {}
    taken = {key(stem) for stem in plain.values() if counts[key(stem)] == 1}
    for f, r in rel.items():
        if counts[key(plain[f])] == 1:
            stems[f] = plain[f]
            continue
        candidate = r.parent / f"{r.stem}_{r.suffix.lstrip('.')}"
        n = 2
        while key(candidate) in taken:
            candidate = r.parent / f"{r.stem}_{r.suffix.lstrip('.')}_{n}"
            n += 1
        taken.add(key(candidate))
        stems[f] = candidate
    return stems


# =============================================================================
# Generic batch runner
# =============================================================================


def _item_file(item: Any) -> str | None:
    """The ``file`` log field for a batch item: its path, if it is one."""
    return str(item) if isinstance(item, (str, Path)) else None


def _short_error(exc: Exception) -> str:
    """The ``error`` log field: the exception's first line, capped at 300 characters.

    The full text stays in the message; the field is for display and grouping, and
    some backends (ffmpeg) put pages of output in their errors.
    """
    text = str(exc).strip().split("\n", 1)[0]
    return text if len(text) <= 300 else text[:297] + "..."


def _elapsed(start: datetime) -> float:
    return round((datetime.now() - start).total_seconds(), 3)


def _log_summary(operation: str, result: BatchResult) -> None:
    """Log a run's last record: counts, duration and status."""
    logger.debug(
        "%s: %d of %d succeeded, %d failed",
        operation,
        result.successful,
        result.total_files,
        result.failed,
        **log_fields(
            operation=operation,
            total=result.total_files,
            elapsed_s=round(result.duration_seconds, 3),
            status="finished" if result.failed == 0 else "failed",
        ),
    )


def run_batch(
    items: list[I],
    process_fn: Callable[[I], T],
    *,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
    operation: str = "batch",
) -> BatchResult:
    """
    Run ``process_fn`` over ``items`` sequentially or in parallel.

    In parallel mode (``max_workers > 1``) a :class:`ThreadPoolExecutor` is used
    and per-item return values are collected via ``future.result()`` so no data
    is lost. Successful return values are appended to ``BatchResult.output_files``
    as strings when they are not None. Threads (not processes) are used so that
    closure ``process_fn``s work and there is no fork/pickling overhead — the
    per-item audio/ML work runs in GIL-releasing native code (soundfile, librosa,
    numpy, torch), so it parallelizes well.

    Args:
        items: Items to process.
        process_fn: Callable applied to each item.
        max_workers: Number of worker threads; 1 runs sequentially.
        continue_on_error: If True, collect errors (each logged as a warning) and
            keep going; if False, re-raise the first exception encountered.
        on_progress: Optional callback ``(completed, total)`` invoked after each
            item completes.
        operation: Name put in the ``operation`` field of the run's log records
            (see :data:`bioamla.common.log.LOG_FIELDS`).

    Returns:
        A :class:`BatchResult` summarizing the run.
    """
    start_time = datetime.now()
    result = BatchResult(start_time=start_time.isoformat())
    result.total_files = len(items)

    completed = 0
    total = len(items)

    def _fields(item: Any, **extra: Any) -> dict[str, dict[str, Any]]:
        return log_fields(
            operation=operation,
            file=_item_file(item),
            index=completed + 1,
            total=total,
            elapsed_s=_elapsed(start_time),
            **extra,
        )

    def _record_success(item: Any, value: Any) -> None:
        result.successful += 1
        if value is not None:
            result.output_files.append(str(value))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s: processed %s", operation, item, **_fields(item))

    def _record_failure(item: Any, exc: Exception) -> None:
        result.failed += 1
        result.errors.append(f"{item}: {exc}")

    if max_workers > 1:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = {executor.submit(process_fn, item): item for item in items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    _record_success(item, future.result())
                except Exception as e:
                    _record_failure(item, e)
                    if not continue_on_error:
                        raise
                    logger.warning(
                        "Error processing %s: %s", item, e, **_fields(item, error=_short_error(e))
                    )
                completed += 1
                if on_progress:
                    on_progress(completed, total)
        except BaseException:
            # Stopping early (a re-raised error, or on_progress raising to
            # cancel): drop the queued items instead of running them all first.
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        executor.shutdown(wait=True)
    else:
        for item in items:
            try:
                _record_success(item, process_fn(item))
            except Exception as e:
                _record_failure(item, e)
                if not continue_on_error:
                    raise
                logger.warning(
                    "Error processing %s: %s", item, e, **_fields(item, error=_short_error(e))
                )
            completed += 1
            if on_progress:
                on_progress(completed, total)

    end_time = datetime.now()
    result.end_time = end_time.isoformat()
    result.duration_seconds = (end_time - start_time).total_seconds()
    _log_summary(operation, result)
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


#: Columns that hold a path relative to the metadata CSV (besides ``file_name``),
#: re-rooted like it when the CSV is written somewhere new.
CSV_PATH_COLUMNS = ("features_file", "embedding_file", "parent_file")


def _csv_relative(path: str | Path, base: Path) -> str:
    """``path`` relative to ``base``, or absolute when no relative path reaches it.

    Paths are made absolute without resolving symlinks, so a CSV that reaches its
    audio through a linked folder keeps doing so.
    """
    target = os.path.abspath(path)
    try:
        return os.path.relpath(target, os.path.abspath(base))
    except ValueError:  # another drive on Windows
        return target


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
    on_progress: OnProgress | None = None,
    operation: str = "batch",
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
        continue_on_error: Collect per-row errors (each logged as a warning) and
            keep going if True.
        on_progress: Optional ``(completed, total)`` progress callback.
        operation: Name put in the ``operation`` field of the run's log records
            (see :data:`bioamla.common.log.LOG_FIELDS`).

    Returns:
        A :class:`BatchResult` summarizing the run.
    """
    start_time = datetime.now()
    result = BatchResult(start_time=start_time.isoformat())

    rows = context.rows
    result.total_files = len(rows)

    completed = 0
    total = len(rows)

    def _fields(row: MetadataRow, **extra: Any) -> dict[str, dict[str, Any]]:
        return log_fields(
            operation=operation,
            file=str(row.file_path),
            index=completed + 1,
            total=total,
            elapsed_s=_elapsed(start_time),
            **extra,
        )

    for row in rows:
        try:
            if not row.file_path.exists():
                raise NotFoundError(f"File not found: {row.file_path}")
            value = process_row(row)
            result.successful += 1
            if value is not None:
                result.output_files.append(str(value))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("%s: processed %s", operation, row.file_path, **_fields(row))
        except Exception as e:
            result.failed += 1
            result.errors.append(f"{row.file_path}: {e}")
            if not continue_on_error:
                raise
            logger.warning(
                "Error processing %s: %s",
                row.file_path,
                e,
                **_fields(row, error=_short_error(e)),
            )
        completed += 1
        if on_progress:
            on_progress(completed, total)

    end_time = datetime.now()
    result.end_time = end_time.isoformat()
    result.duration_seconds = (end_time - start_time).total_seconds()
    _log_summary(operation, result)
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

    # Recorder columns keep their documented order; anything else is sorted.
    from bioamla.audio.metadata import RECORDING_METADATA_FIELDS

    recorder = [f for f in RECORDING_METADATA_FIELDS if f in new_fields]
    all_fieldnames.extend(recorder + sorted(new_fields - set(recorder)))

    # Paths in the CSV are relative to where it's written. A Path value (a file a
    # command just wrote) is made relative to that; relative paths read from the
    # input CSV are re-rooted when the CSV moves to output_dir.
    base = output_csv_path.parent
    moved = os.path.abspath(base) != os.path.abspath(context.csv_dir)

    def reroot(value: str) -> str:
        if not moved or not value or Path(value).is_absolute():
            return value
        return _csv_relative(context.csv_dir / value, base)

    rows_data = []
    for row in context.rows:
        file_name = row.file_name
        if row.output_path is None:  # analysis rows keep the input's path, re-rooted
            file_name = reroot(file_name)
        row_dict: dict[str, Any] = {"file_name": file_name}
        for key, value in row.metadata_fields.items():
            if isinstance(value, Path):
                value = _csv_relative(value, base)
            elif key in CSV_PATH_COLUMNS and isinstance(value, str):
                value = reroot(value)
            row_dict[key] = value
        rows_data.append(row_dict)

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=all_fieldnames)
    writer.writeheader()
    writer.writerows(rows_data)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.write_text(csv_buffer.getvalue(), encoding="utf-8")

    return output_csv_path


# =============================================================================
# Operation requests
# =============================================================================


@config_dataclass
class BatchRequest:
    """The input and output every batch operation takes (see :mod:`bioamla.operations`).

    Exactly one of ``input_dir`` (directory mode) and ``input_file`` (a metadata CSV
    with a ``file_name`` column) is given.

    Attributes:
        input_dir: Folder of audio files.
        input_file: Metadata CSV listing the files.
        output_dir: Where results go (in CSV mode, the updated CSV too; without it,
            CSV-mode outputs go next to their inputs).
        recursive: Search subfolders (directory mode).
        max_workers: Parallel workers.
        continue_on_error: Keep going after a file fails.
    """

    input_dir: PathStr | None = None
    input_file: PathStr | None = None
    output_dir: PathStr | None = None
    recursive: bool = True
    max_workers: int = 1
    continue_on_error: bool = True

    def to_config(self) -> BatchConfig:
        """The :class:`BatchConfig` for this request.

        Raises:
            InvalidInputError: Unless exactly one of ``input_dir`` and ``input_file``
                is given.
        """
        return BatchConfig(
            input_dir=self.input_dir,
            input_file=self.input_file,
            output_dir=self.output_dir or "",
            recursive=self.recursive,
            max_workers=self.max_workers,
            continue_on_error=self.continue_on_error,
            quiet=True,
        )


# =============================================================================
# Reading a BatchConfig
# =============================================================================


def config_csv_context(config: BatchConfig) -> CSVBatchContext:
    """The metadata CSV of a CSV-mode ``config``, with its output folder (see
    :func:`load_csv`)."""
    output_dir = str(config.output_dir) if config.output_dir else None
    return load_csv(config.input_file, output_dir)


def require_input_dir(config: BatchConfig) -> str:
    """``config.input_dir``, for commands that only run in directory mode.

    Raises:
        InvalidInputError: If ``config`` has no ``input_dir``.
    """
    if not config.input_dir:
        raise InvalidInputError("input_dir (--input-dir) is required for this command.")
    return str(config.input_dir)


def require_output_dir(config: BatchConfig) -> str:
    """``config.output_dir``, for commands that must write somewhere.

    Raises:
        InvalidInputError: If ``config`` has no ``output_dir``.
    """
    if not config.output_dir:
        raise InvalidInputError("output_dir (--output-dir) is required for this command.")
    return str(config.output_dir)


# =============================================================================
# CSV-mode runners: one per kind of command
# =============================================================================


def _finish(context: CSVBatchContext, result: BatchResult) -> BatchResult:
    result.metadata["metadata_csv"] = str(write_csv(context))
    return result


def run_csv_merge(
    context: CSVBatchContext,
    analyze_file: Callable[[Path], dict[str, Any]],
    *,
    clear: tuple[str, ...] = (),
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
    operation: str = "batch",
) -> BatchResult:
    """CSV mode for analyses: add each file's result columns to its row.

    Runs ``analyze_file`` on every listed file, merges the returned columns into the
    row and writes the updated CSV (columns unioned, ``file_name`` first). The
    ``clear`` columns are blanked on every row first, so a row that fails this run
    doesn't keep a value from an earlier one.

    Returns:
        The run's result; ``metadata["metadata_csv"]`` is the CSV written.
    """
    for row in context.rows:
        for column in clear:
            if column in row.metadata_fields or column in context.fieldnames:
                row.metadata_fields[column] = ""

    def _process(row: MetadataRow) -> str:
        merge_analysis_results(row, analyze_file(row.file_path))
        return str(row.file_path)

    result = run_csv_batch(
        context,
        _process,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation=operation,
    )
    return _finish(context, result)


def run_csv_transform(
    context: CSVBatchContext,
    transform_file: Callable[[Path, Path], Path | str],
    *,
    new_extension: str | None = None,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
    operation: str = "batch",
) -> BatchResult:
    """CSV mode for commands that write a new file per row (convert, resample, ...).

    ``transform_file(input_path, output_path)`` writes the new file (in place, or
    under the context's output folder; see :func:`resolve_output_path`) and returns
    where it went; the row's ``file_name`` is updated to it before the CSV is written.

    Returns:
        The run's result; ``metadata["metadata_csv"]`` is the CSV written.
    """

    def _process(row: MetadataRow) -> str:
        out_path = resolve_output_path(row.file_path, context, new_extension=new_extension)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_path = Path(transform_file(row.file_path, out_path))
        update_row_path(row, final_path, context)
        return str(final_path)

    result = run_csv_batch(
        context,
        _process,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation=operation,
    )
    return _finish(context, result)


def run_csv_process(
    context: CSVBatchContext,
    process_file: Callable[[Path], Any],
    *,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
    operation: str = "batch",
) -> BatchResult:
    """CSV mode for commands whose results go elsewhere (detections, predictions).

    Runs ``process_file`` on every listed file; the CSV is written back with its rows
    unchanged (paths re-resolved for its new location).

    Returns:
        The run's result; ``metadata["metadata_csv"]`` is the CSV written.
    """

    def _process(row: MetadataRow) -> str:
        process_file(row.file_path)
        return str(row.file_path)

    result = run_csv_batch(
        context,
        _process,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation=operation,
    )
    return _finish(context, result)


def run_csv_segment(
    context: CSVBatchContext,
    segment_file: Callable[[Path, Path], list[SegmentInfo]],
    *,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
    operation: str = "batch",
) -> BatchResult:
    """CSV mode for segmenting: each row becomes one row per segment written.

    ``segment_file(input_path, output_dir)`` writes the segments (next to where the
    file's output goes) and returns their :class:`SegmentInfo`. Each parent row is
    replaced by its segments' rows, carrying ``parent_file``, ``segment_id``,
    ``start_time``, ``end_time`` and ``duration``.

    Returns:
        The run's result; ``metadata["metadata_csv"]`` is the CSV written.
    """
    segment_mapping: dict[Path, list[SegmentInfo]] = {}

    def _process(row: MetadataRow) -> str:
        seg_dir = resolve_output_path(row.file_path, context).parent
        seg_dir.mkdir(parents=True, exist_ok=True)
        segments = segment_file(row.file_path, seg_dir)
        if segments:
            segment_mapping[row.file_path] = segments
        return str(row.file_path)

    result = run_csv_batch(
        context,
        _process,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation=operation,
    )

    new_rows: list[MetadataRow] = []
    for row in context.rows:
        if row.file_path in segment_mapping:
            new_rows.extend(expand_row_for_segments(row, segment_mapping[row.file_path], context))
        else:
            new_rows.append(row)
    context.rows = new_rows
    for name in ("parent_file", "segment_id", "start_time", "end_time", "duration"):
        if name not in context.fieldnames:
            context.fieldnames.append(name)
    return _finish(context, result)


def run_csv_predict_segments(
    context: CSVBatchContext,
    predict_file: Callable[[Path], list[dict[str, Any]]],
    *,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
    operation: str = "batch",
) -> BatchResult:
    """CSV mode for per-segment predictions: each row becomes one row per span.

    ``predict_file(input_path)`` returns one dict per span (``start_time``,
    ``end_time``, ``prediction``, ``confidence``, and any further columns, e.g. ranked
    labels). Each parent row is replaced by one row per span, keeping its
    ``file_name`` and adding ``segment_id`` and the span's columns. No files are written; the rows describe parts of the original file.

    Returns:
        The run's result; ``metadata["metadata_csv"]`` is the CSV written.
    """
    span_mapping: dict[Path, list[dict[str, Any]]] = {}

    def _process(row: MetadataRow) -> str:
        spans = predict_file(row.file_path)
        if spans:
            span_mapping[row.file_path] = spans
        return str(row.file_path)

    result = run_csv_batch(
        context,
        _process,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation=operation,
    )

    new_rows: list[MetadataRow] = []
    for row in context.rows:
        spans = span_mapping.get(row.file_path)
        if not spans:
            new_rows.append(row)
            continue
        for seg_id, span in enumerate(spans):
            fields = row.metadata_fields.copy()
            fields.update({"segment_id": seg_id, **span})
            new_rows.append(
                MetadataRow(
                    file_name=row.file_name,
                    file_path=row.file_path,
                    metadata_fields=fields,
                    output_path=row.output_path,
                )
            )
    context.rows = new_rows
    names = ["segment_id", "start_time", "end_time", "prediction", "confidence"]
    for spans in span_mapping.values():
        names += [k for span in spans for k in span if k not in names]
    for name in names:
        if name not in context.fieldnames:
            context.fieldnames.append(name)
    return _finish(context, result)


__all__ = [
    # Types
    "BatchConfig",
    "BatchRequest",
    "BatchResult",
    "SegmentInfo",
    "MetadataRow",
    "CSVBatchContext",
    "CSV_PATH_COLUMNS",
    # Discovery / runner
    "discover_files",
    "output_stems",
    "run_batch",
    "write_rows_csv",
    "run_csv_batch",
    # CSV helpers
    "load_csv",
    "write_csv",
    "resolve_file_path",
    "resolve_output_path",
    "update_row_path",
    "merge_analysis_results",
    "expand_row_for_segments",
    # Reading a BatchConfig
    "config_csv_context",
    "require_input_dir",
    "require_output_dir",
    # CSV-mode runners
    "run_csv_merge",
    "run_csv_transform",
    "run_csv_process",
    "run_csv_segment",
    "run_csv_predict_segments",
]

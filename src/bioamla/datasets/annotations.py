"""Time-frequency annotations: the :class:`Annotation` data structure and I/O.

This module holds the canonical :class:`Annotation` dataclass for the datasets
domain plus the format readers/writers that convert between in-memory annotations
and Raven Pro selection tables (.txt) and CSV files.

Functions raise :class:`~bioamla.exceptions.NotFoundError` when an input file is
missing and :class:`~bioamla.exceptions.AnnotationError` when parsing/writing
fails. Direct ``open()``/pathlib I/O is used throughout.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bioamla.exceptions import AnnotationError, NotFoundError

logger = logging.getLogger(__name__)


# =============================================================================
# Core Annotation Data Structure
# =============================================================================


@dataclass
class Annotation:
    """Represents a single time-frequency annotation.

    Stores annotation data for a region of an audio file, defined by time
    boundaries and optional frequency boundaries.

    Attributes:
        start_time: Start time in seconds
        end_time: End time in seconds
        low_freq: Lower frequency bound in Hz (optional)
        high_freq: Upper frequency bound in Hz (optional)
        label: Primary label/class for the annotation
        channel: Audio channel (1-indexed, default 1)
        confidence: Confidence score (0.0-1.0, optional)
        notes: Additional notes or comments
        custom_fields: Dictionary for storing custom annotation fields
    """

    start_time: float
    end_time: float
    low_freq: float | None = None
    high_freq: float | None = None
    label: str = ""
    channel: int = 1
    confidence: float | None = None
    notes: str = ""
    custom_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get the duration of the annotation in seconds."""
        return self.end_time - self.start_time

    @property
    def bandwidth(self) -> float | None:
        """Get the frequency bandwidth in Hz, or None if frequencies not set."""
        if self.low_freq is not None and self.high_freq is not None:
            return self.high_freq - self.low_freq
        return None

    @property
    def center_time(self) -> float:
        """Get the center time of the annotation."""
        return (self.start_time + self.end_time) / 2

    @property
    def center_freq(self) -> float | None:
        """Get the center frequency, or None if frequencies not set."""
        if self.low_freq is not None and self.high_freq is not None:
            return (self.low_freq + self.high_freq) / 2
        return None

    def overlaps_time(self, other: Annotation) -> bool:
        """Check if this annotation overlaps in time with another."""
        return not (self.end_time <= other.start_time or self.start_time >= other.end_time)

    def overlaps_freq(self, other: Annotation) -> bool:
        """Check if this annotation overlaps in frequency with another."""
        if self.low_freq is None or self.high_freq is None:
            return True  # If no freq bounds, assume overlap
        if other.low_freq is None or other.high_freq is None:
            return True
        return not (self.high_freq <= other.low_freq or self.low_freq >= other.high_freq)

    def overlaps(self, other: Annotation) -> bool:
        """Check if this annotation overlaps with another in time and frequency."""
        return self.overlaps_time(other) and self.overlaps_freq(other)

    def contains_time(self, time: float) -> bool:
        """Check if a time point falls within this annotation."""
        return self.start_time <= time <= self.end_time

    def contains_freq(self, freq: float) -> bool:
        """Check if a frequency falls within this annotation's bounds."""
        if self.low_freq is None or self.high_freq is None:
            return True
        return self.low_freq <= freq <= self.high_freq

    def to_dict(self) -> dict[str, Any]:
        """Convert annotation to a dictionary."""
        result = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "low_freq": self.low_freq,
            "high_freq": self.high_freq,
            "label": self.label,
            "channel": self.channel,
            "confidence": self.confidence,
            "notes": self.notes,
        }
        result.update(self.custom_fields)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        """Create an Annotation from a dictionary."""
        known_fields = {
            "start_time",
            "end_time",
            "low_freq",
            "high_freq",
            "label",
            "channel",
            "confidence",
            "notes",
        }
        custom = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            start_time=float(data.get("start_time", 0)),
            end_time=float(data.get("end_time", 0)),
            low_freq=float(data["low_freq"])
            if data.get("low_freq") not in (None, "", "nan")
            else None,
            high_freq=float(data["high_freq"])
            if data.get("high_freq") not in (None, "", "nan")
            else None,
            label=str(data.get("label", "")),
            channel=int(data.get("channel", 1)),
            confidence=float(data["confidence"])
            if data.get("confidence") not in (None, "", "nan")
            else None,
            notes=str(data.get("notes", "")),
            custom_fields=custom,
        )


# =============================================================================
# Result dataclass (kept for public API / library convenience)
# =============================================================================


@dataclass
class AnnotationResult:
    """Result of annotation import operations (annotations + optional summary)."""

    annotations: list[Annotation]
    file_path: str | None = None
    summary: dict[str, Any] | None = None


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_float(value: Any) -> float | None:
    """Parse a value to float, returning None if invalid or NaN."""
    if value is None or value == "" or value == "nan":
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    if result != result:  # NaN check without importing numpy
        return None
    return result


# =============================================================================
# Raven Selection Table Import/Export
# =============================================================================

# Standard Raven Pro column mappings
RAVEN_COLUMN_MAP = {
    "Selection": "selection",
    "View": "view",
    "Channel": "channel",
    "Begin Time (s)": "start_time",
    "End Time (s)": "end_time",
    "Low Freq (Hz)": "low_freq",
    "High Freq (Hz)": "high_freq",
    "Begin Path": "begin_path",
    "File Offset (s)": "file_offset",
    "Begin File": "begin_file",
    "Delta Time (s)": "delta_time",
    "Annotation": "label",
    "Species": "label",  # Alternative label column
}


def load_raven_selection_table(
    filepath: str, label_column: str | None = None, encoding: str = "utf-8"
) -> list[Annotation]:
    """Load annotations from a Raven Pro selection table file.

    Raven Pro exports tab-delimited text files with specific column headers.
    This function reads those files and converts them to Annotation objects.

    Args:
        filepath: Path to the Raven selection table file (.txt)
        label_column: Name of the column to use as label. If None, auto-detects
            from 'Annotation', 'Species', 'Label', or 'Class' columns.
        encoding: File encoding (default: utf-8)

    Returns:
        List of Annotation objects

    Raises:
        NotFoundError: If the file doesn't exist.
        AnnotationError: If the file cannot be parsed.
    """
    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Selection table not found: {filepath}")

    annotations: list[Annotation] = []

    try:
        with open(path, newline="", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter="\t")

            if label_column is None:
                fieldnames = reader.fieldnames or []
                for candidate in ["Annotation", "Species", "Label", "Class"]:
                    if candidate in fieldnames:
                        label_column = candidate
                        break

            for row in reader:
                try:
                    start_time = _parse_float(row.get("Begin Time (s)"))
                    end_time = _parse_float(row.get("End Time (s)"))

                    if start_time is None or end_time is None:
                        logger.warning(f"Skipping row with missing time values: {row}")
                        continue

                    low_freq = _parse_float(row.get("Low Freq (Hz)"))
                    high_freq = _parse_float(row.get("High Freq (Hz)"))
                    channel = int(row.get("Channel", 1) or 1)

                    label = ""
                    if label_column and label_column in row:
                        label = str(row[label_column]).strip()

                    standard_cols = set(RAVEN_COLUMN_MAP.keys())
                    custom_fields = {
                        k: v
                        for k, v in row.items()
                        if k not in standard_cols and k != label_column and v
                    }

                    annotations.append(
                        Annotation(
                            start_time=start_time,
                            end_time=end_time,
                            low_freq=low_freq,
                            high_freq=high_freq,
                            label=label,
                            channel=channel,
                            custom_fields=custom_fields,
                        )
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue
    except OSError as e:
        raise AnnotationError(f"Failed to read Raven selection table {filepath}: {e}") from e

    logger.info(f"Loaded {len(annotations)} annotations from {filepath}")
    return annotations


def save_raven_selection_table(
    annotations: list[Annotation],
    filepath: str,
    include_custom_fields: bool = True,
    encoding: str = "utf-8",
) -> str:
    """Save annotations to a Raven Pro selection table file.

    Creates a tab-delimited text file compatible with Raven Pro software.

    Args:
        annotations: List of Annotation objects to save
        filepath: Output file path (.txt)
        include_custom_fields: If True, include custom fields as additional columns
        encoding: File encoding (default: utf-8)

    Returns:
        Path to the saved file

    Raises:
        AnnotationError: If the file cannot be written.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "Selection",
        "View",
        "Channel",
        "Begin Time (s)",
        "End Time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Annotation",
    ]

    if include_custom_fields:
        custom_cols = set()
        for ann in annotations:
            custom_cols.update(ann.custom_fields.keys())
        columns.extend(sorted(custom_cols))

    try:
        with open(path, mode="w", newline="", encoding=encoding) as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
            writer.writeheader()

            for i, ann in enumerate(annotations, start=1):
                row = {
                    "Selection": i,
                    "View": "Spectrogram 1",
                    "Channel": ann.channel,
                    "Begin Time (s)": f"{ann.start_time:.6f}",
                    "End Time (s)": f"{ann.end_time:.6f}",
                    "Low Freq (Hz)": f"{ann.low_freq:.1f}" if ann.low_freq is not None else "",
                    "High Freq (Hz)": f"{ann.high_freq:.1f}" if ann.high_freq is not None else "",
                    "Annotation": ann.label,
                }
                if include_custom_fields:
                    row.update(ann.custom_fields)
                writer.writerow(row)
    except OSError as e:
        raise AnnotationError(f"Failed to write Raven selection table {filepath}: {e}") from e

    logger.info(f"Saved {len(annotations)} annotations to {filepath}")
    return str(path)


# =============================================================================
# CSV Annotation Import/Export
# =============================================================================


def load_csv_annotations(
    filepath: str,
    start_time_col: str = "start_time",
    end_time_col: str = "end_time",
    low_freq_col: str = "low_freq",
    high_freq_col: str = "high_freq",
    label_col: str = "label",
    encoding: str = "utf-8",
) -> list[Annotation]:
    """Load annotations from a CSV file with flexible column mapping.

    Args:
        filepath: Path to the CSV file
        start_time_col: Column name for start time
        end_time_col: Column name for end time
        low_freq_col: Column name for low frequency (optional in data)
        high_freq_col: Column name for high frequency (optional in data)
        label_col: Column name for label
        encoding: File encoding

    Returns:
        List of Annotation objects

    Raises:
        NotFoundError: If the file doesn't exist.
        AnnotationError: If the file cannot be parsed.
    """
    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"CSV file not found: {filepath}")

    annotations: list[Annotation] = []

    try:
        with open(path, newline="", encoding=encoding) as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    start_time = _parse_float(row.get(start_time_col))
                    end_time = _parse_float(row.get(end_time_col))

                    if start_time is None or end_time is None:
                        logger.warning(f"Skipping row with missing time values: {row}")
                        continue

                    low_freq = _parse_float(row.get(low_freq_col))
                    high_freq = _parse_float(row.get(high_freq_col))
                    label = str(row.get(label_col, "")).strip()

                    confidence = _parse_float(row.get("confidence"))
                    notes = str(row.get("notes", "")).strip()
                    channel = int(row.get("channel", 1) or 1)

                    known_cols = {
                        start_time_col,
                        end_time_col,
                        low_freq_col,
                        high_freq_col,
                        label_col,
                        "confidence",
                        "notes",
                        "channel",
                    }
                    custom_fields = {k: v for k, v in row.items() if k not in known_cols and v}

                    annotations.append(
                        Annotation(
                            start_time=start_time,
                            end_time=end_time,
                            low_freq=low_freq,
                            high_freq=high_freq,
                            label=label,
                            channel=channel,
                            confidence=confidence,
                            notes=notes,
                            custom_fields=custom_fields,
                        )
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing CSV row: {e}")
                    continue
    except OSError as e:
        raise AnnotationError(f"Failed to read CSV annotations {filepath}: {e}") from e

    logger.info(f"Loaded {len(annotations)} annotations from {filepath}")
    return annotations


def save_csv_annotations(
    annotations: list[Annotation],
    filepath: str,
    include_custom_fields: bool = True,
    encoding: str = "utf-8",
) -> str:
    """Save annotations to a CSV file.

    Args:
        annotations: List of Annotation objects
        filepath: Output file path
        include_custom_fields: If True, include custom fields as additional columns
        encoding: File encoding

    Returns:
        Path to the saved file

    Raises:
        AnnotationError: If the file cannot be written.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "start_time",
        "end_time",
        "low_freq",
        "high_freq",
        "label",
        "channel",
        "confidence",
        "notes",
    ]

    if include_custom_fields:
        custom_cols = set()
        for ann in annotations:
            custom_cols.update(ann.custom_fields.keys())
        columns.extend(sorted(custom_cols))

    try:
        with open(path, mode="w", newline="", encoding=encoding) as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for ann in annotations:
                row = {
                    "start_time": ann.start_time,
                    "end_time": ann.end_time,
                    "low_freq": ann.low_freq if ann.low_freq is not None else "",
                    "high_freq": ann.high_freq if ann.high_freq is not None else "",
                    "label": ann.label,
                    "channel": ann.channel,
                    "confidence": ann.confidence if ann.confidence is not None else "",
                    "notes": ann.notes,
                }
                if include_custom_fields:
                    row.update(ann.custom_fields)
                writer.writerow(row)
    except OSError as e:
        raise AnnotationError(f"Failed to write CSV annotations {filepath}: {e}") from e

    logger.info(f"Saved {len(annotations)} annotations to {filepath}")
    return str(path)


def save_json_annotations(annotations: list[Annotation], filepath: str) -> str:
    """Save annotations to a JSON file (list of annotation dicts).

    Args:
        annotations: List of Annotation objects
        filepath: Output file path (.json)

    Returns:
        Path to the saved file

    Raises:
        AnnotationError: If the file cannot be written.
    """
    import json

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = [ann.to_dict() for ann in annotations]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except OSError as e:
        raise AnnotationError(f"Failed to write JSON annotations {filepath}: {e}") from e

    logger.info(f"Saved {len(annotations)} annotations to {filepath}")
    return str(path)


# =============================================================================
# Bioamla Annotation Format (JSON container with metadata header)
# =============================================================================

# Versioned format identifier written into every bioamla annotation file. Bump
# the trailing number on breaking schema changes so loaders can adapt.
BIOAMLA_ANNOTATION_FORMAT = "bioamla-annotations/1"

# Header fields managed by the format itself; everything else in a loaded
# header is returned to the caller as free-form metadata.
_BIOAMLA_RESERVED_KEYS = {"format", "annotations"}


def save_bioamla_annotations(
    annotations: list[Annotation],
    filepath: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save annotations in the bioamla JSON format.

    Unlike a flat CSV/Raven table, this format carries a file-level metadata
    header (e.g. ``audio_file``, ``sample_rate``, ``duration``) alongside the
    annotation records, so an annotation file is self-describing and stays
    linked to the recording it describes.

    Args:
        annotations: List of Annotation objects.
        filepath: Output file path (``.json``).
        metadata: Optional file-level metadata (audio_file, sample_rate,
            duration, channels, or any custom keys) merged into the header.

    Returns:
        Path to the saved file.

    Raises:
        AnnotationError: If the file cannot be written.
    """
    import json

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    header: dict[str, Any] = {"format": BIOAMLA_ANNOTATION_FORMAT}
    if metadata:
        # Never let caller metadata clobber the reserved keys.
        header.update({k: v for k, v in metadata.items() if k not in _BIOAMLA_RESERVED_KEYS})
    header["annotations"] = [ann.to_dict() for ann in annotations]

    try:
        path.write_text(json.dumps(header, indent=2, default=str), encoding="utf-8")
    except OSError as e:
        raise AnnotationError(f"Failed to write bioamla annotations {filepath}: {e}") from e

    logger.info(f"Saved {len(annotations)} annotations to {filepath}")
    return str(path)


def load_bioamla_annotations(filepath: str) -> tuple[list[Annotation], dict[str, Any]]:
    """Load annotations from a bioamla JSON format file.

    Args:
        filepath: Path to a ``.json`` file in the bioamla annotation format.

    Returns:
        A ``(annotations, metadata)`` tuple, where ``metadata`` is the
        file-level header (audio_file, sample_rate, etc.) with the reserved
        ``format`` and ``annotations`` keys removed.

    Raises:
        NotFoundError: If the file doesn't exist.
        AnnotationError: If the file cannot be parsed or has the wrong shape.
    """
    import json

    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Bioamla annotation file not found: {filepath}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise AnnotationError(f"Failed to read bioamla annotations {filepath}: {e}") from e

    if not isinstance(data, dict) or "annotations" not in data:
        raise AnnotationError(
            f"Not a bioamla annotation file (missing 'annotations' key): {filepath}"
        )

    records = data.get("annotations", [])
    if not isinstance(records, list):
        raise AnnotationError(f"'annotations' must be a list in {filepath}")

    annotations = [Annotation.from_dict(rec) for rec in records]
    metadata = {k: v for k, v in data.items() if k not in _BIOAMLA_RESERVED_KEYS}

    logger.info(f"Loaded {len(annotations)} annotations from {filepath}")
    return annotations, metadata


def save_parquet_annotations(annotations: list[Annotation], filepath: str) -> str:
    """Save annotations to a Parquet file.

    Args:
        annotations: List of Annotation objects
        filepath: Output file path (.parquet)

    Returns:
        Path to the saved file

    Raises:
        AnnotationError: If the file cannot be written.
    """

    import pandas as pd

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = pd.DataFrame([ann.to_dict() for ann in annotations])
        df.to_parquet(str(path), index=False)
    except Exception as e:
        raise AnnotationError(f"Failed to write Parquet annotations {filepath}: {e}") from e

    logger.info(f"Saved {len(annotations)} annotations to {filepath}")
    return str(path)


# =============================================================================
# Annotation Creation
# =============================================================================


def create_annotation(
    start_time: float,
    end_time: float,
    label: str = "",
    low_freq: float | None = None,
    high_freq: float | None = None,
    channel: int = 1,
    confidence: float | None = None,
    notes: str = "",
) -> Annotation:
    """Create a validated :class:`Annotation`.

    Raises:
        AnnotationError: If ``end_time <= start_time`` or ``high_freq <= low_freq``.
    """
    if end_time <= start_time:
        raise AnnotationError("end_time must be greater than start_time")
    if low_freq is not None and high_freq is not None and high_freq <= low_freq:
        raise AnnotationError("high_freq must be greater than low_freq")

    return Annotation(
        start_time=start_time,
        end_time=end_time,
        low_freq=low_freq,
        high_freq=high_freq,
        label=label,
        channel=channel,
        confidence=confidence,
        notes=notes,
    )


def predictions_to_annotations(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_confidence: float = 0.0,
    exclude_labels: Iterable[str] | None = None,
) -> list[Annotation]:
    """Convert segment-level model predictions into Annotations for manual review.

    Bridges model inference output — e.g. the DataFrame from
    ``segmented_wave_file_inference`` (columns ``filepath/start/stop/prediction``)
    or a batch-predict CSV — into editable :class:`Annotation` objects that a
    human can correct and then feed to ``dataset extract-clips``. This closes the
    predict → review → dataset loop.

    Recognizes both ``start``/``stop`` and ``start_time``/``end_time`` time keys,
    and ``prediction`` or ``label`` for the class. A source filename
    (``filepath``/``file_name``/``source_file``) is preserved in
    ``custom_fields['source_file']`` when present.

    Args:
        rows: Iterable of prediction rows (mappings); pass a DataFrame via
            ``df.to_dict("records")``.
        min_confidence: Drop predictions whose confidence is below this. Rows
            without a confidence value are always kept.
        exclude_labels: Labels to drop (e.g. a background/negative class).

    Returns:
        One :class:`Annotation` per kept prediction row.
    """
    excluded = {str(x) for x in (exclude_labels or ())}
    annotations: list[Annotation] = []
    for row in rows:
        label = str(row.get("label", row.get("prediction", "")) or "")
        if label in excluded:
            continue

        raw_conf = row.get("confidence")
        confidence = float(raw_conf) if raw_conf not in (None, "", "nan") else None
        if confidence is not None and confidence < min_confidence:
            continue

        start = row.get("start_time", row.get("start"))
        end = row.get("end_time", row.get("stop", row.get("end")))
        annotation = Annotation(
            start_time=float(start) if start is not None else 0.0,
            end_time=float(end) if end is not None else 0.0,
            label=label,
            confidence=confidence,
        )
        source = row.get("filepath") or row.get("file_name") or row.get("source_file")
        if source:
            annotation.custom_fields["source_file"] = str(source)
        annotations.append(annotation)
    return annotations


# =============================================================================
# Batch Operations and Summarization
# =============================================================================


def load_annotations_from_directory(
    directory: str, file_pattern: str = "*.txt", format: str = "raven"
) -> dict[str, list[Annotation]]:
    """Load annotations from all matching files in a directory.

    Args:
        directory: Path to directory containing annotation files
        file_pattern: Glob pattern for matching files
        format: Annotation format ('raven' or 'csv')

    Returns:
        Dictionary mapping filename to list of annotations

    Raises:
        NotFoundError: If the directory doesn't exist.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise NotFoundError(f"Directory not found: {directory}")

    results: dict[str, list[Annotation]] = {}
    loader = load_raven_selection_table if format == "raven" else load_csv_annotations

    for filepath in dir_path.glob(file_pattern):
        try:
            results[filepath.name] = loader(str(filepath))
        except Exception as e:
            logger.warning(f"Error loading {filepath}: {e}")

    logger.info(f"Loaded annotations from {len(results)} files")
    return results


def get_unique_labels(annotations: list[Annotation]) -> list[str]:
    """Get the sorted list of unique non-empty labels from annotations."""
    labels = {ann.label for ann in annotations if ann.label}
    return sorted(labels)


def summarize_annotations(annotations: list[Annotation]) -> dict[str, Any]:
    """Generate summary statistics (counts, label histogram, durations)."""
    if not annotations:
        return {
            "total_annotations": 0,
            "unique_labels": 0,
            "labels": {},
            "total_duration": 0.0,
            "min_duration": 0.0,
            "max_duration": 0.0,
            "mean_duration": 0.0,
        }

    labels = get_unique_labels(annotations)
    durations = [ann.duration for ann in annotations]
    label_counts: dict[str, int] = {}
    for ann in annotations:
        label_counts[ann.label] = label_counts.get(ann.label, 0) + 1

    return {
        "total_annotations": len(annotations),
        "unique_labels": len(labels),
        "labels": label_counts,
        "total_duration": sum(durations),
        "min_duration": min(durations),
        "max_duration": max(durations),
        "mean_duration": sum(durations) / len(durations),
    }

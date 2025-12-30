# services/annotation.py
"""
Service for managing audio annotations with support for:
- File-based CRUD operations on annotations
- Raven selection table import/export
- CSV/Parquet/JSON export
- Audio clip extraction from annotations
- Measurement computation for annotations

This module also provides core annotation data structures and I/O functions:
- Annotation dataclass for time-frequency annotations
- CSV and Raven selection table import/export
- Annotation summarization utilities
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bioamla.core.files import TextFile
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.base import BaseService, ServiceResult

logger = logging.getLogger(__name__)


# =============================================================================
# Core Annotation Data Structure
# =============================================================================


@dataclass
class Annotation:
    """
    Represents a single time-frequency annotation.

    This class stores annotation data for a region of an audio file,
    defined by time boundaries and optional frequency boundaries.

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
    low_freq: Optional[float] = None
    high_freq: Optional[float] = None
    label: str = ""
    channel: int = 1
    confidence: Optional[float] = None
    notes: str = ""
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get the duration of the annotation in seconds."""
        return self.end_time - self.start_time

    @property
    def bandwidth(self) -> Optional[float]:
        """Get the frequency bandwidth in Hz, or None if frequencies not set."""
        if self.low_freq is not None and self.high_freq is not None:
            return self.high_freq - self.low_freq
        return None

    @property
    def center_time(self) -> float:
        """Get the center time of the annotation."""
        return (self.start_time + self.end_time) / 2

    @property
    def center_freq(self) -> Optional[float]:
        """Get the center frequency, or None if frequencies not set."""
        if self.low_freq is not None and self.high_freq is not None:
            return (self.low_freq + self.high_freq) / 2
        return None

    def overlaps_time(self, other: "Annotation") -> bool:
        """Check if this annotation overlaps in time with another."""
        return not (self.end_time <= other.start_time or self.start_time >= other.end_time)

    def overlaps_freq(self, other: "Annotation") -> bool:
        """Check if this annotation overlaps in frequency with another."""
        if self.low_freq is None or self.high_freq is None:
            return True  # If no freq bounds, assume overlap
        if other.low_freq is None or other.high_freq is None:
            return True
        return not (self.high_freq <= other.low_freq or self.low_freq >= other.high_freq)

    def overlaps(self, other: "Annotation") -> bool:
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

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
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
# Helper Functions
# =============================================================================


def _parse_float(value: Any) -> Optional[float]:
    """Parse a value to float, returning None if invalid."""
    if value is None or value == "" or value == "nan":
        return None
    try:
        result = float(value)
        if np.isnan(result):
            return None
        return result
    except (ValueError, TypeError):
        return None


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
    filepath: str, label_column: Optional[str] = None, encoding: str = "utf-8"
) -> List[Annotation]:
    """
    Load annotations from a Raven Pro selection table file.

    Raven Pro exports tab-delimited text files with specific column headers.
    This function reads those files and converts them to Annotation objects.

    Args:
        filepath: Path to the Raven selection table file (.txt)
        label_column: Name of the column to use as label. If None, auto-detects
            from 'Annotation', 'Species', or 'Label' columns.
        encoding: File encoding (default: utf-8)

    Returns:
        List of Annotation objects

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing

    Example:
        >>> annotations = load_raven_selection_table("selections.txt")
        >>> for ann in annotations:
        ...     print(f"{ann.label}: {ann.start_time:.2f}s - {ann.end_time:.2f}s")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Selection table not found: {filepath}")

    annotations = []

    with TextFile(path, mode="r", newline="", encoding=encoding) as f:
        # Raven uses tab-delimited format
        reader = csv.DictReader(f.handle, delimiter="\t")

        # Auto-detect label column if not specified
        if label_column is None:
            fieldnames = reader.fieldnames or []
            for candidate in ["Annotation", "Species", "Label", "Class"]:
                if candidate in fieldnames:
                    label_column = candidate
                    break

        for row in reader:
            try:
                # Map Raven columns to standard format
                start_time = _parse_float(row.get("Begin Time (s)"))
                end_time = _parse_float(row.get("End Time (s)"))

                if start_time is None or end_time is None:
                    logger.warning(f"Skipping row with missing time values: {row}")
                    continue

                low_freq = _parse_float(row.get("Low Freq (Hz)"))
                high_freq = _parse_float(row.get("High Freq (Hz)"))
                channel = int(row.get("Channel", 1) or 1)

                # Get label
                label = ""
                if label_column and label_column in row:
                    label = str(row[label_column]).strip()

                # Collect custom fields (non-standard Raven columns)
                standard_cols = set(RAVEN_COLUMN_MAP.keys())
                custom_fields = {
                    k: v
                    for k, v in row.items()
                    if k not in standard_cols and k != label_column and v
                }

                ann = Annotation(
                    start_time=start_time,
                    end_time=end_time,
                    low_freq=low_freq,
                    high_freq=high_freq,
                    label=label,
                    channel=channel,
                    custom_fields=custom_fields,
                )
                annotations.append(ann)

            except (ValueError, KeyError) as e:
                logger.warning(f"Error parsing row: {e}")
                continue

    logger.info(f"Loaded {len(annotations)} annotations from {filepath}")
    return annotations


def save_raven_selection_table(
    annotations: List[Annotation],
    filepath: str,
    include_custom_fields: bool = True,
    encoding: str = "utf-8",
) -> str:
    """
    Save annotations to a Raven Pro selection table file.

    Creates a tab-delimited text file compatible with Raven Pro software.

    Args:
        annotations: List of Annotation objects to save
        filepath: Output file path (.txt)
        include_custom_fields: If True, include custom fields as additional columns
        encoding: File encoding (default: utf-8)

    Returns:
        Path to the saved file

    Example:
        >>> annotations = [
        ...     Annotation(start_time=1.0, end_time=2.0, label="bird_song",
        ...                low_freq=1000, high_freq=8000)
        ... ]
        >>> save_raven_selection_table(annotations, "output.txt")
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build column list
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

    # Collect all custom fields
    if include_custom_fields:
        custom_cols = set()
        for ann in annotations:
            custom_cols.update(ann.custom_fields.keys())
        columns.extend(sorted(custom_cols))

    with TextFile(path, mode="w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f.handle, fieldnames=columns, delimiter="\t")
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
) -> List[Annotation]:
    """
    Load annotations from a CSV file.

    This function supports flexible column mapping for different CSV formats.

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

    Example:
        >>> annotations = load_csv_annotations(
        ...     "annotations.csv",
        ...     start_time_col="begin",
        ...     end_time_col="end",
        ...     label_col="species"
        ... )
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    annotations = []

    with TextFile(path, mode="r", newline="", encoding=encoding) as f:
        reader = csv.DictReader(f.handle)

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

                # Confidence and notes
                confidence = _parse_float(row.get("confidence"))
                notes = str(row.get("notes", "")).strip()
                channel = int(row.get("channel", 1) or 1)

                # Custom fields
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

                ann = Annotation(
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
                annotations.append(ann)

            except (ValueError, KeyError) as e:
                logger.warning(f"Error parsing CSV row: {e}")
                continue

    logger.info(f"Loaded {len(annotations)} annotations from {filepath}")
    return annotations


def save_csv_annotations(
    annotations: List[Annotation],
    filepath: str,
    include_custom_fields: bool = True,
    encoding: str = "utf-8",
) -> str:
    """
    Save annotations to a CSV file.

    Args:
        annotations: List of Annotation objects
        filepath: Output file path
        include_custom_fields: If True, include custom fields as additional columns
        encoding: File encoding

    Returns:
        Path to the saved file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build column list
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

    with TextFile(path, mode="w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f.handle, fieldnames=columns)
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

    logger.info(f"Saved {len(annotations)} annotations to {filepath}")
    return str(path)


# =============================================================================
# Batch Operations and Summarization
# =============================================================================


def load_annotations_from_directory(
    directory: str, file_pattern: str = "*.txt", format: str = "raven"
) -> Dict[str, List[Annotation]]:
    """
    Load annotations from all matching files in a directory.

    Args:
        directory: Path to directory containing annotation files
        file_pattern: Glob pattern for matching files
        format: Annotation format ('raven' or 'csv')

    Returns:
        Dictionary mapping filename to list of annotations
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    results = {}
    loader = load_raven_selection_table if format == "raven" else load_csv_annotations

    for filepath in dir_path.glob(file_pattern):
        try:
            annotations = loader(str(filepath))
            results[filepath.name] = annotations
        except Exception as e:
            logger.warning(f"Error loading {filepath}: {e}")

    logger.info(f"Loaded annotations from {len(results)} files")
    return results


def get_unique_labels(annotations: List[Annotation]) -> List[str]:
    """
    Get sorted list of unique labels from annotations.

    Args:
        annotations: List of Annotation objects

    Returns:
        Sorted list of unique label strings
    """
    labels = {ann.label for ann in annotations if ann.label}
    return sorted(labels)


def summarize_annotations(annotations: List[Annotation]) -> Dict[str, Any]:
    """
    Generate a summary of annotation statistics.

    Args:
        annotations: List of annotations to summarize

    Returns:
        Dictionary with summary statistics
    """
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
    label_counts = {}
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


# =============================================================================
# Service Result Dataclasses
# =============================================================================


@dataclass
class AnnotationResult:
    """Result of annotation operations."""

    annotations: List[Annotation]
    file_path: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None


@dataclass
class ClipExtractionResult:
    """Result of clip extraction operation."""

    total_clips: int = 0
    extracted_clips: List[str] = field(default_factory=list)
    failed_clips: List[str] = field(default_factory=list)
    output_directory: Optional[str] = None


@dataclass
class MeasurementResult:
    """Result of measurement computation."""

    annotation_id: Optional[str] = None
    measurements: Dict[str, float] = field(default_factory=dict)


class AnnotationService(BaseService):
    """
    Service for annotation operations.

    Provides a unified interface for:
    - Loading and saving annotations in various formats
    - Creating annotations in memory
    - Extracting audio clips from annotations
    - Computing measurements for annotations

    All file I/O operations are delegated to the file repository.

    Usage:
        from bioamla.repository.local import LocalFileRepository

        svc = AnnotationService(LocalFileRepository())
        result = svc.import_raven("selections.txt")
        annotations = result.data.annotations
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize AnnotationService.

        Args:
            file_repository: File repository for all file I/O operations (required).
        """
        super().__init__(file_repository)
        self._annotations: List[Annotation] = []

    # =========================================================================
    # Import Operations
    # =========================================================================

    def import_raven(
        self,
        filepath: str,
        label_column: Optional[str] = None,
    ) -> ServiceResult[AnnotationResult]:
        """
        Import annotations from a Raven selection table.

        Args:
            filepath: Path to the Raven selection table (.txt)
            label_column: Optional column name to use for labels

        Returns:
            ServiceResult containing AnnotationResult with imported annotations
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            annotations = load_raven_selection_table(filepath, label_column=label_column)
            summary = summarize_annotations(annotations)

            return ServiceResult.ok(
                data=AnnotationResult(annotations=annotations, file_path=filepath, summary=summary),
                message=f"Imported {len(annotations)} annotations from Raven table",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to import Raven table: {e}")
            return ServiceResult.fail(f"Failed to import Raven table: {e}")

    def import_csv(
        self,
        filepath: str,
        start_time_col: str = "start_time",
        end_time_col: str = "end_time",
        low_freq_col: str = "low_freq",
        high_freq_col: str = "high_freq",
        label_col: str = "label",
    ) -> ServiceResult[AnnotationResult]:
        """
        Import annotations from a CSV file.

        Args:
            filepath: Path to the CSV file
            start_time_col: Column name for start time
            end_time_col: Column name for end time
            low_freq_col: Column name for low frequency
            high_freq_col: Column name for high frequency
            label_col: Column name for label

        Returns:
            ServiceResult containing AnnotationResult with imported annotations
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            annotations = load_csv_annotations(
                filepath,
                start_time_col=start_time_col,
                end_time_col=end_time_col,
                low_freq_col=low_freq_col,
                high_freq_col=high_freq_col,
                label_col=label_col,
            )
            summary = summarize_annotations(annotations)

            return ServiceResult.ok(
                data=AnnotationResult(annotations=annotations, file_path=filepath, summary=summary),
                message=f"Imported {len(annotations)} annotations from CSV",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to import CSV: {e}")
            return ServiceResult.fail(f"Failed to import CSV: {e}")

    # =========================================================================
    # Export Operations
    # =========================================================================

    def export_raven(
        self,
        annotations: List[Annotation],
        output_path: str,
        include_custom_fields: bool = True,
    ) -> ServiceResult[str]:
        """
        Export annotations to a Raven selection table.

        Args:
            annotations: List of annotations to export
            output_path: Output file path (.txt)
            include_custom_fields: Include custom fields as additional columns

        Returns:
            ServiceResult containing the output file path
        """
        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            saved_path = save_raven_selection_table(
                annotations, output_path, include_custom_fields=include_custom_fields
            )

            return ServiceResult.ok(
                data=saved_path,
                message=f"Exported {len(annotations)} annotations to Raven table",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to export Raven table: {e}")
            return ServiceResult.fail(f"Failed to export Raven table: {e}")

    def export_csv(
        self,
        annotations: List[Annotation],
        output_path: str,
        include_custom_fields: bool = True,
    ) -> ServiceResult[str]:
        """
        Export annotations to a CSV file.

        Args:
            annotations: List of annotations to export
            output_path: Output file path (.csv)
            include_custom_fields: Include custom fields as additional columns

        Returns:
            ServiceResult containing the output file path
        """
        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            saved_path = save_csv_annotations(
                annotations, output_path, include_custom_fields=include_custom_fields
            )

            return ServiceResult.ok(
                data=saved_path,
                message=f"Exported {len(annotations)} annotations to CSV",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to export CSV: {e}")
            return ServiceResult.fail(f"Failed to export CSV: {e}")

    def export_parquet(
        self,
        annotations: List[Annotation],
        output_path: str,
    ) -> ServiceResult[str]:
        """
        Export annotations to a Parquet file.

        Args:
            annotations: List of annotations to export
            output_path: Output file path (.parquet)

        Returns:
            ServiceResult containing the output file path
        """
        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            import pandas as pd

            # Convert annotations to list of dicts
            data = [ann.to_dict() for ann in annotations]
            df = pd.DataFrame(data)

            # Ensure output path
            path = Path(output_path)
            self.file_repository.mkdir(path.parent, parents=True)

            df.to_parquet(str(path), index=False)

            return ServiceResult.ok(
                data=str(path),
                message=f"Exported {len(annotations)} annotations to Parquet",
                count=len(annotations),
            )

        except ImportError:
            return ServiceResult.fail("pandas and pyarrow required for Parquet export")
        except Exception as e:
            logger.exception(f"Failed to export Parquet: {e}")
            return ServiceResult.fail(f"Failed to export Parquet: {e}")

    def export_json(
        self,
        annotations: List[Annotation],
        output_path: str,
    ) -> ServiceResult[str]:
        """
        Export annotations to a JSON file.

        Args:
            annotations: List of annotations to export
            output_path: Output file path (.json)

        Returns:
            ServiceResult containing the output file path
        """
        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            path = Path(output_path)
            self.file_repository.mkdir(path.parent, parents=True)

            data = [ann.to_dict() for ann in annotations]
            content = json.dumps(data, indent=2, default=str)
            self.file_repository.write_text(path, content)

            return ServiceResult.ok(
                data=str(path),
                message=f"Exported {len(annotations)} annotations to JSON",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to export JSON: {e}")
            return ServiceResult.fail(f"Failed to export JSON: {e}")

    # =========================================================================
    # Annotation Creation
    # =========================================================================

    def create_annotation(
        self,
        start_time: float,
        end_time: float,
        label: str = "",
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
        channel: int = 1,
        confidence: Optional[float] = None,
        notes: str = "",
    ) -> ServiceResult[Annotation]:
        """
        Create a new annotation.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            label: Label/class for the annotation
            low_freq: Lower frequency bound in Hz
            high_freq: Upper frequency bound in Hz
            channel: Audio channel (1-indexed)
            confidence: Confidence score (0.0-1.0)
            notes: Additional notes

        Returns:
            ServiceResult containing the created annotation
        """
        if end_time <= start_time:
            return ServiceResult.fail("end_time must be greater than start_time")

        if low_freq is not None and high_freq is not None and high_freq <= low_freq:
            return ServiceResult.fail("high_freq must be greater than low_freq")

        try:
            annotation = Annotation(
                start_time=start_time,
                end_time=end_time,
                low_freq=low_freq,
                high_freq=high_freq,
                label=label,
                channel=channel,
                confidence=confidence,
                notes=notes,
            )

            return ServiceResult.ok(data=annotation, message="Created annotation")

        except Exception as e:
            logger.exception(f"Failed to create annotation: {e}")
            return ServiceResult.fail(f"Failed to create annotation: {e}")

    # =========================================================================
    # Clip Extraction
    # =========================================================================

    def extract_clips(
        self,
        annotations: List[Annotation],
        audio_path: str,
        output_dir: str,
        padding_ms: float = 0.0,
        format: str = "wav",
        include_label_in_filename: bool = True,
    ) -> ServiceResult[ClipExtractionResult]:
        """
        Extract audio clips from annotations.

        Args:
            annotations: List of annotations to extract
            audio_path: Path to the source audio file
            output_dir: Directory for output clips
            padding_ms: Padding in milliseconds to add before/after each clip
            format: Output audio format (wav, flac, etc.)
            include_label_in_filename: Include label in clip filename

        Returns:
            ServiceResult containing ClipExtractionResult
        """
        error = self._validate_input_path(audio_path)
        if error:
            return ServiceResult.fail(error)

        error = self._validate_output_path(output_dir)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.adapters.pydub import load_audio

            # Load audio file
            audio_data, sample_rate = load_audio(audio_path)

            # Handle mono/stereo
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)

            output_path = Path(output_dir)
            self.file_repository.mkdir(output_path, parents=True)

            padding_samples = int(padding_ms * sample_rate / 1000)
            total_samples = len(audio_data)

            result = ClipExtractionResult(
                total_clips=len(annotations), output_directory=str(output_path)
            )

            audio_stem = Path(audio_path).stem

            for i, ann in enumerate(annotations):
                try:
                    # Calculate sample indices
                    start_sample = max(0, int(ann.start_time * sample_rate) - padding_samples)
                    end_sample = min(
                        total_samples, int(ann.end_time * sample_rate) + padding_samples
                    )

                    # Extract clip
                    clip = audio_data[start_sample:end_sample]

                    # Build filename
                    if include_label_in_filename and ann.label:
                        safe_label = ann.label.replace(" ", "_").replace("/", "-")
                        filename = f"{audio_stem}_{i:04d}_{safe_label}.{format}"
                    else:
                        filename = f"{audio_stem}_{i:04d}.{format}"

                    clip_path = output_path / filename

                    # Write clip
                    from bioamla.adapters.pydub import save_audio

                    save_audio(str(clip_path), clip, sample_rate)
                    result.extracted_clips.append(str(clip_path))

                except Exception as e:
                    logger.warning(f"Failed to extract clip {i}: {e}")
                    result.failed_clips.append(f"Clip {i}: {e}")

            return ServiceResult.ok(
                data=result,
                message=f"Extracted {len(result.extracted_clips)} clips",
                extracted=len(result.extracted_clips),
                failed=len(result.failed_clips),
            )

        except Exception as e:
            logger.exception(f"Failed to extract clips: {e}")
            return ServiceResult.fail(f"Failed to extract clips: {e}")

    # =========================================================================
    # Measurement Operations
    # =========================================================================

    def compute_measurements(
        self,
        annotation: Annotation,
        audio_path: str,
        metrics: Optional[List[str]] = None,
    ) -> ServiceResult[MeasurementResult]:
        """
        Compute acoustic measurements for an annotation.

        Args:
            annotation: The annotation to measure
            audio_path: Path to the source audio file
            metrics: List of metrics to compute. If None, computes default set.
                     Options: duration, bandwidth, rms, peak, centroid, etc.

        Returns:
            ServiceResult containing MeasurementResult
        """
        error = self._validate_input_path(audio_path)
        if error:
            return ServiceResult.fail(error)

        try:
            from scipy import signal as scipy_signal

            from bioamla.adapters.pydub import load_audio

            # Load audio region
            audio_data, sample_rate = load_audio(audio_path)

            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)

            # Extract the annotation region
            start_sample = int(annotation.start_time * sample_rate)
            end_sample = int(annotation.end_time * sample_rate)

            # Use specified channel or first channel
            channel_idx = min(annotation.channel - 1, audio_data.shape[1] - 1)
            clip = audio_data[start_sample:end_sample, channel_idx]

            # Default metrics
            if metrics is None:
                metrics = ["duration", "bandwidth", "rms", "peak", "centroid"]

            measurements = {}

            # Basic temporal measurements
            if "duration" in metrics:
                measurements["duration"] = annotation.duration

            if "bandwidth" in metrics and annotation.bandwidth is not None:
                measurements["bandwidth"] = annotation.bandwidth

            # Amplitude measurements
            if "rms" in metrics:
                measurements["rms"] = float(np.sqrt(np.mean(clip**2)))

            if "peak" in metrics:
                measurements["peak"] = float(np.max(np.abs(clip)))

            if "crest_factor" in metrics:
                rms = np.sqrt(np.mean(clip**2))
                peak = np.max(np.abs(clip))
                measurements["crest_factor"] = float(peak / rms) if rms > 0 else 0.0

            # Spectral measurements
            if any(m in metrics for m in ["centroid", "bandwidth_spectral", "rolloff"]):
                # Compute spectrum
                n_fft = min(2048, len(clip))
                freqs, psd = scipy_signal.welch(clip, sample_rate, nperseg=n_fft)

                # Apply frequency bounds if specified
                if annotation.low_freq is not None and annotation.high_freq is not None:
                    mask = (freqs >= annotation.low_freq) & (freqs <= annotation.high_freq)
                    freqs = freqs[mask]
                    psd = psd[mask]

                if len(psd) > 0 and np.sum(psd) > 0:
                    if "centroid" in metrics:
                        measurements["centroid"] = float(np.sum(freqs * psd) / np.sum(psd))

                    if "bandwidth_spectral" in metrics:
                        centroid = np.sum(freqs * psd) / np.sum(psd)
                        measurements["bandwidth_spectral"] = float(
                            np.sqrt(np.sum((freqs - centroid) ** 2 * psd) / np.sum(psd))
                        )

                    if "rolloff" in metrics:
                        cumsum = np.cumsum(psd)
                        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
                        measurements["rolloff"] = float(freqs[min(rolloff_idx, len(freqs) - 1)])

            return ServiceResult.ok(
                data=MeasurementResult(measurements=measurements),
                message=f"Computed {len(measurements)} measurements",
            )

        except Exception as e:
            logger.exception(f"Failed to compute measurements: {e}")
            return ServiceResult.fail(f"Failed to compute measurements: {e}")

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def import_batch(
        self,
        directory: str,
        file_pattern: str = "*.txt",
        format: str = "raven",
    ) -> ServiceResult[AnnotationResult]:
        """
        Import annotations from all matching files in a directory.

        Args:
            directory: Path to directory containing annotation files
            file_pattern: Glob pattern for matching files
            format: Annotation format ('raven' or 'csv')

        Returns:
            ServiceResult containing combined AnnotationResult
        """
        error = self._validate_input_path(directory)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.annotations import load_annotations_from_directory

            all_annotations_dict = load_annotations_from_directory(
                directory, file_pattern=file_pattern, format=format
            )

            # Flatten all annotations
            all_annotations = []
            for _filename, annotations in all_annotations_dict.items():
                all_annotations.extend(annotations)

            summary = summarize_annotations(all_annotations)

            return ServiceResult.ok(
                data=AnnotationResult(
                    annotations=all_annotations, file_path=directory, summary=summary
                ),
                message=f"Imported {len(all_annotations)} annotations from {len(all_annotations_dict)} files",
                count=len(all_annotations),
                files=len(all_annotations_dict),
            )

        except Exception as e:
            logger.exception(f"Failed to batch import: {e}")
            return ServiceResult.fail(f"Failed to batch import: {e}")

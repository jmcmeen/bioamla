"""
Annotation System
=================

This module provides functionality for managing audio annotations including:
- CSV annotation import/export
- Raven Pro selection table import/export (.txt tab-delimited)
- Time-frequency box annotations (start_time, end_time, low_freq, high_freq)
- Annotation-to-clip label generation (one-hot encoding)
- Custom annotation fields support
- Label remapping/conversion utilities

Raven Pro Selection Table Format:
    Raven Pro uses tab-delimited text files with specific column headers.
    Standard columns include: Selection, View, Channel, Begin Time (s),
    End Time (s), Low Freq (Hz), High Freq (Hz), and optional annotation columns.

Example:
    >>> from bioamla.annotations import load_raven_selection_table
    >>> annotations = load_raven_selection_table("selections.txt")
    >>> print(annotations[0])
    Annotation(start_time=1.5, end_time=3.2, low_freq=1000, high_freq=8000, label='bird_song')
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from bioamla.core.files import TextFile

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
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


@dataclass
class AnnotationSet:
    """
    A collection of annotations for a single audio file.

    Attributes:
        file_path: Path to the associated audio file
        annotations: List of Annotation objects
        sample_rate: Sample rate of the audio file (optional)
        duration: Total duration of the audio file in seconds (optional)
        metadata: Additional metadata about the file or annotation set
    """

    file_path: str
    annotations: List[Annotation] = field(default_factory=list)
    sample_rate: Optional[int] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)

    def __getitem__(self, idx: int) -> Annotation:
        return self.annotations[idx]

    def add(self, annotation: Annotation) -> None:
        """Add an annotation to the set."""
        self.annotations.append(annotation)

    def get_labels(self) -> Set[str]:
        """Get all unique labels in this annotation set."""
        return {a.label for a in self.annotations if a.label}

    def filter_by_label(self, label: str) -> List[Annotation]:
        """Get all annotations with a specific label."""
        return [a for a in self.annotations if a.label == label]

    def filter_by_time_range(self, start: float, end: float) -> List[Annotation]:
        """Get all annotations that overlap with a time range."""
        return [a for a in self.annotations if not (a.end_time <= start or a.start_time >= end)]

    def filter_by_freq_range(self, low: float, high: float) -> List[Annotation]:
        """Get all annotations that overlap with a frequency range."""
        return [
            a
            for a in self.annotations
            if a.low_freq is None
            or a.high_freq is None
            or not (a.high_freq <= low or a.low_freq >= high)
        ]

    def sort_by_time(self) -> None:
        """Sort annotations by start time."""
        self.annotations.sort(key=lambda a: (a.start_time, a.end_time))

    def merge_overlapping(self, same_label_only: bool = True) -> "AnnotationSet":
        """
        Merge overlapping annotations.

        Args:
            same_label_only: If True, only merge annotations with the same label

        Returns:
            New AnnotationSet with merged annotations
        """
        if not self.annotations:
            return AnnotationSet(
                file_path=self.file_path,
                sample_rate=self.sample_rate,
                duration=self.duration,
                metadata=self.metadata.copy(),
            )

        sorted_annots = sorted(self.annotations, key=lambda a: a.start_time)
        merged = []

        current = Annotation(
            start_time=sorted_annots[0].start_time,
            end_time=sorted_annots[0].end_time,
            low_freq=sorted_annots[0].low_freq,
            high_freq=sorted_annots[0].high_freq,
            label=sorted_annots[0].label,
            channel=sorted_annots[0].channel,
        )

        for ann in sorted_annots[1:]:
            should_merge = current.overlaps_time(ann)
            if same_label_only:
                should_merge = should_merge and current.label == ann.label

            if should_merge:
                # Extend current annotation
                current.end_time = max(current.end_time, ann.end_time)
                if current.low_freq is not None and ann.low_freq is not None:
                    current.low_freq = min(current.low_freq, ann.low_freq)
                if current.high_freq is not None and ann.high_freq is not None:
                    current.high_freq = max(current.high_freq, ann.high_freq)
            else:
                merged.append(current)
                current = Annotation(
                    start_time=ann.start_time,
                    end_time=ann.end_time,
                    low_freq=ann.low_freq,
                    high_freq=ann.high_freq,
                    label=ann.label,
                    channel=ann.channel,
                )

        merged.append(current)

        return AnnotationSet(
            file_path=self.file_path,
            annotations=merged,
            sample_rate=self.sample_rate,
            duration=self.duration,
            metadata=self.metadata.copy(),
        )


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

# Reverse mapping for export
RAVEN_EXPORT_COLUMNS = {
    "selection": "Selection",
    "view": "View",
    "channel": "Channel",
    "start_time": "Begin Time (s)",
    "end_time": "End Time (s)",
    "low_freq": "Low Freq (Hz)",
    "high_freq": "High Freq (Hz)",
    "label": "Annotation",
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
# Label Generation and One-Hot Encoding
# =============================================================================


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


def create_label_map(labels: List[str]) -> Dict[str, int]:
    """
    Create a mapping from label strings to integer indices.

    Args:
        labels: List of label strings

    Returns:
        Dictionary mapping label strings to integer indices
    """
    return {label: idx for idx, label in enumerate(sorted(set(labels)))}


def annotations_to_one_hot(
    annotations: List[Annotation], label_map: Dict[str, int], num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Convert annotations to one-hot encoded labels.

    Args:
        annotations: List of Annotation objects
        label_map: Dictionary mapping labels to indices
        num_classes: Number of classes (if None, inferred from label_map)

    Returns:
        One-hot encoded numpy array of shape (num_annotations, num_classes)

    Example:
        >>> annotations = [Annotation(..., label="bird"), Annotation(..., label="frog")]
        >>> label_map = {"bird": 0, "frog": 1}
        >>> one_hot = annotations_to_one_hot(annotations, label_map)
        >>> print(one_hot)
        [[1, 0],
         [0, 1]]
    """
    if num_classes is None:
        num_classes = len(label_map)

    one_hot = np.zeros((len(annotations), num_classes), dtype=np.float32)

    for i, ann in enumerate(annotations):
        if ann.label in label_map:
            one_hot[i, label_map[ann.label]] = 1.0

    return one_hot


def generate_clip_labels(
    annotations: List[Annotation],
    clip_start: float,
    clip_end: float,
    label_map: Dict[str, int],
    min_overlap: float = 0.0,
    multi_label: bool = True,
) -> np.ndarray:
    """
    Generate labels for a clip based on overlapping annotations.

    This function determines which labels apply to a given time window
    based on annotation overlap.

    Args:
        annotations: List of annotations to check
        clip_start: Start time of the clip in seconds
        clip_end: End time of the clip in seconds
        label_map: Dictionary mapping labels to indices
        min_overlap: Minimum overlap ratio (0.0-1.0) required to assign label
        multi_label: If True, return multi-hot encoding. If False, return
            single label (most overlapping annotation wins)

    Returns:
        Label vector (one-hot or multi-hot encoded)

    Example:
        >>> # Generate labels for a 3-second clip starting at t=5
        >>> labels = generate_clip_labels(annotations, 5.0, 8.0, label_map)
    """
    num_classes = len(label_map)
    clip_duration = clip_end - clip_start

    if multi_label:
        labels = np.zeros(num_classes, dtype=np.float32)
    else:
        labels = np.zeros(num_classes, dtype=np.float32)
        best_overlap = 0.0
        best_label = None

    for ann in annotations:
        # Calculate overlap
        overlap_start = max(clip_start, ann.start_time)
        overlap_end = min(clip_end, ann.end_time)
        overlap = max(0, overlap_end - overlap_start)

        if overlap <= 0 or ann.label not in label_map:
            continue

        overlap_ratio = overlap / clip_duration

        if overlap_ratio >= min_overlap:
            if multi_label:
                labels[label_map[ann.label]] = 1.0
            elif overlap > best_overlap:
                best_overlap = overlap
                best_label = ann.label

    if not multi_label and best_label is not None:
        labels[label_map[best_label]] = 1.0

    return labels


def generate_frame_labels(
    annotations: List[Annotation],
    total_duration: float,
    frame_size: float,
    hop_length: float,
    label_map: Dict[str, int],
) -> np.ndarray:
    """
    Generate frame-level labels for audio.

    This creates a label matrix where each column represents a time frame.

    Args:
        annotations: List of annotations
        total_duration: Total audio duration in seconds
        frame_size: Frame size in seconds
        hop_length: Hop length in seconds
        label_map: Dictionary mapping labels to indices

    Returns:
        Label matrix of shape (num_classes, num_frames)
    """
    num_classes = len(label_map)
    num_frames = int((total_duration - frame_size) / hop_length) + 1
    labels = np.zeros((num_classes, num_frames), dtype=np.float32)

    for frame_idx in range(num_frames):
        frame_start = frame_idx * hop_length
        frame_end = frame_start + frame_size

        for ann in annotations:
            if ann.label not in label_map:
                continue

            # Check for overlap
            if ann.start_time < frame_end and ann.end_time > frame_start:
                labels[label_map[ann.label], frame_idx] = 1.0

    return labels


# =============================================================================
# Label Remapping and Conversion
# =============================================================================


def remap_labels(
    annotations: List[Annotation], label_mapping: Dict[str, str], keep_unmapped: bool = True
) -> List[Annotation]:
    """
    Remap annotation labels using a mapping dictionary.

    This is useful for consolidating similar labels, renaming categories,
    or mapping to a standard taxonomy.

    Args:
        annotations: List of annotations to remap
        label_mapping: Dictionary mapping old labels to new labels
        keep_unmapped: If True, keep annotations with unmapped labels unchanged.
            If False, remove annotations with unmapped labels.

    Returns:
        List of annotations with remapped labels

    Example:
        >>> mapping = {
        ...     "bird_song": "bird",
        ...     "bird_call": "bird",
        ...     "frog_croak": "frog"
        ... }
        >>> remapped = remap_labels(annotations, mapping)
    """
    result = []

    for ann in annotations:
        if ann.label in label_mapping:
            # Create new annotation with mapped label
            new_ann = Annotation(
                start_time=ann.start_time,
                end_time=ann.end_time,
                low_freq=ann.low_freq,
                high_freq=ann.high_freq,
                label=label_mapping[ann.label],
                channel=ann.channel,
                confidence=ann.confidence,
                notes=ann.notes,
                custom_fields=ann.custom_fields.copy(),
            )
            result.append(new_ann)
        elif keep_unmapped:
            result.append(ann)

    return result


def filter_labels(
    annotations: List[Annotation],
    include_labels: Optional[Set[str]] = None,
    exclude_labels: Optional[Set[str]] = None,
) -> List[Annotation]:
    """
    Filter annotations by label.

    Args:
        annotations: List of annotations to filter
        include_labels: If provided, only keep annotations with these labels
        exclude_labels: If provided, remove annotations with these labels

    Returns:
        Filtered list of annotations
    """
    result = annotations

    if include_labels is not None:
        result = [a for a in result if a.label in include_labels]

    if exclude_labels is not None:
        result = [a for a in result if a.label not in exclude_labels]

    return result


def load_label_mapping(filepath: str, encoding: str = "utf-8") -> Dict[str, str]:
    """
    Load a label mapping from a CSV file.

    The CSV should have two columns: 'source' and 'target'.

    Args:
        filepath: Path to the CSV file
        encoding: File encoding

    Returns:
        Dictionary mapping source labels to target labels
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {filepath}")

    mapping = {}

    with TextFile(path, mode="r", newline="", encoding=encoding) as f:
        reader = csv.DictReader(f.handle)
        for row in reader:
            source = row.get("source", "").strip()
            target = row.get("target", "").strip()
            if source and target:
                mapping[source] = target

    logger.info(f"Loaded {len(mapping)} label mappings from {filepath}")
    return mapping


def save_label_mapping(mapping: Dict[str, str], filepath: str, encoding: str = "utf-8") -> str:
    """
    Save a label mapping to a CSV file.

    Args:
        mapping: Dictionary mapping source labels to target labels
        filepath: Output file path
        encoding: File encoding

    Returns:
        Path to the saved file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with TextFile(path, mode="w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f.handle, fieldnames=["source", "target"])
        writer.writeheader()
        for source, target in sorted(mapping.items()):
            writer.writerow({"source": source, "target": target})

    logger.info(f"Saved {len(mapping)} label mappings to {filepath}")
    return str(path)


# =============================================================================
# Batch Operations
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

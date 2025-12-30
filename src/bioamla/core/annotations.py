"""
Annotation System
=================

This module provides functionality for managing audio annotations including:
- AnnotationSet for managing collections of annotations
- Label generation and one-hot encoding utilities
- Label remapping/conversion utilities

Core annotation I/O functions (load/save CSV/Raven) have been moved to
bioamla.services.annotation for a cleaner service-based architecture.

Example:
    >>> from bioamla.services.annotation import load_raven_selection_table
    >>> annotations = load_raven_selection_table("selections.txt")
    >>> print(annotations[0])
    Annotation(start_time=1.5, end_time=3.2, low_freq=1000, high_freq=8000, label='bird_song')
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import numpy as np

from bioamla.core.files import TextFile
from bioamla.services.annotation import Annotation

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


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

    def __iter__(self) -> "Iterator[Annotation]":
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

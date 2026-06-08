"""Annotation label utilities: collections, encoding, remapping, and mappings.

Holds :class:`AnnotationSet` (a collection of annotations for one audio file)
plus label-engineering helpers used to turn annotations into model targets
(label maps, one-hot/multi-hot encodings, frame/clip labels) and to remap or
filter labels and persist label-mapping CSVs.

NumPy is imported lazily inside the functions that need it, so a slim install
can still import this module.
"""

from __future__ import annotations

import copy
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from bioamla.datasets.annotations import Annotation
from bioamla.exceptions import InvalidInputError, NotFoundError

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AnnotationSet:
    """A collection of annotations for a single audio file.

    Attributes:
        file_path: Path to the associated audio file
        annotations: List of Annotation objects
        sample_rate: Sample rate of the audio file (optional)
        duration: Total duration of the audio file in seconds (optional)
        metadata: Additional metadata about the file or annotation set
    """

    file_path: str
    annotations: list[Annotation] = field(default_factory=list)
    sample_rate: int | None = None
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self) -> Iterator[Annotation]:
        return iter(self.annotations)

    def __getitem__(self, idx: int) -> Annotation:
        return self.annotations[idx]

    def add(self, annotation: Annotation) -> None:
        """Add an annotation to the set."""
        self.annotations.append(annotation)

    def get_labels(self) -> set[str]:
        """Get all unique labels in this annotation set."""
        return {a.label for a in self.annotations if a.label}

    def filter_by_label(self, label: str) -> list[Annotation]:
        """Get all annotations with a specific label."""
        return [a for a in self.annotations if a.label == label]

    def filter_by_time_range(self, start: float, end: float) -> list[Annotation]:
        """Get all annotations that overlap with a time range."""
        return [a for a in self.annotations if not (a.end_time <= start or a.start_time >= end)]

    def filter_by_freq_range(self, low: float, high: float) -> list[Annotation]:
        """Get all annotations that overlap with a frequency range."""
        return [
            a
            for a in self.annotations
            if a.low_freq is None
            or a.high_freq is None
            or not (a.high_freq <= low or a.low_freq >= high)
        ]

    def sort_by_time(self) -> None:
        """Sort annotations by start time (in place)."""
        self.annotations.sort(key=lambda a: (a.start_time, a.end_time))

    def merge_overlapping(self, same_label_only: bool = True) -> AnnotationSet:
        """Merge overlapping annotations into a new :class:`AnnotationSet`.

        Args:
            same_label_only: If True, only merge annotations with the same label.
        """
        if not self.annotations:
            return AnnotationSet(
                file_path=self.file_path,
                sample_rate=self.sample_rate,
                duration=self.duration,
                metadata=self.metadata.copy(),
            )

        sorted_annots = sorted(self.annotations, key=lambda a: a.start_time)
        merged: list[Annotation] = []

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


def create_label_map(labels: list[str]) -> dict[str, int]:
    """Create a mapping from label strings to integer indices (sorted, unique)."""
    return {label: idx for idx, label in enumerate(sorted(set(labels)))}


def annotations_to_one_hot(
    annotations: list[Annotation], label_map: dict[str, int], num_classes: int | None = None
) -> np.ndarray:
    """Convert annotations to one-hot encoded labels.

    Returns a numpy array of shape ``(num_annotations, num_classes)``.
    """
    import numpy as np

    if num_classes is None:
        num_classes = len(label_map)

    one_hot = np.zeros((len(annotations), num_classes), dtype=np.float32)
    for i, ann in enumerate(annotations):
        if ann.label in label_map:
            one_hot[i, label_map[ann.label]] = 1.0
    return one_hot


def generate_clip_labels(
    annotations: list[Annotation],
    clip_start: float,
    clip_end: float,
    label_map: dict[str, int],
    min_overlap: float = 0.0,
    multi_label: bool = True,
) -> np.ndarray:
    """Generate a label vector for a clip based on overlapping annotations.

    Args:
        annotations: List of annotations to check
        clip_start: Start time of the clip in seconds
        clip_end: End time of the clip in seconds
        label_map: Dictionary mapping labels to indices
        min_overlap: Minimum overlap ratio (0.0-1.0) required to assign a label
        multi_label: If True, return multi-hot encoding; otherwise the single
            most-overlapping label wins.

    Returns:
        Label vector (one-hot or multi-hot encoded).

    Raises:
        InvalidInputError: If ``clip_end <= clip_start``.
    """
    import numpy as np

    if clip_end <= clip_start:
        raise InvalidInputError("clip_end must be greater than clip_start")

    num_classes = len(label_map)
    clip_duration = clip_end - clip_start

    labels = np.zeros(num_classes, dtype=np.float32)
    best_overlap = 0.0
    best_label: str | None = None

    for ann in annotations:
        overlap_start = max(clip_start, ann.start_time)
        overlap_end = min(clip_end, ann.end_time)
        overlap = max(0.0, overlap_end - overlap_start)

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
    annotations: list[Annotation],
    total_duration: float,
    frame_size: float,
    hop_length: float,
    label_map: dict[str, int],
) -> np.ndarray:
    """Generate frame-level labels of shape ``(num_classes, num_frames)``.

    Args:
        annotations: List of annotations
        total_duration: Total audio duration in seconds
        frame_size: Frame size in seconds (must be positive)
        hop_length: Hop length in seconds (must be positive)
        label_map: Dictionary mapping labels to indices

    Raises:
        InvalidInputError: If frame_size or hop_length is non-positive.
    """
    import numpy as np

    if frame_size <= 0:
        raise InvalidInputError("frame_size must be positive")
    if hop_length <= 0:
        raise InvalidInputError("hop_length must be positive")

    num_classes = len(label_map)
    num_frames = max(0, int((total_duration - frame_size) / hop_length) + 1)
    labels = np.zeros((num_classes, num_frames), dtype=np.float32)

    for frame_idx in range(num_frames):
        frame_start = frame_idx * hop_length
        frame_end = frame_start + frame_size

        for ann in annotations:
            if ann.label not in label_map:
                continue
            if ann.start_time < frame_end and ann.end_time > frame_start:
                labels[label_map[ann.label], frame_idx] = 1.0

    return labels


# =============================================================================
# Label Remapping and Conversion
# =============================================================================


def remap_labels(
    annotations: list[Annotation], label_mapping: dict[str, str], keep_unmapped: bool = True
) -> list[Annotation]:
    """Remap annotation labels using a mapping dictionary.

    Args:
        annotations: List of annotations to remap
        label_mapping: Dictionary mapping old labels to new labels
        keep_unmapped: If True, keep annotations with unmapped labels unchanged;
            otherwise drop them.

    Returns:
        New list of annotations with remapped labels.
    """
    result: list[Annotation] = []
    for ann in annotations:
        if ann.label in label_mapping:
            new_ann = copy.copy(ann)
            new_ann.label = label_mapping[ann.label]
            new_ann.custom_fields = ann.custom_fields.copy()
            result.append(new_ann)
        elif keep_unmapped:
            result.append(ann)
    return result


def filter_labels(
    annotations: list[Annotation],
    include_labels: set[str] | None = None,
    exclude_labels: set[str] | None = None,
) -> list[Annotation]:
    """Filter annotations by include/exclude label sets."""
    result = annotations
    if include_labels is not None:
        result = [a for a in result if a.label in include_labels]
    if exclude_labels is not None:
        result = [a for a in result if a.label not in exclude_labels]
    return result


def load_label_mapping(filepath: str, encoding: str = "utf-8") -> dict[str, str]:
    """Load a label mapping (columns ``source``, ``target``) from a CSV file.

    Raises:
        NotFoundError: If the file doesn't exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Mapping file not found: {filepath}")

    mapping: dict[str, str] = {}
    with open(path, newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = (row.get("source") or "").strip()
            target = (row.get("target") or "").strip()
            if source and target:
                mapping[source] = target

    logger.info(f"Loaded {len(mapping)} label mappings from {filepath}")
    return mapping


def save_label_mapping(mapping: dict[str, str], filepath: str, encoding: str = "utf-8") -> str:
    """Save a label mapping to a CSV file with columns ``source``, ``target``."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, mode="w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target"])
        writer.writeheader()
        for source, target in sorted(mapping.items()):
            writer.writerow({"source": source, "target": target})

    logger.info(f"Saved {len(mapping)} label mappings to {filepath}")
    return str(path)

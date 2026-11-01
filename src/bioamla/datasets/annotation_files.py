"""Annotation files: create, convert, summarize, rewrite and turn into training labels.

These are the file-level calls behind the ``bioamla annotation`` commands. Each takes
paths, reads the annotation file in whatever format it is (raven, csv, bioamla
``.json``, arbimon; detected from the name or header unless given) and returns a
result with ``to_dict()``.

Example:
    >>> from bioamla.datasets import clip_labels_file, convert_annotation_file
    >>> convert_annotation_file("rec.Table.1.selections.txt", "rec.json")
    >>> clip_labels_file("rec.json", "labels.csv", clip_duration=3.0).clips
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bioamla.datasets._io import (
    annotations_for_recording,
    detect_annotation_format,
    load_annotations,
    recordings_named,
    save_annotations,
)
from bioamla.datasets.annotation_utils import (
    create_label_map,
    filter_labels,
    generate_clip_labels,
    generate_frame_labels,
    load_label_mapping,
    remap_labels,
)
from bioamla.datasets.annotations import Annotation, get_unique_labels, summarize_annotations
from bioamla.exceptions import AnnotationError, InvalidInputError, OptionConflictError


@dataclass
class AnnotationTemplate:
    """A starter annotation file written for a recording.

    Attributes:
        audio_file: The recording.
        output: The annotation file written.
        format: Its format.
        rows: Rows written (one placeholder spanning the file, or none).
        duration: The recording's duration (s).
        sample_rate: Its sample rate (Hz).
        channels: Its channel count.
    """

    audio_file: str
    output: str
    format: str
    rows: int
    duration: float
    sample_rate: int
    channels: int

    def to_dict(self) -> dict[str, Any]:
        """``{"audio_file", "output", "format", "rows", "audio": {...}}``."""
        return {
            "audio_file": self.audio_file,
            "output": self.output,
            "format": self.format,
            "rows": self.rows,
            "audio": {
                "duration": self.duration,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
            },
        }


def annotation_template(
    audio_file: str | Path,
    output_file: str | Path,
    *,
    format: str | None = None,
    label: str = "",
    empty: bool = False,
) -> AnnotationTemplate:
    """Write a starter annotation file for a recording.

    The file holds the recording's metadata (only the bioamla ``.json`` format keeps
    it) and, unless ``empty``, one placeholder row spanning the whole file.

    Args:
        audio_file: The recording (its header is read; it isn't decoded).
        output_file: The annotation file to write.
        format: Its format; by default from its name.
        label: Label of the placeholder row.
        empty: Write no placeholder row.

    Raises:
        AudioLoadError / NotFoundError: If the recording can't be read.
        AnnotationError: If the file can't be written.
    """
    from bioamla.audio import get_audio_info

    output_path = Path(output_file)
    fmt = detect_annotation_format(output_path, format)
    info = get_audio_info(str(audio_file))
    annotations = [] if empty else [Annotation(start_time=0.0, end_time=info.duration, label=label)]
    metadata = {
        "audio_file": Path(audio_file).name,
        "sample_rate": info.sample_rate,
        "duration": round(info.duration, 6),
        "channels": info.channels,
        "created": datetime.now(UTC).isoformat(),
    }
    save_annotations(annotations, output_path, fmt, metadata=metadata)
    return AnnotationTemplate(
        audio_file=str(audio_file),
        output=str(output_file),
        format=fmt,
        rows=len(annotations),
        duration=info.duration,
        sample_rate=info.sample_rate,
        channels=info.channels,
    )


@dataclass
class ConvertedAnnotations:
    """An annotation file converted to another format.

    Attributes:
        input: The source file.
        output: The file written.
        from_format: The source format.
        to_format: The output format.
        count: Annotations converted.
    """

    input: str
    output: str
    from_format: str
    to_format: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        """``{"input", "output", "from_format", "to_format", "count"}``."""
        return {
            "input": self.input,
            "output": self.output,
            "from_format": self.from_format,
            "to_format": self.to_format,
            "count": self.count,
        }


def convert_annotation_file(
    input_file: str | Path,
    output_file: str | Path,
    *,
    from_format: str | None = None,
    to_format: str | None = None,
    label_column: str | None = None,
    recording: str | None = None,
) -> ConvertedAnnotations:
    """Convert an annotation file between raven, csv, bioamla (``.json``) and arbimon.

    Args:
        input_file: The source file.
        output_file: The file to write.
        from_format: Source format; by default from its name or header (an arbimon
            CSV is recognized from its header).
        to_format: Output format; by default from its name (arbimon must be named).
        label_column: The source column holding labels.
        recording: arbimon output only: the recording name for rows that don't carry
            one (default: the ``audio_file`` of a bioamla ``.json`` input).

    Raises:
        OptionConflictError: If ``recording`` is given for a non-arbimon output.
        AnnotationError: If the file can't be read or written.
    """
    input_path, output_path = Path(input_file), Path(output_file)
    from_format = detect_annotation_format(input_path, from_format, read_header=True)
    to_format = detect_annotation_format(output_path, to_format)
    if recording is not None and to_format != "arbimon":
        raise OptionConflictError("--recording only applies to --to arbimon")
    annotations, metadata = load_annotations(input_path, from_format, label_column=label_column)
    save_annotations(annotations, output_path, to_format, metadata=metadata, recording=recording)
    return ConvertedAnnotations(
        input=str(input_file),
        output=str(output_file),
        from_format=from_format,
        to_format=to_format,
        count=len(annotations),
    )


@dataclass
class AnnotationFileSummary:
    """Summary statistics of an annotation file.

    Attributes:
        file: The annotation file.
        summary: Counts and durations (see
            :func:`~bioamla.datasets.summarize_annotations`).
        metadata: The file's own metadata (a bioamla ``.json``'s ``audio_file``,
            ``duration``, ...).
    """

    file: str
    summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The summary itself (what ``annotation summary --json`` reports)."""
        return self.summary


def summarize_annotation_file(
    path: str | Path, *, format: str | None = None
) -> AnnotationFileSummary:
    """Summarize an annotation file: counts, label counts, durations.

    Raises:
        AnnotationError: If the file can't be read.
    """
    file_path = Path(path)
    fmt = detect_annotation_format(file_path, format, read_header=True)
    annotations, metadata = load_annotations(file_path, fmt)
    return AnnotationFileSummary(
        file=str(path), summary=summarize_annotations(annotations), metadata=metadata
    )


@dataclass
class RewrittenAnnotations:
    """An annotation file rewritten with some rows changed or left out.

    Attributes:
        input: The source file.
        output: The file written.
        input_count: Rows read.
        output_count: Rows written.
    """

    input: str
    output: str
    input_count: int
    output_count: int

    def to_dict(self) -> dict[str, Any]:
        """``{"input", "output", "input_count", "output_count"}``."""
        return {
            "input": self.input,
            "output": self.output,
            "input_count": self.input_count,
            "output_count": self.output_count,
        }


def _rewrite_formats(input_path: Path, output_path: Path) -> tuple[str, str]:
    """The input's format, and the output's: the input's when the extension is the same
    (or missing), so an arbimon CSV stays one."""
    in_format = detect_annotation_format(input_path, read_header=True)
    same = not output_path.suffix or output_path.suffix.lower() == input_path.suffix.lower()
    return in_format, in_format if same else detect_annotation_format(output_path)


def remap_annotation_file(
    input_file: str | Path,
    output_file: str | Path,
    mapping_file: str | Path,
    *,
    keep_unmapped: bool = True,
) -> RewrittenAnnotations:
    """Rename labels with a mapping CSV (columns ``source``, ``target``) and save.

    Args:
        input_file: The source file.
        output_file: The file to write (same format as the input unless its
            extension says otherwise).
        mapping_file: The mapping CSV.
        keep_unmapped: Keep rows whose label isn't in the mapping (else drop them).

    Raises:
        AnnotationError: If a file can't be read or written.
    """
    input_path, output_path = Path(input_file), Path(output_file)
    in_format, out_format = _rewrite_formats(input_path, output_path)
    label_mapping = load_label_mapping(str(mapping_file))
    annotations, metadata = load_annotations(input_path, in_format)
    remapped = remap_labels(annotations, label_mapping, keep_unmapped=keep_unmapped)
    save_annotations(remapped, output_path, out_format, metadata=metadata)
    return RewrittenAnnotations(str(input_file), str(output_file), len(annotations), len(remapped))


def filter_annotation_file(
    input_file: str | Path,
    output_file: str | Path,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> RewrittenAnnotations:
    """Keep the rows that pass label and duration filters, and save them.

    Args:
        input_file: The source file.
        output_file: The file to write (same format as the input unless its
            extension says otherwise).
        include: Keep only these labels.
        exclude: Drop these labels.
        min_duration: Drop rows shorter than this (s).
        max_duration: Drop rows longer than this (s).

    Raises:
        AnnotationError: If a file can't be read or written.
    """
    input_path, output_path = Path(input_file), Path(output_file)
    in_format, out_format = _rewrite_formats(input_path, output_path)
    annotations, metadata = load_annotations(input_path, in_format)
    kept = filter_labels(
        annotations,
        include_labels=set(include) if include else None,
        exclude_labels=set(exclude) if exclude else None,
    )
    if min_duration is not None:
        kept = [a for a in kept if a.duration >= min_duration]
    if max_duration is not None:
        kept = [a for a in kept if a.duration <= max_duration]
    save_annotations(kept, output_path, out_format, metadata=metadata)
    return RewrittenAnnotations(str(input_file), str(output_file), len(annotations), len(kept))


@dataclass
class GeneratedLabels:
    """Training labels made from an annotation file.

    Attributes:
        annotation_file: The annotation file.
        output: The labels file written (``.csv``, or ``.npy``).
        label_map: The ``.labels.csv`` label -> column map written beside a ``.npy``.
        labels: The labels, in column order.
        clips: Clips labelled (clip labels).
        classes: Classes (frame labels).
        frames: Frames labelled (frame labels).
    """

    annotation_file: str
    output: str
    label_map: str | None
    labels: list[str]
    clips: int | None = None
    classes: int | None = None
    frames: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """``{"annotation_file", "output", "label_map", ..., "labels"}`` with ``clips``,
        or ``classes`` and ``frames``."""
        result: dict[str, Any] = {
            "annotation_file": self.annotation_file,
            "output": self.output,
            "label_map": self.label_map,
        }
        if self.clips is not None:
            result["clips"] = self.clips
        if self.classes is not None:
            result["classes"] = self.classes
            result["frames"] = self.frames
        result["labels"] = sorted(self.labels)
        return result


def _one_recording(annotations: list[Annotation], recording: str | None, source: str) -> list:
    """The rows for one recording: ``recording``, or the file's only one."""
    names = recordings_named(annotations)
    if recording is not None:
        kept = annotations_for_recording(annotations, recording)
        if names and not kept:
            raise InvalidInputError(f"No rows in {source} are for recording {recording!r}")
        return kept
    if len(names) > 1:
        shown = ", ".join(sorted(names)[:3]) + (", ..." if len(names) > 3 else "")
        raise InvalidInputError(
            f"{source} covers {len(names)} recordings ({shown}); "
            "pick one with recording (--recording)"
        )
    return annotations


def _labelled_rows(
    annotation_file: str | Path, recording: str | None, audio_duration: float | None
) -> tuple[list[Annotation], float, dict[str, int]]:
    path = Path(annotation_file)
    annotations, metadata = load_annotations(path, detect_annotation_format(path, read_header=True))
    annotations = _one_recording(annotations, recording, str(annotation_file))
    if not annotations:
        raise InvalidInputError("No annotations found in file")
    if audio_duration is None:
        if metadata.get("duration") is None:
            raise InvalidInputError(
                "audio_duration (--audio-duration) is required (no duration metadata in this file)"
            )
        audio_duration = float(metadata["duration"])
    return annotations, audio_duration, create_label_map(get_unique_labels(annotations))


def _write_labels(
    output_file: str | Path,
    output_format: str,
    array: Any,
    label_map: dict[str, int],
    header_prefix: list[str],
    rows: Any,
) -> tuple[Path, Path | None]:
    """``numpy``: the array as ``<name>.npy`` (``.npy`` added unless ``output_file`` has
    it) plus ``<name>.labels.csv``, named after it; ``csv``: one table."""
    import csv

    import numpy as np

    output_path = Path(output_file)
    ordered = sorted(label_map, key=lambda label: label_map[label])
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "numpy":
            data_path = output_path if output_path.suffix == ".npy" else Path(f"{output_path}.npy")
            np.save(data_path, array)
            # Named after the data file, so outputs that differ only before their last
            # dot (run.v1, run.v2) don't share a map.
            label_map_path = data_path.with_name(data_path.name[: -len(".npy")] + ".labels.csv")
            with open(label_map_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["label", "index"])
                writer.writerows([label, label_map[label]] for label in ordered)
            return data_path, label_map_path
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([*header_prefix, *ordered])
            writer.writerows(rows)
        return output_path, None
    except OSError as e:
        raise AnnotationError(f"Failed to write {output_file}: {e}") from e


def clip_labels_file(
    annotation_file: str | Path,
    output_file: str | Path,
    *,
    clip_duration: float,
    audio_duration: float | None = None,
    hop_length: float | None = None,
    min_overlap: float = 0.0,
    multi_label: bool = True,
    output_format: str = "csv",
    recording: str | None = None,
) -> GeneratedLabels:
    """Label fixed-length clips of a recording from its annotations, and save them.

    Clips start every ``hop_length`` seconds (default ``clip_duration``); a label is
    set where an annotation covers at least ``min_overlap`` of the clip. ``csv``
    writes ``clip_start, clip_end, <label>...``; ``numpy`` writes the array and a
    ``.labels.csv`` beside it.

    Args:
        annotation_file: The annotation file.
        output_file: The labels file to write.
        clip_duration: Clip length (s).
        audio_duration: The recording's length (s); by default from a bioamla
            ``.json`` file's metadata.
        hop_length: Step between clip starts (s).
        min_overlap: Overlap ratio needed to set a label (0–1).
        multi_label: Several labels per clip (else the best one).
        output_format: ``csv`` or ``numpy``.
        recording: For a file covering several recordings: the one to label.

    Raises:
        InvalidInputError: If there are no annotations, no duration, a non-positive
            clip or hop, a clip longer than the audio, or an ambiguous recording.
        AnnotationError: If the labels can't be written.
    """
    import numpy as np

    annotations, audio_duration, label_map = _labelled_rows(
        annotation_file, recording, audio_duration
    )
    hop = clip_duration if hop_length is None else hop_length
    if clip_duration <= 0 or hop <= 0:
        raise InvalidInputError(
            "clip_duration (--clip-duration) and hop_length (--hop-length) must be positive"
        )
    if clip_duration > audio_duration:
        raise InvalidInputError(
            f"clip_duration (--clip-duration, {clip_duration}s) is longer than the audio "
            f"({audio_duration}s)"
        )
    # The small epsilon keeps float error from dropping the last clip
    # (e.g. 0.2 / 0.1 == 1.9999999999999998).
    num_clips = math.floor((audio_duration - clip_duration) / hop + 1e-9) + 1
    labels = np.array(
        [
            generate_clip_labels(
                annotations,
                i * hop,
                i * hop + clip_duration,
                label_map,
                min_overlap=min_overlap,
                multi_label=multi_label,
            )
            for i in range(num_clips)
        ]
    )
    data_path, label_map_path = _write_labels(
        output_file,
        output_format,
        labels,
        label_map,
        ["clip_start", "clip_end"],
        (
            [f"{i * hop:.3f}", f"{i * hop + clip_duration:.3f}"] + [int(v) for v in row]
            for i, row in enumerate(labels)
        ),
    )
    return GeneratedLabels(
        annotation_file=str(annotation_file),
        output=str(data_path),
        label_map=str(label_map_path) if label_map_path else None,
        labels=list(label_map),
        clips=num_clips,
    )


def frame_labels_file(
    annotation_file: str | Path,
    output_file: str | Path,
    *,
    frame_size: float,
    hop_length: float | None = None,
    audio_duration: float | None = None,
    output_format: str = "csv",
    recording: str | None = None,
) -> GeneratedLabels:
    """Frame-level multi-hot labels of a recording (for sound-event detection models).

    ``csv`` writes ``frame_start, frame_end, <label>...``; ``numpy`` writes the
    ``(classes, frames)`` array and a ``.labels.csv`` beside it.

    Args:
        annotation_file: The annotation file.
        output_file: The labels file to write.
        frame_size: Frame length (s).
        hop_length: Step between frames (s); default ``frame_size``.
        audio_duration: The recording's length (s); by default from a bioamla
            ``.json`` file's metadata.
        output_format: ``csv`` or ``numpy``.
        recording: For a file covering several recordings: the one to label.

    Raises:
        InvalidInputError: If there are no annotations, no duration, a non-positive
            frame or hop, a frame longer than the audio, or an ambiguous recording.
        AnnotationError: If the labels can't be written.
    """
    annotations, audio_duration, label_map = _labelled_rows(
        annotation_file, recording, audio_duration
    )
    hop = frame_size if hop_length is None else hop_length
    if frame_size <= 0 or hop <= 0:
        raise InvalidInputError(
            "frame_size (--frame-size) and hop_length (--hop-length) must be positive"
        )
    if frame_size > audio_duration:
        raise InvalidInputError(
            f"frame_size (--frame-size, {frame_size}s) is longer than the audio ({audio_duration}s)"
        )
    labels = generate_frame_labels(annotations, audio_duration, frame_size, hop, label_map)
    num_classes, num_frames = labels.shape
    data_path, label_map_path = _write_labels(
        output_file,
        output_format,
        labels,
        label_map,
        ["frame_start", "frame_end"],
        (
            [f"{i * hop:.3f}", f"{i * hop + frame_size:.3f}"]
            + [int(labels[c, i]) for c in range(num_classes)]
            for i in range(num_frames)
        ),
    )
    return GeneratedLabels(
        annotation_file=str(annotation_file),
        output=str(data_path),
        label_map=str(label_map_path) if label_map_path else None,
        labels=list(label_map),
        classes=num_classes,
        frames=num_frames,
    )


__all__ = [
    "AnnotationFileSummary",
    "AnnotationTemplate",
    "ConvertedAnnotations",
    "GeneratedLabels",
    "RewrittenAnnotations",
    "annotation_template",
    "clip_labels_file",
    "convert_annotation_file",
    "filter_annotation_file",
    "frame_labels_file",
    "remap_annotation_file",
    "summarize_annotation_file",
]

"""Operations for ``annotation``: thin adapters over :mod:`bioamla.datasets.annotation_files`.

See :mod:`bioamla.operations`. Reading and rewriting annotation files is ``quick``;
generating labels is a ``job``, since its size grows with the recording's length over
the clip or frame step.
"""

from __future__ import annotations

from typing import Literal

from bioamla.common.config import config_dataclass
from bioamla.datasets.annotation_files import (
    AnnotationFileSummary,
    AnnotationTemplate,
    ConvertedAnnotations,
    GeneratedLabels,
    RewrittenAnnotations,
    annotation_template,
    clip_labels_file,
    convert_annotation_file,
    filter_annotation_file,
    frame_labels_file,
    remap_annotation_file,
    summarize_annotation_file,
)
from bioamla.operations import OperationContext, PathStr, operation

AnnotationFormat = Literal["raven", "csv", "bioamla", "arbimon"]


@config_dataclass
class TemplateRequest:
    """A starter annotation file for a recording.

    Attributes:
        audio_file: The recording.
        output: The annotation file to write.
        format: Its format; by default from its name.
        label: Label of the placeholder row.
        empty: Write no placeholder row.
    """

    audio_file: PathStr
    output: PathStr
    format: AnnotationFormat | None = None
    label: str = ""
    empty: bool = False


@operation("annotation.template", request=TemplateRequest)
def template_op(request: TemplateRequest, context: OperationContext) -> AnnotationTemplate:
    """Write a starter annotation file for a recording."""
    return annotation_template(
        request.audio_file,
        request.output,
        format=request.format,
        label=request.label,
        empty=request.empty,
    )


@config_dataclass
class ConvertRequest:
    """Convert an annotation file between formats.

    Attributes:
        input: The source file.
        output: The file to write.
        from_format: Source format; by default from its name or header.
        to_format: Output format; by default from its name.
        label_column: The source column holding labels.
        recording: arbimon output only: the recording name for rows without one.
    """

    input: PathStr
    output: PathStr
    from_format: AnnotationFormat | None = None
    to_format: AnnotationFormat | None = None
    label_column: str | None = None
    recording: str | None = None


@operation("annotation.convert", request=ConvertRequest)
def convert_op(request: ConvertRequest, context: OperationContext) -> ConvertedAnnotations:
    """Convert an annotation file between formats."""
    return convert_annotation_file(
        request.input,
        request.output,
        from_format=request.from_format,
        to_format=request.to_format,
        label_column=request.label_column,
        recording=request.recording,
    )


@config_dataclass
class SummaryRequest:
    """Summary statistics of an annotation file.

    Attributes:
        file: The annotation file.
        format: Its format; by default from its name or header.
    """

    file: PathStr
    format: AnnotationFormat | None = None


@operation("annotation.summary", request=SummaryRequest)
def summary_op(request: SummaryRequest, context: OperationContext) -> AnnotationFileSummary:
    """Summarize an annotation file."""
    return summarize_annotation_file(request.file, format=request.format)


@config_dataclass
class RemapRequest:
    """Rename labels with a mapping CSV (``source``, ``target``).

    Attributes:
        input: The source file.
        output: The file to write.
        mapping: The mapping CSV.
        keep_unmapped: Keep rows whose label isn't mapped.
    """

    input: PathStr
    output: PathStr
    mapping: PathStr
    keep_unmapped: bool = True


@operation("annotation.remap", request=RemapRequest)
def remap_op(request: RemapRequest, context: OperationContext) -> RewrittenAnnotations:
    """Rename labels with a mapping CSV."""
    return remap_annotation_file(
        request.input, request.output, request.mapping, keep_unmapped=request.keep_unmapped
    )


@config_dataclass
class FilterRequest:
    """Keep the rows that pass label and duration filters.

    Attributes:
        input: The source file.
        output: The file to write.
        include: Keep only these labels.
        exclude: Drop these labels.
        min_duration: Drop rows shorter than this (s).
        max_duration: Drop rows longer than this (s).
    """

    input: PathStr
    output: PathStr
    include: list[str] | None = None
    exclude: list[str] | None = None
    min_duration: float | None = None
    max_duration: float | None = None


@operation("annotation.filter", request=FilterRequest)
def filter_op(request: FilterRequest, context: OperationContext) -> RewrittenAnnotations:
    """Filter annotations by label or duration."""
    return filter_annotation_file(
        request.input,
        request.output,
        include=request.include,
        exclude=request.exclude,
        min_duration=request.min_duration,
        max_duration=request.max_duration,
    )


@config_dataclass
class ClipLabelsRequest:
    """Clip-level labels from an annotation file.

    Attributes:
        annotation_file: The annotation file.
        output: The labels file to write.
        clip_duration: Clip length (s).
        audio_duration: The recording's length (s); by default from the file.
        hop_length: Step between clips (s); default the clip length.
        min_overlap: Overlap ratio needed to set a label (0–1).
        multi_label: Several labels per clip.
        output_format: ``csv`` or ``numpy``.
        recording: For a file covering several recordings: the one to label.
    """

    annotation_file: PathStr
    output: PathStr
    clip_duration: float
    audio_duration: float | None = None
    hop_length: float | None = None
    min_overlap: float = 0.0
    multi_label: bool = True
    output_format: Literal["csv", "numpy"] = "csv"
    recording: str | None = None


@operation("annotation.generate-labels", request=ClipLabelsRequest, kind="job")
def generate_labels_op(request: ClipLabelsRequest, context: OperationContext) -> GeneratedLabels:
    """Generate clip-level labels from annotations."""
    return clip_labels_file(
        request.annotation_file,
        request.output,
        clip_duration=request.clip_duration,
        audio_duration=request.audio_duration,
        hop_length=request.hop_length,
        min_overlap=request.min_overlap,
        multi_label=request.multi_label,
        output_format=request.output_format,
        recording=request.recording,
    )


@config_dataclass
class FrameLabelsRequest:
    """Frame-level multi-hot labels from an annotation file.

    Attributes:
        annotation_file: The annotation file.
        output: The labels file to write.
        frame_size: Frame length (s).
        hop_length: Step between frames (s); default the frame size.
        audio_duration: The recording's length (s); by default from the file.
        output_format: ``csv`` or ``numpy``.
        recording: For a file covering several recordings: the one to label.
    """

    annotation_file: PathStr
    output: PathStr
    frame_size: float
    hop_length: float | None = None
    audio_duration: float | None = None
    output_format: Literal["csv", "numpy"] = "csv"
    recording: str | None = None


@operation("annotation.generate-frame-labels", request=FrameLabelsRequest, kind="job")
def generate_frame_labels_op(
    request: FrameLabelsRequest, context: OperationContext
) -> GeneratedLabels:
    """Generate frame-level multi-hot labels from annotations."""
    return frame_labels_file(
        request.annotation_file,
        request.output,
        frame_size=request.frame_size,
        hop_length=request.hop_length,
        audio_duration=request.audio_duration,
        output_format=request.output_format,
        recording=request.recording,
    )

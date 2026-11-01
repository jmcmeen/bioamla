"""Annotation management commands for audio datasets."""

from pathlib import Path

import click

from bioamla.common.constants import ANNOTATION_FORMATS as _FORMAT_CHOICES


@click.group()
def annotation() -> None:
    """Annotation management commands for audio datasets."""
    pass


@annotation.command("template")
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file")
@click.option(
    "--format",
    "out_format",
    type=click.Choice(_FORMAT_CHOICES),
    default=None,
    help="Output format (auto-detected from extension if not specified)",
)
@click.option("--label", default="", help="Label for the placeholder full-file row")
@click.option("--empty", is_flag=True, help="Write metadata only, with no placeholder row")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def annotation_template(
    audio_file: str,
    output_file: str,
    out_format: str,
    label: str,
    empty: bool,
    quiet: bool,
) -> None:
    """Generate a starter annotation file from an audio file.

    Reads the recording's duration and sample rate and writes a skeleton
    annotation file pre-filled with that metadata plus (unless --empty) a single
    placeholder row spanning the whole file, ready to edit. The bioamla (.json)
    format stores the audio metadata in the file; raven/csv hold the rows only.
    """
    from bioamla.cli.console import echo, print_kv, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    made = run_operation(
        "annotation.template",
        audio_file=audio_file,
        output=output_file,
        format=out_format,
        label=label,
        empty=empty,
    )
    if emit_result(made.to_dict()):
        return

    if not quiet:
        print_success(f"Created {made.format} annotation template: {output_file}")
        echo(
            f"  audio: {Path(audio_file).name}  duration: {made.duration:.2f}s  "
            f"sr: {made.sample_rate} Hz"
        )
        print_kv("  rows", made.rows)
        if made.format != "bioamla":
            echo("  note: audio metadata is only persisted in the bioamla (.json) format")


@annotation.command("convert")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file")
@click.option(
    "--from",
    "from_format",
    type=click.Choice(_FORMAT_CHOICES),
    default=None,
    help="Input format (auto-detected from extension if not specified)",
)
@click.option(
    "--to",
    "to_format",
    type=click.Choice(_FORMAT_CHOICES),
    default=None,
    help="Output format (auto-detected from extension if not specified)",
)
@click.option("--label-column", default=None, help="Column name for labels in input file")
@click.option(
    "--recording",
    default=None,
    help="arbimon output: recording name for rows that don't carry one "
    "(default: the audio_file of a bioamla .json input)",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def annotation_convert(
    input_file: str,
    output_file: str,
    from_format: str,
    to_format: str,
    label_column: str,
    recording: str | None,
    quiet: bool,
) -> None:
    """Convert annotation files between formats.

    Formats: raven (.txt selection table), csv, bioamla (.json), and arbimon (the
    CSV layout of Arbimon's pattern-matching ROI export: x1/x2 seconds, y1/y2 Hz,
    species, songtype, recording, ...). An arbimon input is recognized from its
    header; for arbimon output pass --to arbimon.

    \b
    Examples:
        bioamla annotation convert rec.Table.1.selections.txt rois.csv --to arbimon --recording rec.wav
        bioamla annotation convert pattern_matching_rois.csv labels.txt
    """
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    converted = run_operation(
        "annotation.convert",
        input=input_file,
        output=output_file,
        from_format=from_format,
        to_format=to_format,
        label_column=label_column,
        recording=recording,
    )
    if emit_result(converted.to_dict()):
        return

    if not quiet:
        print_success(
            f"Converted {converted.count} annotations from {converted.from_format} "
            f"to {converted.to_format}"
        )
        print_success(f"Output: {output_file}")


@annotation.command("summary")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--format",
    "file_format",
    type=click.Choice(_FORMAT_CHOICES),
    default=None,
    help="Annotation format (auto-detected from extension if not specified)",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Deprecated: bare JSON. Use `bioamla --json annotation summary` instead.",
)
def annotation_summary(path: str, file_format: str, output_json: bool) -> None:
    """Display summary statistics for an annotation file."""
    from bioamla.cli.console import echo, print_header, print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result, json_mode, legacy_json

    result = run_operation("annotation.summary", file=path, format=file_format)
    summary, metadata = result.summary, result.metadata

    if json_mode():
        emit_result(summary)
    elif output_json:
        legacy_json(summary)
    else:
        print_header(f"\nAnnotation Summary: {path}")
        echo("=" * 50)
        if metadata.get("audio_file"):
            print_kv("Audio file", metadata["audio_file"])
        if metadata.get("duration") is not None:
            print_kv("Audio duration", f"{float(metadata['duration']):.2f}s")
        print_kv("Total annotations", summary["total_annotations"])
        print_kv("Unique labels", summary["unique_labels"])
        print_header("\nDuration statistics:")
        print_kv("  Total", f"{summary['total_duration']:.2f}s")
        print_kv("  Min", f"{summary['min_duration']:.2f}s")
        print_kv("  Max", f"{summary['max_duration']:.2f}s")
        print_kv("  Mean", f"{summary['mean_duration']:.2f}s")
        print_header("\nLabel counts:")
        for label, count in sorted(summary["labels"].items()):
            print_kv(f"  {label}", count)


@annotation.command("remap")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file")
@click.option(
    "--mapping", "-m", required=True, help="Path to label mapping CSV (columns: source, target)"
)
@click.option(
    "--keep-unmapped/--drop-unmapped",
    default=True,
    help="Keep or drop annotations with unmapped labels",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def annotation_remap(
    input_file: str, output_file: str, mapping: str, keep_unmapped: bool, quiet: bool
) -> None:
    """Remap annotation labels using a mapping file."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    result = run_operation(
        "annotation.remap",
        input=input_file,
        output=output_file,
        mapping=mapping,
        keep_unmapped=keep_unmapped,
    )
    if emit_result(result.to_dict()):
        return

    if not quiet:
        print_success(
            f"Remapped {result.input_count} annotations -> {result.output_count} annotations"
        )
        print_success(f"Output: {output_file}")


@annotation.command("filter")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file")
@click.option("--include", "-i", multiple=True, help="Labels to include (can specify multiple)")
@click.option("--exclude", "-e", multiple=True, help="Labels to exclude (can specify multiple)")
@click.option("--min-duration", type=float, default=None, help="Minimum duration in seconds")
@click.option("--max-duration", type=float, default=None, help="Maximum duration in seconds")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def annotation_filter(
    input_file: str,
    output_file: str,
    include: tuple,
    exclude: tuple,
    min_duration: float,
    max_duration: float,
    quiet: bool,
) -> None:
    """Filter annotations by label or duration."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    result = run_operation(
        "annotation.filter",
        input=input_file,
        output=output_file,
        include=list(include) or None,
        exclude=list(exclude) or None,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    if emit_result(result.to_dict()):
        return

    if not quiet:
        print_success(
            f"Filtered {result.input_count} annotations -> {result.output_count} annotations"
        )
        print_success(f"Output: {output_file}")


@annotation.command("generate-labels")
@click.argument("annotation_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file")
@click.option(
    "--audio-duration",
    type=float,
    default=None,
    help="Total audio duration in seconds (inferred from bioamla metadata if omitted)",
)
@click.option("--clip-duration", type=float, required=True, help="Duration of each clip in seconds")
@click.option(
    "--hop-length",
    type=float,
    default=None,
    help="Hop length between clips (default: same as clip duration)",
)
@click.option(
    "--min-overlap", type=float, default=0.0, help="Minimum overlap ratio to assign label (0.0-1.0)"
)
@click.option(
    "--multi-label/--single-label", default=True, help="Generate multi-label or single-label output"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["csv", "numpy"]),
    default="csv",
    help="Output format for labels",
)
@click.option(
    "--recording",
    default=None,
    help="For a file covering several recordings (e.g. an Arbimon export): the one to label",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def annotation_generate_labels(
    annotation_file: str,
    output_file: str,
    audio_duration: float,
    clip_duration: float,
    hop_length: float,
    min_overlap: float,
    multi_label: bool,
    output_format: str,
    recording: str | None,
    quiet: bool,
) -> None:
    """Generate clip-level labels from annotations."""
    from bioamla.cli.console import print_kv, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    result = run_operation(
        "annotation.generate-labels",
        annotation_file=annotation_file,
        output=output_file,
        clip_duration=clip_duration,
        audio_duration=audio_duration,
        hop_length=hop_length,
        min_overlap=min_overlap,
        multi_label=multi_label,
        output_format=output_format,
        recording=recording,
    )
    if emit_result(result.to_dict()):
        return

    if not quiet:
        print_success(f"Generated labels for {result.clips} clips")
        print_kv("Labels", ", ".join(sorted(result.labels)))
        print_success(f"Output: {result.output}")


@annotation.command("generate-frame-labels")
@click.argument("annotation_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file")
@click.option("--frame-size", type=float, required=True, help="Frame size in seconds")
@click.option(
    "--hop-length",
    type=float,
    default=None,
    help="Hop length between frames in seconds (default: same as frame size)",
)
@click.option(
    "--audio-duration",
    type=float,
    default=None,
    help="Total audio duration in seconds (inferred from bioamla metadata if omitted)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["csv", "numpy"]),
    default="csv",
    help="Output format for labels",
)
@click.option(
    "--recording",
    default=None,
    help="For a file covering several recordings (e.g. an Arbimon export): the one to label",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def annotation_generate_frame_labels(
    annotation_file: str,
    output_file: str,
    frame_size: float,
    hop_length: float,
    audio_duration: float,
    output_format: str,
    recording: str | None,
    quiet: bool,
) -> None:
    """Generate frame-level multi-hot labels from annotations (for SED-style models)."""
    from bioamla.cli.console import print_kv, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    result = run_operation(
        "annotation.generate-frame-labels",
        annotation_file=annotation_file,
        output=output_file,
        frame_size=frame_size,
        hop_length=hop_length,
        audio_duration=audio_duration,
        output_format=output_format,
        recording=recording,
    )
    if emit_result(result.to_dict()):
        return

    if not quiet:
        print_success(f"Generated frame labels: {result.classes} classes x {result.frames} frames")
        print_kv("Labels", ", ".join(sorted(result.labels)))
        print_success(f"Output: {result.output}")

"""Annotation management commands for audio datasets."""

import math
from pathlib import Path

import click

from bioamla.common.constants import ANNOTATION_FORMATS as _FORMAT_CHOICES
from bioamla.exceptions import BioamlaError

# The datasets package (and numpy/pandas with it) is imported only when a command
# runs, so `bioamla --help` stays fast. These wrappers keep the call sites short.


def _detect_format(*args, **kwargs):
    from bioamla.datasets._io import detect_annotation_format

    return detect_annotation_format(*args, **kwargs)


def _load(*args, **kwargs):
    from bioamla.datasets._io import load_annotations

    return load_annotations(*args, **kwargs)


def _one_recording(annotations, recording: str | None, source: str):
    """The rows for one recording: ``--recording``, or the file's only one.

    Labels laid on one timeline from several recordings would be meaningless, so a
    file naming more than one recording needs ``--recording``.
    """
    from bioamla.datasets._io import annotations_for_recording, recordings_named

    names = recordings_named(annotations)
    if recording is not None:
        kept = annotations_for_recording(annotations, recording)
        if names and not kept:
            raise click.ClickException(f"No rows in {source} are for recording {recording!r}")
        return kept
    if len(names) > 1:
        shown = ", ".join(sorted(names)[:3]) + (", ..." if len(names) > 3 else "")
        raise click.ClickException(
            f"{source} covers {len(names)} recordings ({shown}); pick one with --recording"
        )
    return annotations


def _save(*args, **kwargs):
    from bioamla.datasets._io import save_annotations

    return save_annotations(*args, **kwargs)


def _write_labels(output_file, output_format, array, label_map, header_prefix, rows):
    """Write generated labels; return (data path, label-map path or None).

    ``numpy``: the array as ``.npy`` (``np.save`` adds the suffix when missing)
    plus a ``<name>.labels.csv`` mapping each label to its column. ``csv``: one
    table with ``header_prefix`` columns, one column per label, and ``rows``.
    A failed write is a ClickException.
    """
    import csv as csv_lib

    import numpy as np

    output_path = Path(output_file)
    ordered = sorted(label_map, key=lambda label: label_map[label])
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "numpy":
            data_path = output_path if output_path.suffix == ".npy" else Path(f"{output_path}.npy")
            np.save(data_path, array)
            label_map_path = output_path.with_suffix(".labels.csv")
            with open(label_map_path, "w", newline="", encoding="utf-8") as f:
                writer = csv_lib.writer(f)
                writer.writerow(["label", "index"])
                for label in ordered:
                    writer.writerow([label, label_map[label]])
            return data_path, label_map_path
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_lib.writer(f)
            writer.writerow([*header_prefix, *ordered])
            writer.writerows(rows)
        return output_path, None
    except OSError as e:
        raise click.ClickException(f"Failed to write {output_file}: {e}") from e


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
    from datetime import datetime, timezone

    from bioamla.audio import get_audio_info
    from bioamla.cli.console import echo, print_kv, print_success
    from bioamla.cli.output import emit_result
    from bioamla.datasets import Annotation

    audio_path = Path(audio_file)

    output_path = Path(output_file)
    fmt = _detect_format(output_path, out_format)

    try:
        info = get_audio_info(audio_file)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    annotations = []
    if not empty:
        annotations.append(Annotation(start_time=0.0, end_time=info.duration, label=label))

    metadata = {
        "audio_file": audio_path.name,
        "sample_rate": info.sample_rate,
        "duration": round(info.duration, 6),
        "channels": info.channels,
        "created": datetime.now(timezone.utc).isoformat(),
    }

    try:
        _save(annotations, output_path, fmt, metadata=metadata)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if emit_result(
        {
            "audio_file": audio_file,
            "output": output_file,
            "format": fmt,
            "rows": len(annotations),
            "audio": {
                "duration": info.duration,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
            },
        }
    ):
        return

    if not quiet:
        print_success(f"Created {fmt} annotation template: {output_file}")
        echo(
            f"  audio: {audio_path.name}  duration: {info.duration:.2f}s  sr: {info.sample_rate} Hz"
        )
        print_kv("  rows", len(annotations))
        if fmt != "bioamla":
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
    from bioamla.cli.output import emit_result

    input_path = Path(input_file)
    output_path = Path(output_file)

    from_format = _detect_format(input_path, from_format, read_header=True)
    to_format = _detect_format(output_path, to_format)

    if recording is not None and to_format != "arbimon":
        raise click.UsageError("--recording only applies to --to arbimon")
    try:
        annotations, metadata = _load(input_path, from_format, label_column=label_column)
        _save(annotations, output_path, to_format, metadata=metadata, recording=recording)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if emit_result(
        {
            "input": input_file,
            "output": output_file,
            "from_format": from_format,
            "to_format": to_format,
            "count": len(annotations),
        }
    ):
        return

    if not quiet:
        print_success(f"Converted {len(annotations)} annotations from {from_format} to {to_format}")
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
    from bioamla.cli.output import emit_result, json_mode, legacy_json
    from bioamla.datasets import summarize_annotations

    input_path = Path(path)

    file_format = _detect_format(input_path, file_format, read_header=True)

    try:
        annotations, metadata = _load(input_path, file_format)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    summary = summarize_annotations(annotations)

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
    from bioamla.cli.output import emit_result
    from bioamla.datasets import load_label_mapping, remap_labels

    input_path = Path(input_file)
    output_path = Path(output_file)

    in_format = _detect_format(input_path, read_header=True)
    # Same extension (or none): keep the input's format, so an Arbimon CSV stays one.
    same = not output_path.suffix or output_path.suffix.lower() == input_path.suffix.lower()
    out_format = in_format if same else _detect_format(output_path)

    try:
        label_mapping = load_label_mapping(mapping)
        annotations, metadata = _load(input_path, in_format)
        original_count = len(annotations)
        remapped = remap_labels(annotations, label_mapping, keep_unmapped=keep_unmapped)
        _save(remapped, output_path, out_format, metadata=metadata)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if emit_result(
        {
            "input": input_file,
            "output": output_file,
            "input_count": original_count,
            "output_count": len(remapped),
        }
    ):
        return

    if not quiet:
        print_success(f"Remapped {original_count} annotations -> {len(remapped)} annotations")
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
    from bioamla.cli.output import emit_result
    from bioamla.datasets import filter_labels

    input_path = Path(input_file)
    output_path = Path(output_file)

    in_format = _detect_format(input_path, read_header=True)
    # Same extension (or none): keep the input's format, so an Arbimon CSV stays one.
    same = not output_path.suffix or output_path.suffix.lower() == input_path.suffix.lower()
    out_format = in_format if same else _detect_format(output_path)

    try:
        annotations, metadata = _load(input_path, in_format)
        original_count = len(annotations)

        include_set = set(include) if include else None
        exclude_set = set(exclude) if exclude else None
        filtered = filter_labels(
            annotations, include_labels=include_set, exclude_labels=exclude_set
        )

        if min_duration is not None:
            filtered = [a for a in filtered if a.duration >= min_duration]
        if max_duration is not None:
            filtered = [a for a in filtered if a.duration <= max_duration]

        _save(filtered, output_path, out_format, metadata=metadata)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if emit_result(
        {
            "input": input_file,
            "output": output_file,
            "input_count": original_count,
            "output_count": len(filtered),
        }
    ):
        return

    if not quiet:
        print_success(f"Filtered {original_count} annotations -> {len(filtered)} annotations")
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

    import numpy as np

    from bioamla.cli.console import print_kv, print_success
    from bioamla.cli.output import emit_result
    from bioamla.datasets import (
        create_label_map,
        generate_clip_labels,
        get_unique_labels,
    )

    input_path = Path(annotation_file)

    in_format = _detect_format(input_path, read_header=True)

    try:
        annotations, metadata = _load(input_path, in_format)
        annotations = _one_recording(annotations, recording, annotation_file)

        if not annotations:
            raise click.ClickException("No annotations found in file")

        # Fall back to the duration embedded in a bioamla file when not given.
        if audio_duration is None:
            if metadata.get("duration") is not None:
                audio_duration = float(metadata["duration"])
            else:
                raise click.ClickException(
                    "--audio-duration is required (no duration metadata in this file)"
                )

        labels = get_unique_labels(annotations)
        label_map = create_label_map(labels)

        if hop_length is None:
            hop_length = clip_duration
        if clip_duration <= 0 or hop_length <= 0:
            raise click.ClickException("--clip-duration and --hop-length must be positive")
        if clip_duration > audio_duration:
            raise click.ClickException(
                f"--clip-duration ({clip_duration}s) is longer than the audio ({audio_duration}s)"
            )

        # The small epsilon keeps float error from dropping the last clip
        # (e.g. 0.2 / 0.1 == 1.9999999999999998).
        num_clips = math.floor((audio_duration - clip_duration) / hop_length + 1e-9) + 1
        all_labels = []

        for i in range(num_clips):
            clip_start = i * hop_length
            clip_end = clip_start + clip_duration
            clip_labels = generate_clip_labels(
                annotations,
                clip_start,
                clip_end,
                label_map,
                min_overlap=min_overlap,
                multi_label=multi_label,
            )
            all_labels.append(clip_labels)

        labels_array = np.array(all_labels)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    data_path, label_map_path = _write_labels(
        output_file,
        output_format,
        labels_array,
        label_map,
        ["clip_start", "clip_end"],
        (
            [f"{i * hop_length:.3f}", f"{i * hop_length + clip_duration:.3f}"]
            + [int(v) for v in clip_labels]
            for i, clip_labels in enumerate(labels_array)
        ),
    )

    if emit_result(
        {
            "annotation_file": annotation_file,
            "output": str(data_path),
            "label_map": str(label_map_path) if label_map_path else None,
            "clips": num_clips,
            "labels": sorted(label_map),
        }
    ):
        return

    if not quiet:
        print_success(f"Generated labels for {num_clips} clips")
        print_kv("Labels", ", ".join(sorted(label_map.keys())))
        print_success(f"Output: {output_file}")


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
    from bioamla.cli.output import emit_result
    from bioamla.datasets import (
        create_label_map,
        generate_frame_labels,
        get_unique_labels,
    )

    input_path = Path(annotation_file)

    in_format = _detect_format(input_path, read_header=True)

    try:
        annotations, metadata = _load(input_path, in_format)
        annotations = _one_recording(annotations, recording, annotation_file)

        if not annotations:
            raise click.ClickException("No annotations found in file")

        # Fall back to the duration embedded in a bioamla file when not given.
        if audio_duration is None:
            if metadata.get("duration") is not None:
                audio_duration = float(metadata["duration"])
            else:
                raise click.ClickException(
                    "--audio-duration is required (no duration metadata in this file)"
                )

        if hop_length is None:
            hop_length = frame_size

        labels = get_unique_labels(annotations)
        label_map = create_label_map(labels)

        # Shape: (num_classes, num_frames).
        frame_labels = generate_frame_labels(
            annotations, audio_duration, frame_size, hop_length, label_map
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    num_classes, num_frames = frame_labels.shape
    data_path, label_map_path = _write_labels(
        output_file,
        output_format,
        frame_labels,
        label_map,
        ["frame_start", "frame_end"],
        (
            # Transpose (num_classes, num_frames) -> per-frame multi-hot row.
            [f"{i * hop_length:.3f}", f"{i * hop_length + frame_size:.3f}"]
            + [int(frame_labels[c, i]) for c in range(num_classes)]
            for i in range(num_frames)
        ),
    )

    if emit_result(
        {
            "annotation_file": annotation_file,
            "output": str(data_path),
            "label_map": str(label_map_path) if label_map_path else None,
            "classes": num_classes,
            "frames": num_frames,
            "labels": sorted(label_map),
        }
    ):
        return

    if not quiet:
        print_success(f"Generated frame labels: {num_classes} classes x {num_frames} frames")
        print_kv("Labels", ", ".join(sorted(label_map.keys())))
        print_success(f"Output: {output_file}")

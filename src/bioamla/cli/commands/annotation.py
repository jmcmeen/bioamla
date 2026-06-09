"""Annotation management commands for audio datasets."""

from pathlib import Path

import click

from bioamla.exceptions import BioamlaError

# Supported annotation file formats for the --from/--to/--format options.
_FORMAT_CHOICES = ["raven", "csv", "bioamla"]


def _detect_format(path: Path, explicit: str | None = None) -> str:
    """Resolve an annotation format from an explicit flag or the file extension.

    ``.txt`` -> raven selection table, ``.json`` -> bioamla format, anything
    else -> flat CSV.
    """
    if explicit:
        return explicit
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return "raven"
    if suffix == ".json":
        return "bioamla"
    return "csv"


def _load(path: Path, fmt: str, label_column: str | None = None):
    """Load annotations in the given format, returning ``(annotations, metadata)``.

    Only the bioamla format carries file-level metadata; the others return an
    empty metadata dict so callers can treat every format uniformly.
    """
    from bioamla.datasets import (
        load_bioamla_annotations,
        load_csv_annotations,
        load_raven_selection_table,
    )

    if fmt == "raven":
        return load_raven_selection_table(str(path), label_column=label_column), {}
    if fmt == "bioamla":
        return load_bioamla_annotations(str(path))
    return load_csv_annotations(str(path)), {}


def _save(annotations, path: Path, fmt: str, metadata: dict | None = None) -> None:
    """Save annotations in the given format, preserving metadata for bioamla."""
    from bioamla.datasets import (
        save_bioamla_annotations,
        save_csv_annotations,
        save_raven_selection_table,
    )

    if fmt == "raven":
        save_raven_selection_table(annotations, str(path))
    elif fmt == "bioamla":
        save_bioamla_annotations(annotations, str(path), metadata=metadata)
    else:
        save_csv_annotations(annotations, str(path))


@click.group()
def annotation() -> None:
    """Annotation management commands for audio datasets."""
    pass


@annotation.command("template")
@click.argument("audio_file")
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
@click.option("--quiet", is_flag=True, help="Suppress progress output")
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
    from bioamla.datasets import Annotation

    audio_path = Path(audio_file)
    if not audio_path.exists():
        click.echo(f"Error: Audio file not found: {audio_file}")
        raise SystemExit(1)

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

    if not quiet:
        click.echo(f"Created {fmt} annotation template: {output_file}")
        click.echo(
            f"  audio: {audio_path.name}  "
            f"duration: {info.duration:.2f}s  sr: {info.sample_rate} Hz"
        )
        click.echo(f"  rows: {len(annotations)}")
        if fmt != "bioamla":
            click.echo("  note: audio metadata is only persisted in the bioamla (.json) format")


@annotation.command("convert")
@click.argument("input_file")
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
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def annotation_convert(
    input_file: str,
    output_file: str,
    from_format: str,
    to_format: str,
    label_column: str,
    quiet: bool,
) -> None:
    """Convert annotation files between formats."""
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        click.echo(f"Error: Input file not found: {input_file}")
        raise SystemExit(1)

    from_format = _detect_format(input_path, from_format)
    to_format = _detect_format(output_path, to_format)

    try:
        annotations, metadata = _load(input_path, from_format, label_column=label_column)
        _save(annotations, output_path, to_format, metadata=metadata)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if not quiet:
        click.echo(f"Converted {len(annotations)} annotations from {from_format} to {to_format}")
        click.echo(f"Output: {output_file}")


@annotation.command("summary")
@click.argument("path")
@click.option(
    "--format",
    "file_format",
    type=click.Choice(_FORMAT_CHOICES),
    default=None,
    help="Annotation format (auto-detected from extension if not specified)",
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def annotation_summary(path: str, file_format: str, output_json: str) -> None:
    """Display summary statistics for an annotation file."""
    import json

    from bioamla.datasets import summarize_annotations

    input_path = Path(path)

    if not input_path.exists():
        click.echo(f"Error: File not found: {path}")
        raise SystemExit(1)

    file_format = _detect_format(input_path, file_format)

    try:
        annotations, metadata = _load(input_path, file_format)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    summary = summarize_annotations(annotations)

    if output_json:
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(f"\nAnnotation Summary: {path}")
        click.echo("=" * 50)
        if metadata.get("audio_file"):
            click.echo(f"Audio file: {metadata['audio_file']}")
        if metadata.get("duration") is not None:
            click.echo(f"Audio duration: {float(metadata['duration']):.2f}s")
        click.echo(f"Total annotations: {summary['total_annotations']}")
        click.echo(f"Unique labels: {summary['unique_labels']}")
        click.echo("\nDuration statistics:")
        click.echo(f"  Total: {summary['total_duration']:.2f}s")
        click.echo(f"  Min: {summary['min_duration']:.2f}s")
        click.echo(f"  Max: {summary['max_duration']:.2f}s")
        click.echo(f"  Mean: {summary['mean_duration']:.2f}s")
        click.echo("\nLabel counts:")
        for label, count in sorted(summary["labels"].items()):
            click.echo(f"  {label}: {count}")


@annotation.command("remap")
@click.argument("input_file")
@click.argument("output_file")
@click.option(
    "--mapping", "-m", required=True, help="Path to label mapping CSV (columns: source, target)"
)
@click.option(
    "--keep-unmapped/--drop-unmapped",
    default=True,
    help="Keep or drop annotations with unmapped labels",
)
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def annotation_remap(
    input_file: str, output_file: str, mapping: str, keep_unmapped: bool, quiet: bool
) -> None:
    """Remap annotation labels using a mapping file."""
    from bioamla.datasets import load_label_mapping, remap_labels

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        click.echo(f"Error: Input file not found: {input_file}")
        raise SystemExit(1)

    in_format = _detect_format(input_path)
    out_format = _detect_format(output_path) if output_path.suffix else in_format

    try:
        label_mapping = load_label_mapping(mapping)
        annotations, metadata = _load(input_path, in_format)
        original_count = len(annotations)
        remapped = remap_labels(annotations, label_mapping, keep_unmapped=keep_unmapped)
        _save(remapped, output_path, out_format, metadata=metadata)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if not quiet:
        click.echo(f"Remapped {original_count} annotations -> {len(remapped)} annotations")
        click.echo(f"Output: {output_file}")


@annotation.command("filter")
@click.argument("input_file")
@click.argument("output_file")
@click.option("--include", "-i", multiple=True, help="Labels to include (can specify multiple)")
@click.option("--exclude", "-e", multiple=True, help="Labels to exclude (can specify multiple)")
@click.option("--min-duration", type=float, default=None, help="Minimum duration in seconds")
@click.option("--max-duration", type=float, default=None, help="Maximum duration in seconds")
@click.option("--quiet", is_flag=True, help="Suppress progress output")
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
    from bioamla.datasets import filter_labels

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        click.echo(f"Error: Input file not found: {input_file}")
        raise SystemExit(1)

    in_format = _detect_format(input_path)
    out_format = _detect_format(output_path) if output_path.suffix else in_format

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

    if not quiet:
        click.echo(f"Filtered {original_count} annotations -> {len(filtered)} annotations")
        click.echo(f"Output: {output_file}")


@annotation.command("generate-labels")
@click.argument("annotation_file")
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
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def annotation_generate_labels(
    annotation_file: str,
    output_file: str,
    audio_duration: float,
    clip_duration: float,
    hop_length: float,
    min_overlap: float,
    multi_label: bool,
    output_format: str,
    quiet: bool,
) -> None:
    """Generate clip-level labels from annotations."""
    import csv as csv_lib

    import numpy as np

    from bioamla.datasets import (
        create_label_map,
        generate_clip_labels,
        get_unique_labels,
    )

    input_path = Path(annotation_file)

    if not input_path.exists():
        click.echo(f"Error: Annotation file not found: {annotation_file}")
        raise SystemExit(1)

    in_format = _detect_format(input_path)

    try:
        annotations, metadata = _load(input_path, in_format)

        if not annotations:
            click.echo("Error: No annotations found in file")
            raise SystemExit(1)

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

        num_clips = int((audio_duration - clip_duration) / hop_length) + 1
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

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "numpy":
        np.save(output_file, labels_array)
        label_map_file = output_path.with_suffix(".labels.csv")
        with open(label_map_file, "w", newline="", encoding="utf-8") as f:
            writer = csv_lib.writer(f)
            writer.writerow(["label", "index"])
            for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
                writer.writerow([label, idx])
    else:
        header = ["clip_start", "clip_end"] + sorted(label_map.keys(), key=lambda x: label_map[x])
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv_lib.writer(f)
            writer.writerow(header)
            for i, clip_labels in enumerate(labels_array):
                clip_start = i * hop_length
                clip_end = clip_start + clip_duration
                row = [f"{clip_start:.3f}", f"{clip_end:.3f}"] + [int(v) for v in clip_labels]
                writer.writerow(row)

    if not quiet:
        click.echo(f"Generated labels for {num_clips} clips")
        click.echo(f"Labels: {', '.join(sorted(label_map.keys()))}")
        click.echo(f"Output: {output_file}")

"""Annotation management commands for audio datasets."""

import click


@click.group()
def annotation() -> None:
    """Annotation management commands for audio datasets."""
    pass


@annotation.command("convert")
@click.argument("input_file")
@click.argument("output_file")
@click.option(
    "--from",
    "from_format",
    type=click.Choice(["raven", "csv"]),
    default=None,
    help="Input format (auto-detected from extension if not specified)",
)
@click.option(
    "--to",
    "to_format",
    type=click.Choice(["raven", "csv"]),
    default=None,
    help="Output format (auto-detected from extension if not specified)",
)
@click.option("--label-column", default=None, help="Column name for labels in input file")
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def annotation_convert(input_file : str, output_file : str, from_format : str, to_format : str, label_column : str, quiet : bool) -> None:
    """Convert annotation files between formats."""
    from pathlib import Path

    from bioamla.services.annotation import (
        load_csv_annotations,
        load_raven_selection_table,
        save_csv_annotations,
        save_raven_selection_table,
    )

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        click.echo(f"Error: Input file not found: {input_file}")
        raise SystemExit(1)

    if from_format is None:
        if input_path.suffix.lower() == ".txt":
            from_format = "raven"
        else:
            from_format = "csv"

    if to_format is None:
        if output_path.suffix.lower() == ".txt":
            to_format = "raven"
        else:
            to_format = "csv"

    if from_format == "raven":
        annotations = load_raven_selection_table(input_file, label_column=label_column)
    else:
        annotations = load_csv_annotations(input_file)

    if to_format == "raven":
        save_raven_selection_table(annotations, output_file)
    else:
        save_csv_annotations(annotations, output_file)

    if not quiet:
        click.echo(f"Converted {len(annotations)} annotations from {from_format} to {to_format}")
        click.echo(f"Output: {output_file}")


@annotation.command("summary")
@click.argument("path")
@click.option(
    "--format",
    "file_format",
    type=click.Choice(["raven", "csv"]),
    default=None,
    help="Annotation format (auto-detected from extension if not specified)",
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def annotation_summary(path : str, file_format : str, output_json : str) -> None:
    """Display summary statistics for an annotation file."""
    import json
    from pathlib import Path

    from bioamla.services.annotation import (
        load_csv_annotations,
        load_raven_selection_table,
        summarize_annotations,
    )

    input_path = Path(path)

    if not input_path.exists():
        click.echo(f"Error: File not found: {path}")
        raise SystemExit(1)

    if file_format is None:
        if input_path.suffix.lower() == ".txt":
            file_format = "raven"
        else:
            file_format = "csv"

    if file_format == "raven":
        annotations = load_raven_selection_table(path)
    else:
        annotations = load_csv_annotations(path)

    summary = summarize_annotations(annotations)

    if output_json:
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(f"\nAnnotation Summary: {path}")
        click.echo("=" * 50)
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
def annotation_remap(input_file : str, output_file : str, mapping : str, keep_unmapped: bool, quiet : bool) -> None:
    """Remap annotation labels using a mapping file."""
    from pathlib import Path

    from bioamla.core.annotations import (
        load_label_mapping,
        remap_labels,
    )
    from bioamla.services.annotation import (
        load_csv_annotations,
        load_raven_selection_table,
        save_csv_annotations,
        save_raven_selection_table,
    )

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        click.echo(f"Error: Input file not found: {input_file}")
        raise SystemExit(1)

    label_mapping = load_label_mapping(mapping)

    if input_path.suffix.lower() == ".txt":
        annotations = load_raven_selection_table(input_file)
        is_raven = True
    else:
        annotations = load_csv_annotations(input_file)
        is_raven = False

    original_count = len(annotations)

    remapped = remap_labels(annotations, label_mapping, keep_unmapped=keep_unmapped)

    if output_path.suffix.lower() == ".txt" or is_raven:
        save_raven_selection_table(remapped, output_file)
    else:
        save_csv_annotations(remapped, output_file)

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
def annotation_filter(input_file: str, output_file: str, include: tuple, exclude: tuple, min_duration: float, max_duration: float, quiet: bool) -> None:
    """Filter annotations by label or duration."""
    from pathlib import Path

    from bioamla.core.annotations import filter_labels
    from bioamla.services.annotation import (
        load_csv_annotations,
        load_raven_selection_table,
        save_csv_annotations,
        save_raven_selection_table,
    )

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        click.echo(f"Error: Input file not found: {input_file}")
        raise SystemExit(1)

    if input_path.suffix.lower() == ".txt":
        annotations = load_raven_selection_table(input_file)
        is_raven = True
    else:
        annotations = load_csv_annotations(input_file)
        is_raven = False

    original_count = len(annotations)

    include_set = set(include) if include else None
    exclude_set = set(exclude) if exclude else None
    filtered = filter_labels(annotations, include_labels=include_set, exclude_labels=exclude_set)

    if min_duration is not None:
        filtered = [a for a in filtered if a.duration >= min_duration]
    if max_duration is not None:
        filtered = [a for a in filtered if a.duration <= max_duration]

    if output_path.suffix.lower() == ".txt" or is_raven:
        save_raven_selection_table(filtered, output_file)
    else:
        save_csv_annotations(filtered, output_file)

    if not quiet:
        click.echo(f"Filtered {original_count} annotations -> {len(filtered)} annotations")
        click.echo(f"Output: {output_file}")


@annotation.command("generate-labels")
@click.argument("annotation_file")
@click.argument("output_file")
@click.option("--audio-duration", type=float, required=True, help="Total audio duration in seconds")
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
    from pathlib import Path

    import numpy as np

    from bioamla.cli.service_helpers import services
    from bioamla.core.annotations import (
        create_label_map,
        generate_clip_labels,
    )
    from bioamla.services.annotation import (
        get_unique_labels,
        load_csv_annotations,
        load_raven_selection_table,
    )

    input_path = Path(annotation_file)

    if not input_path.exists():
        click.echo(f"Error: Annotation file not found: {annotation_file}")
        raise SystemExit(1)

    if input_path.suffix.lower() == ".txt":
        annotations = load_raven_selection_table(annotation_file)
    else:
        annotations = load_csv_annotations(annotation_file)

    if not annotations:
        click.echo("Error: No annotations found in file")
        raise SystemExit(1)

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

    output_path = Path(output_file)
    services.file.ensure_directory(output_path.parent)

    if output_format == "numpy":
        services.file.write_npy(output_file, labels_array)
        label_map_file = output_path.with_suffix(".labels.csv")
        rows = [["label", "index"]]
        for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
            rows.append([label, idx])
        services.file.write_csv(label_map_file, rows[1:], headers=rows[0])
    else:
        header = ["clip_start", "clip_end"] + sorted(
            label_map.keys(), key=lambda x: label_map[x]
        )
        rows = []
        for i, clip_labels in enumerate(labels_array):
            clip_start = i * hop_length
            clip_end = clip_start + clip_duration
            row = [f"{clip_start:.3f}", f"{clip_end:.3f}"] + [int(v) for v in clip_labels]
            rows.append(row)
        services.file.write_csv(output_file, rows, headers=header)

    if not quiet:
        click.echo(f"Generated labels for {num_clips} clips")
        click.echo(f"Labels: {', '.join(sorted(label_map.keys()))}")
        click.echo(f"Output: {output_file}")

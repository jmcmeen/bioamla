"""Dataset management commands."""

import click


@click.group()
def dataset() -> None:
    """Dataset management commands."""
    pass


@dataset.command("merge")
@click.argument("output_dir")
@click.argument("dataset_paths", nargs=-1, required=True)
@click.option(
    "--metadata-filename", default="metadata.csv", help="Name of metadata CSV file in each dataset"
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing files instead of skipping")
@click.option(
    "--no-organize",
    is_flag=True,
    help="Preserve original directory structure instead of organizing by category",
)
@click.option(
    "--target-format",
    default=None,
    help="Convert all audio files to this format (wav, mp3, flac, etc.)",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def dataset_merge(
    output_dir: str,
    dataset_paths: tuple,
    metadata_filename: str,
    overwrite: bool,
    no_organize: bool,
    target_format: str,
    quiet: bool,
) -> None:
    """Merge multiple audio datasets into a single dataset."""
    from bioamla.cli.console import print_success
    from bioamla.cli.logging_setup import configure_cli_logging
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    configure_cli_logging()
    stats = run_operation(
        "dataset.merge",
        output_dir=output_dir,
        dataset_paths=list(dataset_paths),
        metadata_filename=metadata_filename,
        overwrite=overwrite,
        organize=not no_organize,
        target_format=target_format,
        verbose=not quiet,
    )
    if emit_result(stats):
        return

    if quiet:
        msg = f"Merged {stats['datasets_merged']} datasets: {stats['total_files']} total files"
        if target_format:
            msg += f", {stats['files_converted']} converted"
        print_success(msg)


@dataset.command("extract-clips")
@click.argument("source")
@click.argument("output_dir")
@click.option(
    "--annotations",
    default=None,
    help="Annotation file when SOURCE is a single audio file (else a sibling file is used)",
)
@click.option(
    "--layout",
    type=click.Choice(["both", "audiofolder", "flat"]),
    default="both",
    help="Output layout: label subdirs + metadata.csv (both), subdirs only, or flat + metadata.csv",
)
@click.option("--padding-ms", type=float, default=0.0, help="Padding before/after each clip (ms)")
@click.option(
    "--bandpass/--no-bandpass",
    default=False,
    help="Bandpass-filter clips to each annotation's frequency band",
)
@click.option("--format", "audio_format", default="wav", help="Output audio format")
@click.option(
    "--sample-rate", type=int, default=None, help="Resample clips to this rate (e.g. 16000 for AST)"
)
@click.option("--include", "-i", multiple=True, help="Labels to include (repeatable)")
@click.option("--exclude", "-e", multiple=True, help="Labels to exclude (repeatable)")
@click.option("--min-duration", type=float, default=None, help="Drop clips shorter than this (s)")
@click.option(
    "--source-metadata",
    default=None,
    help="Catalog metadata.csv to join license/attribution onto clips "
    "(auto-detected as a metadata.csv sibling of SOURCE when omitted)",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def dataset_extract_clips(
    source: str,
    output_dir: str,
    annotations: str,
    layout: str,
    padding_ms: float,
    bandpass: bool,
    audio_format: str,
    sample_rate: int,
    include: tuple,
    exclude: tuple,
    min_duration: float,
    source_metadata: str,
    quiet: bool,
) -> None:
    """Extract annotated regions into a labeled clip dataset (training-ready).

    SOURCE is an audio file (with --annotations or a sibling annotation file) or
    a directory of audio files each paired with a sibling annotation. The output
    is consumable by `bioamla models ast train` directly (label subdirs and/or a
    metadata.csv). License/attribution is joined from the source catalog
    metadata.csv (auto-detected, or via --source-metadata) so clips stay
    traceable to their original recordings.
    """
    from bioamla.cli.console import echo, print_kv, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    result = run_operation(
        "dataset.extract-clips",
        source=source,
        output_dir=output_dir,
        annotations=annotations,
        layout=layout,
        padding_ms=padding_ms,
        bandpass=bandpass,
        audio_format=audio_format,
        sample_rate=sample_rate,
        include=list(include) or None,
        exclude=list(exclude) or None,
        min_duration=min_duration,
        source_metadata=source_metadata,
        verbose=not quiet,
    )
    if emit_result(result):
        return

    if not quiet:
        print_success(
            f"Extracted {result['clips_written']} clips from "
            f"{result['files_processed']} file(s) into {result['output_dir']}"
        )
        print_kv(f"Labels ({len(result['labels'])})", ", ".join(result["labels"]))
        if result["metadata_file"]:
            print_kv("Metadata", result["metadata_file"])
        prov = result.get("provenance", {})
        if prov.get("joined"):
            echo(
                f"Provenance: joined {prov['matched']}/{result['clips_written']} clips "
                f"({', '.join(prov['columns'])})"
            )
            if prov.get("unmatched"):
                echo(f"  {prov['unmatched']} clip(s) had no source-metadata match")
        if result.get("skipped"):
            echo(f"Skipped (out of range): {len(result['skipped'])} clip(s)")
        if result["failed"]:
            echo(f"Failed: {len(result['failed'])} clip(s)")


@dataset.command("stats")
@click.argument("dataset_dir")
@click.option(
    "--metadata-filename", default="metadata.csv", help="Name of the metadata CSV in the dataset"
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Deprecated: bare JSON. Use `bioamla --json dataset stats` instead.",
)
def dataset_stats(dataset_dir: str, metadata_filename: str, output_json: bool) -> None:
    """Show summary statistics for a dataset's metadata.csv.

    DATASET_DIR is the dataset directory, or the path of its metadata CSV.
    """
    from bioamla.cli.console import echo, print_header, print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result, json_mode, legacy_json

    stats = run_operation(
        "dataset.stats", dataset_dir=dataset_dir, metadata_filename=metadata_filename
    )
    if json_mode():
        emit_result(stats)
        return
    if output_json:
        legacy_json(stats)
        return

    print_header(f"\nDataset Statistics: {dataset_dir}")
    echo("=" * 50)
    print_kv("Total files", stats["total_files"])
    print_kv("Labels", stats["num_categories"])
    if stats.get("splits"):
        split_str = ", ".join(f"{k}={v}" for k, v in sorted(stats["splits"].items()))
        print_kv("Splits", split_str)
    print_header("\nLabel counts:")
    for label, count in sorted(stats["categories"].items()):
        echo(f"  {label}: {count}")
    if stats.get("licenses"):
        print_header("\nLicenses:")
        for lic, count in sorted(stats["licenses"].items()):
            echo(f"  {lic or '(none)'}: {count}")


@dataset.command("manifest")
@click.argument("dataset_dir")
@click.option(
    "--name", default="", help="Dataset name recorded in the manifest (defaults to dir name)"
)
@click.option(
    "--kind",
    type=click.Choice(["labeled", "partitioned"]),
    default="labeled",
    help="Dataset kind",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output manifest path (default: DATASET_DIR/dataset.json)",
)
@click.option(
    "--metadata-filename", default="metadata.csv", help="Name of the metadata CSV in the dataset"
)
@click.option("--sample-rate", type=int, default=None, help="Sample rate to record in the manifest")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def dataset_manifest(
    dataset_dir: str,
    name: str,
    kind: str,
    output: str,
    metadata_filename: str,
    sample_rate: int,
    quiet: bool,
) -> None:
    """Build a dataset.json manifest (label vocabulary, counts, splits) from metadata.csv."""
    from bioamla.cli.console import echo, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    written = run_operation(
        "dataset.manifest",
        dataset_dir=dataset_dir,
        name=name,
        kind=kind,
        output=output,
        metadata_filename=metadata_filename,
        sample_rate=sample_rate,
    )
    if emit_result(written):
        return

    manifest, output_path = written["manifest"], written["output"]
    if not quiet:
        print_success(f"Wrote manifest: {output_path}")
        echo(f"  classes: {len(manifest.label2id)}  files: {sum(manifest.class_counts.values())}")
        if manifest.splits:
            split_str = ", ".join(f"{k}={v}" for k, v in sorted(manifest.splits.items()))
            echo(f"  splits: {split_str}")
        if manifest.recorded_from:
            echo(f"  recorded: {manifest.recorded_from} to {manifest.recorded_to}")
        if manifest.devices or manifest.sites:
            echo(f"  recorders: {len(manifest.devices)}  sites: {len(manifest.sites)}")


@dataset.command("partition")
@click.argument("dataset_dir")
@click.option("--train", "train_frac", type=float, default=0.70, help="Train fraction")
@click.option("--val", "val_frac", type=float, default=0.15, help="Validation fraction")
@click.option("--test", "test_frac", type=float, default=0.15, help="Test fraction")
@click.option("--seed", type=int, default=0, help="Reproducible shuffle seed")
@click.option("--stratify/--no-stratify", default=True, help="Balance labels across splits")
@click.option(
    "--mode",
    type=click.Choice(["subdirs", "column"]),
    default="subdirs",
    show_default=True,
    help="Reorganize into train/val/test/<label>/ (subdirs) or populate a split column",
)
@click.option(
    "--group-by",
    default="source_file",
    show_default=True,
    help="Keep rows sharing this column's value in one split (prevents clip leakage); "
    '"" to split row by row',
)
@click.option(
    "--background-label",
    default=None,
    help="Partition this label as its own stratum so it appears in every split",
)
@click.option(
    "--metadata-filename", default="metadata.csv", help="Name of the metadata CSV in the dataset"
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def dataset_partition(
    dataset_dir: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    stratify: bool,
    mode: str,
    group_by: str,
    background_label: str,
    metadata_filename: str,
    quiet: bool,
) -> None:
    """Partition a dataset into train/val/test (stratified, grouped, reproducible).

    DATASET_DIR is the dataset directory, or the path of its metadata CSV.

    \b
    Examples:
        # Default: 80/10/10, clips from one recording kept together, files
        # moved into train/val/test/<label>/ (and a split column written)
        bioamla dataset partition ./dataset
        # Leave the files where they are; only write the split column
        bioamla dataset partition ./dataset --mode column
        # Split clip by clip, ignoring which recording each came from
        bioamla dataset partition ./dataset --group-by ""
    """
    from bioamla.cli.console import print_kv, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    result = run_operation(
        "dataset.partition",
        dataset_dir=dataset_dir,
        train=train_frac,
        val=val_frac,
        test=test_frac,
        seed=seed,
        stratify=stratify,
        mode=mode,
        group_by=group_by,
        background_label=background_label,
        metadata_filename=metadata_filename,
        verbose=not quiet,
    )
    if emit_result(result):
        return

    if not quiet:
        split_str = ", ".join(f"{k}={v}" for k, v in sorted(result["splits"].items()))
        print_success(
            f"Partitioned {result['groups']} group(s) into splits ({result['mode']}): {split_str}"
        )
        print_kv("Metadata", result["metadata_file"])


@dataset.command("build")
@click.argument("source")
@click.argument("output_dir")
@click.option(
    "--annotations",
    default=None,
    help="Annotation file when SOURCE is a single audio file (else a sibling file is used)",
)
@click.option("--padding-ms", type=float, default=0.0, help="Padding before/after each clip (ms)")
@click.option(
    "--bandpass/--no-bandpass",
    default=False,
    help="Bandpass-filter clips to each annotation's frequency band",
)
@click.option(
    "--sample-rate", type=int, default=None, help="Resample clips to this rate (e.g. 16000 for AST)"
)
@click.option("--include", "-i", multiple=True, help="Labels to include (repeatable)")
@click.option("--exclude", "-e", multiple=True, help="Labels to exclude (repeatable)")
@click.option("--min-duration", type=float, default=None, help="Drop clips shorter than this (s)")
@click.option("--train", "train_frac", type=float, default=0.70, help="Train fraction")
@click.option("--val", "val_frac", type=float, default=0.15, help="Validation fraction")
@click.option("--test", "test_frac", type=float, default=0.15, help="Test fraction")
@click.option("--seed", type=int, default=0, help="Reproducible shuffle seed")
@click.option("--no-partition", is_flag=True, help="Extract clips but skip train/val/test split")
@click.option("--name", default="", help="Dataset name for the manifest (defaults to dir name)")
@click.option(
    "--source-metadata",
    default=None,
    help="Catalog metadata.csv to join license/attribution onto clips "
    "(auto-detected as a metadata.csv sibling of SOURCE when omitted)",
)
@click.option(
    "--attributions/--no-attributions",
    default=True,
    help="Write ATTRIBUTIONS.md from joined clip provenance (skipped if none)",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def dataset_build(
    source: str,
    output_dir: str,
    annotations: str,
    padding_ms: float,
    bandpass: bool,
    sample_rate: int,
    include: tuple,
    exclude: tuple,
    min_duration: float,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    no_partition: bool,
    name: str,
    source_metadata: str,
    attributions: bool,
    quiet: bool,
) -> None:
    """Build a training-ready dataset: extract clips, partition, and write a manifest.

    Chains `extract-clips` (layout=both) -> `partition` (subdirs) -> `manifest`,
    producing a dataset directory + dataset.json consumable by `models ast train`.
    License/attribution joined from the source catalog metadata is written to
    ATTRIBUTIONS.md (unless --no-attributions).

    \b
    Examples:
        # Annotated recordings dir -> train-ready dataset
        bioamla dataset build ./recordings ./dataset --sample-rate 16000
        # Keep two labels and set an 80/10/10 train/val/test split
        bioamla dataset build ./recordings ./dataset \\
            --include frog --include bird --train 0.8 --val 0.1 --test 0.1
    """
    from bioamla.cli.console import echo, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    built = run_operation(
        "dataset.build",
        source=source,
        output_dir=output_dir,
        annotations=annotations,
        padding_ms=padding_ms,
        bandpass=bandpass,
        sample_rate=sample_rate,
        include=list(include) or None,
        exclude=list(exclude) or None,
        min_duration=min_duration,
        train=train_frac,
        val=val_frac,
        test=test_frac,
        seed=seed,
        partition=not no_partition,
        name=name,
        source_metadata=source_metadata,
        attributions=attributions,
        verbose=not quiet,
    )
    if emit_result(built):
        return

    if not quiet:
        print_success(f"Built dataset at {output_dir}")
        echo(
            f"  clips: {built['extract']['clips_written']}  classes: {len(built['manifest'].label2id)}"
        )
        if built["partition"]:
            split_str = ", ".join(
                f"{k}={v}" for k, v in sorted(built["partition"]["splits"].items())
            )
            echo(f"  splits: {split_str}")
        echo(f"  manifest: {built['manifest_path']}")
        if built["attributions"]:
            echo(f"  attributions: {built['attributions']}")
        echo(f"Push to the Hub with: bioamla catalogs hf push-dataset {output_dir} <repo-id>")


@dataset.command("license")
@click.argument("path")
@click.option("--template", "-t", default=None, help="Template file to prepend to the license file")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output filename (default: LICENSE for text, ATTRIBUTIONS.md for md)",
)
@click.option(
    "--format",
    "out_format",
    type=click.Choice(["text", "md"]),
    default="text",
    help="Output format: plain-text LICENSE or Markdown ATTRIBUTIONS",
)
@click.option("--metadata-filename", default="metadata.csv", help="Name of metadata CSV file")
@click.option(
    "--batch",
    is_flag=True,
    help="Process all datasets in directory (each subdirectory with metadata.csv)",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def dataset_license(
    path: str,
    template: str,
    output: str,
    out_format: str,
    metadata_filename: str,
    batch: bool,
    quiet: bool,
) -> None:
    """Generate license/attribution file from dataset metadata."""
    from pathlib import Path

    from bioamla.cli.console import echo, print_header, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    stats = run_operation(
        "dataset.license",
        path=path,
        template=template,
        output=output,
        format=out_format,
        metadata_filename=metadata_filename,
        batch=batch,
    )
    if not quiet:  # after the checks, so a bad path only prints the error
        echo(
            f"Scanning directory for datasets: {path}"
            if batch
            else f"Generating license file for: {path}"
        )

    if batch:
        failed = [r["dataset_name"] for r in stats["results"] if r["status"] != "success"]
        if not failed and emit_result(stats):
            return

        if not quiet:
            print_header(f"\nProcessed {stats['datasets_found']} dataset(s):")
            echo(f"  Successful: {stats['datasets_processed']}")
            echo(f"  Failed: {stats['datasets_failed']}")

            for item in stats["results"]:
                if item["status"] == "success":
                    echo(f"  - {item['dataset_name']}: {item['attributions_count']} attributions")
                else:
                    echo(
                        f"  - {item['dataset_name']}: FAILED - {item.get('error', 'Unknown error')}"
                    )
        else:
            print_success(f"Generated {stats['datasets_processed']} license files")

        if failed:
            from bioamla.cli.output import fail

            fail(
                f"License generation failed for {len(failed)} of {stats['datasets_found']} "
                f"dataset(s): {', '.join(failed)}",
                "DatasetError",
                details=stats,
            )
        return

    if emit_result(stats):
        return
    if not quiet:
        print_success(f"License file generated: {stats['output_path']}")
        echo(f"  Attributions: {stats['attributions_count']}")
        echo(f"  File size: {stats['file_size']:,} bytes")
    else:
        out_name = Path(stats["output_path"]).name
        print_success(f"Generated {out_name} with {stats['attributions_count']} attributions")


def _parse_range(value: str) -> tuple[float, float]:
    """Parse ``MIN,MAX``, ``MIN-MAX`` or a single number into ``(min, max)``.

    Negative bounds work in either form (``-2,2``, ``-2-2``, ``-6--1``). Anything
    else raises :class:`click.BadParameter`.
    """
    import re

    number = r"-?\d+(?:\.\d+)?"
    match = re.fullmatch(rf"\s*({number})\s*(?:,|-)\s*({number})\s*", value)
    if match:
        return float(match.group(1)), float(match.group(2))
    if re.fullmatch(rf"\s*{number}\s*", value):
        return float(value), float(value)
    raise click.BadParameter(
        f"{value!r} is not a range; use MIN,MAX (e.g. -2,2), MIN-MAX (e.g. 0.8-1.2) or one number"
    )


@dataset.command("augment")
@click.argument("input_dir")
@click.option("--output", "-o", required=True, help="Output directory for augmented files")
@click.option(
    "--add-noise", default=None, help='Add Gaussian noise with SNR range (e.g., "3-30" dB)'
)
@click.option("--time-stretch", default=None, help='Time stretch range (e.g., "0.8-1.2")')
@click.option("--pitch-shift", default=None, help='Pitch shift range in semitones (e.g., "-2,2")')
@click.option("--gain", default=None, help='Gain range in dB (e.g., "-12,12")')
@click.option(
    "--multiply", default=1, type=int, help="Number of augmented copies to create per file"
)
@click.option("--sample-rate", default=16000, type=int, help="Target sample rate for output")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def dataset_augment(
    input_dir: str,
    output: str,
    add_noise: str,
    time_stretch: str,
    pitch_shift: str,
    gain: str,
    multiply: int,
    sample_rate: int,
    recursive: bool,
    quiet: bool,
) -> None:
    """Augment audio files to expand training datasets.

    Each enabled transform takes a range; ranges are sampled per output copy.
    This writes new augmented files (offline) — distinct from `models ast train`'s
    on-the-fly augmentation.

    \b
    Examples:
        # Two noisy + pitch-shifted copies per file
        bioamla dataset augment ./clips -o ./aug \\
            --add-noise 3-30 --pitch-shift -2,2 --multiply 2
        # Time-stretch only
        bioamla dataset augment ./clips -o ./aug --time-stretch 0.9-1.1
    """
    from bioamla.cli.console import print_success, print_warning
    from bioamla.cli.logging_setup import configure_cli_logging
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    ranges = {
        "add_noise": _parse_range(add_noise) if add_noise else None,
        "time_stretch": _parse_range(time_stretch) if time_stretch else None,
        "pitch_shift": _parse_range(pitch_shift) if pitch_shift else None,
        "gain": _parse_range(gain) if gain else None,
    }
    configure_cli_logging()
    stats = run_operation(
        "dataset.augment",
        input_dir=input_dir,
        output_dir=output,
        multiply=multiply,
        sample_rate=sample_rate,
        recursive=recursive,
        verbose=not quiet,
        **ranges,
    )
    if emit_result(stats):
        return

    summary = (
        f"Created {stats['files_created']} augmented files from "
        f"{stats['files_processed']} source files in {stats['output_dir']}"
    )
    if stats["files_failed"]:
        print_warning(f"{summary}; {stats['files_failed']} failed")
    else:
        print_success(summary)


@dataset.command("remix")
@click.argument("output_dir")
@click.option(
    "--recipe", "-r", default=None, help="TOML recipe ([remix] table plus [[add]]/[[replace]])"
)
@click.option("--source", "-s", default=None, help="Source dataset dir (overrides the recipe)")
@click.option("--keep", "-k", multiple=True, help="Keep only these labels (repeatable)")
@click.option("--drop", "-d", multiple=True, help="Drop these labels (repeatable)")
@click.option("--name", default=None, help="Dataset name for reporting (defaults to dir name)")
@click.option(
    "--label-column", default=None, help="Column holding the label (auto: label/category)"
)
@click.option(
    "-t", "--template", default=None, help="Header file to prepend to the generated LICENSE"
)
@click.option("--metadata-filename", default="metadata.csv", help="Name of the metadata CSV")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def dataset_remix(
    output_dir: str,
    recipe: str | None,
    source: str | None,
    keep: tuple,
    drop: tuple,
    name: str | None,
    label_column: str | None,
    template: str | None,
    metadata_filename: str,
    quiet: bool,
) -> None:
    """Build a derived dataset: keep/drop classes, add classes, replace examples.

    Reads a TOML recipe and/or command-line selections and writes a new dataset
    directory organized by label, with metadata.csv and a generated LICENSE.
    Flags override the matching recipe fields; --keep or --drop replaces the
    recipe's own selection.

    Examples:

        bioamla dataset remix build/esc-50-animals --recipe remixes/esc-50-animals.toml

        bioamla dataset remix build/no-planes --source datasets/esc-50 --drop airplane
    """
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    stats = run_operation(
        "dataset.remix",
        output_dir=output_dir,
        recipe=recipe,
        source=source,
        keep=list(keep) or None,
        drop=list(drop) or None,
        name=name,
        label_column=label_column,
        template=template,
        metadata_filename=metadata_filename,
        verbose=not quiet,
    )
    if emit_result(stats):
        return

    if not quiet:
        click.echo(f"Remix: {stats['name']}")
        click.echo(f"  Labels ({stats['kept']}): {', '.join(stats['labels'])}")
        click.echo(f"  Files copied: {stats['files_copied']}")
        if stats["files_missing"]:
            click.echo(f"  Source files missing: {stats['files_missing']}")
        if stats["files_duplicate"]:
            click.echo(f"  Duplicate file names skipped: {stats['files_duplicate']}")
        click.echo(
            f"  Source rows dropped: {stats['dropped']}, labels added: {stats['added']}, "
            f"replaced: {stats['replaced']}"
        )
        click.echo(f"  Metadata: {stats['metadata_file']}")
        if stats["license_file"]:
            click.echo(f"  License: {stats['license_file']}")
    print_success(f"Wrote {stats['total_files']} files to {stats['output_dir']}")

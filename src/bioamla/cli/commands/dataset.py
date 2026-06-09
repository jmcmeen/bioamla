"""Dataset management commands."""

import click

from bioamla.exceptions import BioamlaError


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
@click.option("--quiet", is_flag=True, help="Suppress progress output")
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
    from bioamla.datasets import merge_datasets

    try:
        stats = merge_datasets(
            dataset_paths=list(dataset_paths),
            output_dir=output_dir,
            metadata_filename=metadata_filename,
            skip_existing=not overwrite,
            organize_by_category=not no_organize,
            target_format=target_format,
            verbose=not quiet,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if quiet:
        msg = f"Merged {stats['datasets_merged']} datasets: {stats['total_files']} total files"
        if target_format:
            msg += f", {stats['files_converted']} converted"
        click.echo(msg)


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
@click.option("--quiet", is_flag=True, help="Suppress progress output")
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
    quiet: bool,
) -> None:
    """Extract annotated regions into a labeled clip dataset (training-ready).

    SOURCE is an audio file (with --annotations or a sibling annotation file) or
    a directory of audio files each paired with a sibling annotation. The output
    is consumable by `bioamla models ast train` directly (label subdirs and/or a
    metadata.csv).
    """
    from bioamla.datasets import extract_labeled_dataset

    try:
        result = extract_labeled_dataset(
            source=source,
            output_dir=output_dir,
            annotations=annotations,
            layout=layout,
            padding_ms=padding_ms,
            bandpass=bandpass,
            format=audio_format,
            target_sample_rate=sample_rate,
            include_labels=set(include) if include else None,
            exclude_labels=set(exclude) if exclude else None,
            min_duration=min_duration,
            verbose=not quiet,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if not quiet:
        click.echo(
            f"Extracted {result['clips_written']} clips from "
            f"{result['files_processed']} file(s) into {result['output_dir']}"
        )
        click.echo(f"Labels ({len(result['labels'])}): {', '.join(result['labels'])}")
        if result["metadata_file"]:
            click.echo(f"Metadata: {result['metadata_file']}")
        if result.get("skipped"):
            click.echo(f"Skipped (out of range): {len(result['skipped'])} clip(s)")
        if result["failed"]:
            click.echo(f"Failed: {len(result['failed'])} clip(s)")


@dataset.command("stats")
@click.argument("dataset_dir")
@click.option(
    "--metadata-filename", default="metadata.csv", help="Name of the metadata CSV in the dataset"
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def dataset_stats(dataset_dir: str, metadata_filename: str, output_json: bool) -> None:
    """Show summary statistics for a dataset's metadata.csv."""
    import json as json_lib

    from bioamla.datasets import get_dataset_stats

    try:
        stats = get_dataset_stats(dataset_dir, metadata_filename=metadata_filename)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if output_json:
        click.echo(json_lib.dumps(stats, indent=2))
        return

    click.echo(f"\nDataset Statistics: {dataset_dir}")
    click.echo("=" * 50)
    click.echo(f"Total files: {stats['total_files']}")
    click.echo(f"Classes: {stats['num_categories']}")
    if stats.get("splits"):
        split_str = ", ".join(f"{k}={v}" for k, v in sorted(stats["splits"].items()))
        click.echo(f"Splits: {split_str}")
    click.echo("\nLabel counts:")
    for label, count in sorted(stats["categories"].items()):
        click.echo(f"  {label}: {count}")
    if stats.get("licenses"):
        click.echo("\nLicenses:")
        for lic, count in sorted(stats["licenses"].items()):
            click.echo(f"  {lic or '(none)'}: {count}")


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
    default=None,
    help="Output manifest path (default: DATASET_DIR/dataset.json)",
)
@click.option(
    "--metadata-filename", default="metadata.csv", help="Name of the metadata CSV in the dataset"
)
@click.option("--sample-rate", type=int, default=None, help="Sample rate to record in the manifest")
@click.option("--quiet", is_flag=True, help="Suppress progress output")
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
    from datetime import datetime, timezone
    from pathlib import Path

    from bioamla.datasets import build_manifest_from_metadata, save_dataset_manifest

    dataset_path = Path(dataset_dir)
    manifest_name = name or dataset_path.name
    output_path = output or str(dataset_path / "dataset.json")

    try:
        manifest = build_manifest_from_metadata(
            dataset_dir,
            name=manifest_name,
            kind=kind,
            created=datetime.now(timezone.utc).isoformat(),
            metadata_filename=metadata_filename,
            sample_rate=sample_rate,
        )
        save_dataset_manifest(manifest, output_path)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if not quiet:
        click.echo(f"Wrote manifest: {output_path}")
        click.echo(
            f"  classes: {len(manifest.label2id)}  files: {sum(manifest.class_counts.values())}"
        )
        if manifest.splits:
            split_str = ", ".join(f"{k}={v}" for k, v in sorted(manifest.splits.items()))
            click.echo(f"  splits: {split_str}")


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
    help="Reorganize into train/val/test/<label>/ (subdirs) or populate a split column",
)
@click.option(
    "--group-by",
    default="source_file",
    help="Keep rows sharing this column's value in one split (prevents clip leakage)",
)
@click.option(
    "--background-label",
    default=None,
    help="Partition this label as its own stratum so it appears in every split",
)
@click.option(
    "--metadata-filename", default="metadata.csv", help="Name of the metadata CSV in the dataset"
)
@click.option("--quiet", is_flag=True, help="Suppress progress output")
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
    """Partition a dataset into train/val/test (stratified, grouped, reproducible)."""
    from bioamla.datasets import partition_dataset

    try:
        result = partition_dataset(
            dataset_dir,
            splits=(train_frac, val_frac, test_frac),
            seed=seed,
            stratify=stratify,
            mode=mode,
            group_by=group_by or None,
            background_label=background_label,
            metadata_filename=metadata_filename,
            verbose=not quiet,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if not quiet:
        split_str = ", ".join(f"{k}={v}" for k, v in sorted(result["splits"].items()))
        click.echo(
            f"Partitioned {result['groups']} group(s) into splits ({result['mode']}): {split_str}"
        )
        click.echo(f"Metadata: {result['metadata_file']}")


# `split` is an alias for `partition`.
dataset.add_command(dataset_partition, "split")


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
@click.option("--quiet", is_flag=True, help="Suppress progress output")
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
    quiet: bool,
) -> None:
    """Build a training-ready dataset: extract clips, partition, and write a manifest.

    Chains `extract-clips` (layout=both) -> `partition` (subdirs) -> `manifest`,
    producing a dataset directory + dataset.json consumable by `models ast train`.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    from bioamla.datasets import (
        build_manifest_from_metadata,
        extract_labeled_dataset,
        partition_dataset,
        save_dataset_manifest,
    )

    output_path = Path(output_dir)
    try:
        extract = extract_labeled_dataset(
            source=source,
            output_dir=output_dir,
            annotations=annotations,
            layout="both",
            padding_ms=padding_ms,
            bandpass=bandpass,
            target_sample_rate=sample_rate,
            include_labels=set(include) if include else None,
            exclude_labels=set(exclude) if exclude else None,
            min_duration=min_duration,
            verbose=not quiet,
        )

        kind = "labeled"
        partition_result = None
        if not no_partition:
            partition_result = partition_dataset(
                output_dir,
                splits=(train_frac, val_frac, test_frac),
                seed=seed,
                mode="subdirs",
                verbose=not quiet,
            )
            kind = "partitioned"

        manifest = build_manifest_from_metadata(
            output_dir,
            name=name or output_path.name,
            kind=kind,
            created=datetime.now(timezone.utc).isoformat(),
            sample_rate=sample_rate,
        )
        save_dataset_manifest(manifest, str(output_path / "dataset.json"))
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if not quiet:
        click.echo(f"Built dataset at {output_dir}")
        click.echo(f"  clips: {extract['clips_written']}  classes: {len(manifest.label2id)}")
        if partition_result:
            split_str = ", ".join(f"{k}={v}" for k, v in sorted(partition_result["splits"].items()))
            click.echo(f"  splits: {split_str}")
        click.echo(f"  manifest: {output_path / 'dataset.json'}")
        click.echo(f"Push to the Hub with: bioamla catalogs hf push-dataset {output_dir} <repo-id>")


@dataset.command("license")
@click.argument("path")
@click.option("--template", "-t", default=None, help="Template file to prepend to the license file")
@click.option("--output", "-o", default="LICENSE", help="Output filename for the license file")
@click.option("--metadata-filename", default="metadata.csv", help="Name of metadata CSV file")
@click.option(
    "--batch",
    is_flag=True,
    help="Process all datasets in directory (each subdirectory with metadata.csv)",
)
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def dataset_license(
    path: str, template: str, output: str, metadata_filename: str, batch: bool, quiet: bool
) -> None:
    """Generate license/attribution file from dataset metadata."""
    from pathlib import Path

    from bioamla.datasets import (
        generate_license_for_dataset,
        generate_licenses_for_directory,
    )

    path_obj = Path(path)
    template_path = Path(template) if template else None

    if template_path and not template_path.exists():
        click.echo(f"Error: Template file '{template}' not found.")
        raise SystemExit(1)

    if batch:
        if not path_obj.is_dir():
            click.echo(f"Error: Path '{path}' is not a directory.")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"Scanning directory for datasets: {path}")

        try:
            stats = generate_licenses_for_directory(
                audio_dir=path_obj,
                template_path=template_path,
                output_filename=output,
                metadata_filename=metadata_filename,
            )
        except BioamlaError as e:
            raise click.ClickException(str(e)) from e

        if stats["datasets_found"] == 0:
            click.echo("No datasets found (no directories with metadata.csv)")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"\nProcessed {stats['datasets_found']} dataset(s):")
            click.echo(f"  Successful: {stats['datasets_processed']}")
            click.echo(f"  Failed: {stats['datasets_failed']}")

            for item in stats["results"]:
                if item["status"] == "success":
                    click.echo(
                        f"  - {item['dataset_name']}: {item['attributions_count']} attributions"
                    )
                else:
                    click.echo(
                        f"  - {item['dataset_name']}: FAILED - {item.get('error', 'Unknown error')}"
                    )
        else:
            click.echo(f"Generated {stats['datasets_processed']} license files")

        if stats["datasets_failed"] > 0:
            raise SystemExit(1)

    else:
        if not path_obj.is_dir():
            click.echo(f"Error: Path '{path}' is not a directory.")
            raise SystemExit(1)

        csv_path = path_obj / metadata_filename
        if not csv_path.exists():
            click.echo(f"Error: Metadata file '{csv_path}' not found.")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"Generating license file for: {path}")

        try:
            stats = generate_license_for_dataset(
                dataset_path=path_obj,
                template_path=template_path,
                output_filename=output,
                metadata_filename=metadata_filename,
            )
        except BioamlaError as e:
            raise click.ClickException(str(e)) from e

        if not quiet:
            click.echo(f"License file generated: {stats['output_path']}")
            click.echo(f"  Attributions: {stats['attributions_count']}")
            click.echo(f"  File size: {stats['file_size']:,} bytes")
        else:
            click.echo(f"Generated {output} with {stats['attributions_count']} attributions")


def _parse_range(value: str) -> tuple[float, float]:
    """Parse a range string like '0.8-1.2' or '-2,2' into (min, max)."""
    if "-" in value and not value.startswith("-"):
        parts = value.split("-")
        return float(parts[0]), float(parts[1])
    elif "," in value:
        parts = value.split(",")
        return float(parts[0]), float(parts[1])
    else:
        val = float(value)
        return val, val


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
@click.option("--quiet", is_flag=True, help="Suppress progress output")
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
    """Augment audio files to expand training datasets."""
    from bioamla.datasets import AugmentationConfig, batch_augment

    noise_enabled = add_noise is not None
    noise_min_snr, noise_max_snr = _parse_range(add_noise) if add_noise else (3.0, 30.0)

    stretch_enabled = time_stretch is not None
    stretch_min, stretch_max = _parse_range(time_stretch) if time_stretch else (0.8, 1.2)

    pitch_enabled = pitch_shift is not None
    pitch_min, pitch_max = _parse_range(pitch_shift) if pitch_shift else (-2.0, 2.0)

    gain_enabled = gain is not None
    gain_min, gain_max = _parse_range(gain) if gain else (-12.0, 12.0)

    if not any([noise_enabled, stretch_enabled, pitch_enabled, gain_enabled]):
        click.echo("Error: At least one augmentation option must be specified")
        click.echo("Use --help for available options")
        raise SystemExit(1)

    config = AugmentationConfig(
        sample_rate=sample_rate,
        multiply=multiply,
        add_noise=noise_enabled,
        noise_min_snr=noise_min_snr,
        noise_max_snr=noise_max_snr,
        time_stretch=stretch_enabled,
        time_stretch_min=stretch_min,
        time_stretch_max=stretch_max,
        pitch_shift=pitch_enabled,
        pitch_shift_min=pitch_min,
        pitch_shift_max=pitch_max,
        gain=gain_enabled,
        gain_min_db=gain_min,
        gain_max_db=gain_max,
    )

    try:
        stats = batch_augment(
            input_dir=input_dir,
            output_dir=output,
            config=config,
            recursive=recursive,
            verbose=not quiet,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if quiet:
        click.echo(
            f"Created {stats['files_created']} augmented files from "
            f"{stats['files_processed']} source files in {stats['output_dir']}"
        )


@dataset.command("download")
@click.argument("url", required=True)
@click.argument("output_dir", required=False, default=".")
def dataset_download(url: str, output_dir: str) -> None:
    """Download a file from the specified URL to the target directory."""
    import os
    from urllib.parse import urlparse

    from bioamla.common.files import download_file

    if output_dir == ".":
        output_dir = os.getcwd()

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path) or "downloaded_file"
    output_path = os.path.join(output_dir, filename)

    try:
        download_file(url, output_path)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    except OSError as e:
        raise click.ClickException(f"Download failed: {e}") from e

    click.echo(f"Downloaded to {output_path}")


@dataset.command("unzip")
@click.argument("file_path")
@click.argument("output_path", required=False, default=".")
def dataset_unzip(file_path: str, output_path: str) -> None:
    """Extract a ZIP archive to the specified output directory."""
    import os

    from bioamla.common.files import extract_zip_file

    if output_path == ".":
        output_path = os.getcwd()

    try:
        extract_zip_file(file_path, output_path)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    except OSError as e:
        raise click.ClickException(f"Extraction failed: {e}") from e

    click.echo(f"Extracted to {output_path}")


@dataset.command("zip")
@click.argument("source_path")
@click.argument("output_file")
def dataset_zip(source_path: str, output_file: str) -> None:
    """Create a ZIP archive from a file or directory."""
    from pathlib import Path

    from bioamla.common.files import create_zip_file, zip_directory

    try:
        if Path(source_path).is_dir():
            zip_directory(source_path, output_file)
        else:
            create_zip_file([source_path], output_file)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    except OSError as e:
        raise click.ClickException(f"ZIP creation failed: {e}") from e

    click.echo(f"Created {output_file}")

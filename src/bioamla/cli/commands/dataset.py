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
    from bioamla.cli.output import emit_result
    from bioamla.datasets import merge_datasets

    configure_cli_logging()
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
    from bioamla.cli.output import emit_result
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
            source_metadata=source_metadata,
            verbose=not quiet,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

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
    from bioamla.cli.output import emit_result, json_mode, legacy_json
    from bioamla.datasets import get_dataset_stats

    try:
        stats = get_dataset_stats(dataset_dir, metadata_filename=metadata_filename)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

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
    from datetime import datetime, timezone
    from pathlib import Path

    from bioamla.cli.console import echo, print_success
    from bioamla.cli.output import emit_result
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

    if emit_result({"output": str(output_path), "manifest": manifest}):
        return

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
    from bioamla.cli.output import emit_result
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
    from datetime import datetime, timezone
    from pathlib import Path

    from bioamla.cli.console import echo, print_success
    from bioamla.cli.output import emit_result
    from bioamla.datasets import (
        build_manifest_from_metadata,
        extract_labeled_dataset,
        generate_license_for_dataset,
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
            source_metadata=source_metadata,
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

    # Non-fatal: write ATTRIBUTIONS.md when clips carry license/attribution.
    attributions_path = None
    if attributions:
        try:
            stats = generate_license_for_dataset(output_path, format="md")
            attributions_path = stats["output_path"]
        except BioamlaError:
            attributions_path = None  # no provenance to attribute — skip silently

    if emit_result(
        {
            "output_dir": output_dir,
            "extract": extract,
            "partition": partition_result,
            "manifest": manifest,
            "manifest_path": str(output_path / "dataset.json"),
            "attributions": attributions_path,
        }
    ):
        return

    if not quiet:
        print_success(f"Built dataset at {output_dir}")
        echo(f"  clips: {extract['clips_written']}  classes: {len(manifest.label2id)}")
        if partition_result:
            split_str = ", ".join(f"{k}={v}" for k, v in sorted(partition_result["splits"].items()))
            echo(f"  splits: {split_str}")
        echo(f"  manifest: {output_path / 'dataset.json'}")
        if attributions_path:
            echo(f"  attributions: {attributions_path}")
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
    from bioamla.cli.output import emit_result
    from bioamla.datasets import (
        generate_license_for_dataset,
        generate_licenses_for_directory,
    )

    path_obj = Path(path)
    template_path = Path(template) if template else None

    if template_path and not template_path.exists():
        raise click.ClickException(f"Template file '{template}' not found.")

    if batch:
        if not path_obj.is_dir():
            raise click.ClickException(f"Path '{path}' is not a directory.")

        if not quiet:
            echo(f"Scanning directory for datasets: {path}")

        try:
            stats = generate_licenses_for_directory(
                audio_dir=path_obj,
                template_path=template_path,
                output_filename=output,
                metadata_filename=metadata_filename,
                format=out_format,
            )
        except BioamlaError as e:
            raise click.ClickException(str(e)) from e

        if stats["datasets_found"] == 0:
            raise click.ClickException("No datasets found (no directories with metadata.csv)")
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

    else:
        if not path_obj.is_dir():
            raise click.ClickException(f"Path '{path}' is not a directory.")

        csv_path = path_obj / metadata_filename
        if not csv_path.exists():
            raise click.ClickException(f"Metadata file '{csv_path}' not found.")

        if not quiet:
            echo(f"Generating license file for: {path}")

        try:
            stats = generate_license_for_dataset(
                dataset_path=path_obj,
                template_path=template_path,
                output_filename=output,
                metadata_filename=metadata_filename,
                format=out_format,
            )
        except BioamlaError as e:
            raise click.ClickException(str(e)) from e

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
    from bioamla.cli.output import emit_result
    from bioamla.datasets import WaveformAugmentationConfig, batch_augment

    noise_enabled = add_noise is not None
    noise_min_snr, noise_max_snr = _parse_range(add_noise) if add_noise else (3.0, 30.0)

    stretch_enabled = time_stretch is not None
    stretch_min, stretch_max = _parse_range(time_stretch) if time_stretch else (0.8, 1.2)

    pitch_enabled = pitch_shift is not None
    pitch_min, pitch_max = _parse_range(pitch_shift) if pitch_shift else (-2.0, 2.0)

    gain_enabled = gain is not None
    gain_min, gain_max = _parse_range(gain) if gain else (-12.0, 12.0)

    if not any([noise_enabled, stretch_enabled, pitch_enabled, gain_enabled]):
        raise click.ClickException(
            "At least one augmentation option must be specified (see --help)"
        )

    config = WaveformAugmentationConfig(
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

    configure_cli_logging()
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
    from dataclasses import replace

    from bioamla.cli.console import print_success
    from bioamla.cli.output import emit_result
    from bioamla.datasets import RemixRecipe, remix_dataset

    if keep and drop:
        raise click.UsageError("Use --keep or --drop, not both")
    if not recipe and not source:
        raise click.UsageError("Give --recipe and/or --source")

    overrides: dict = {}
    if source:
        overrides["source"] = source
    if keep:
        overrides.update(keep=list(keep), drop=None)
    if drop:
        overrides.update(drop=list(drop), keep=None)
    if name:
        overrides["name"] = name
    if label_column:
        overrides["label_column"] = label_column

    try:
        if recipe:
            spec = replace(RemixRecipe.from_toml(recipe), **overrides)
        else:
            spec = RemixRecipe(**overrides)
        stats = remix_dataset(
            spec,
            output_dir,
            metadata_filename=metadata_filename,
            license_template=template,
            verbose=not quiet,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

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

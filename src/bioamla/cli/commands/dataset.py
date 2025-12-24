"""Dataset management commands."""

import click

from bioamla.core.files import TextFile


@click.group()
def dataset():
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
):
    """Merge multiple audio datasets into a single dataset."""
    from bioamla.core.datasets import merge_datasets as do_merge

    stats = do_merge(
        dataset_paths=list(dataset_paths),
        output_dir=output_dir,
        metadata_filename=metadata_filename,
        skip_existing=not overwrite,
        organize_by_category=not no_organize,
        target_format=target_format,
        verbose=not quiet,
    )

    if quiet:
        msg = f"Merged {stats['datasets_merged']} datasets: {stats['total_files']} total files"
        if target_format:
            msg += f", {stats['files_converted']} converted"
        click.echo(msg)


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
):
    """Generate license/attribution file from dataset metadata."""
    from pathlib import Path as PathLib

    from bioamla.core.license import (
        generate_license_for_dataset,
        generate_licenses_for_directory,
    )

    path_obj = PathLib(path)
    template_path = PathLib(template) if template else None

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
        except FileNotFoundError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1) from e

        if stats["datasets_found"] == 0:
            click.echo("No datasets found (no directories with metadata.csv)")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"\nProcessed {stats['datasets_found']} dataset(s):")
            click.echo(f"  Successful: {stats['datasets_processed']}")
            click.echo(f"  Failed: {stats['datasets_failed']}")

            for result in stats["results"]:
                if result["status"] == "success":
                    click.echo(
                        f"  - {result['dataset_name']}: {result['attributions_count']} attributions"
                    )
                else:
                    click.echo(
                        f"  - {result['dataset_name']}: FAILED - {result.get('error', 'Unknown error')}"
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
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1) from e

        if not quiet:
            click.echo(f"License file generated: {stats['output_path']}")
            click.echo(f"  Attributions: {stats['attributions_count']}")
            click.echo(f"  File size: {stats['file_size']:,} bytes")
        else:
            click.echo(f"Generated {output} with {stats['attributions_count']} attributions")


def _parse_range(value: str) -> tuple:
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
):
    """Augment audio files to expand training datasets."""
    from bioamla.core.augment import AugmentationConfig, batch_augment

    config = AugmentationConfig(
        sample_rate=sample_rate,
        multiply=multiply,
    )

    if add_noise:
        config.add_noise = True
        min_snr, max_snr = _parse_range(add_noise)
        config.noise_min_snr = min_snr
        config.noise_max_snr = max_snr

    if time_stretch:
        config.time_stretch = True
        min_rate, max_rate = _parse_range(time_stretch)
        config.time_stretch_min = min_rate
        config.time_stretch_max = max_rate

    if pitch_shift:
        config.pitch_shift = True
        min_semi, max_semi = _parse_range(pitch_shift)
        config.pitch_shift_min = min_semi
        config.pitch_shift_max = max_semi

    if gain:
        config.gain = True
        min_db, max_db = _parse_range(gain)
        config.gain_min_db = min_db
        config.gain_max_db = max_db

    if not any([config.add_noise, config.time_stretch, config.pitch_shift, config.gain]):
        click.echo("Error: At least one augmentation option must be specified")
        click.echo("Use --help for available options")
        raise SystemExit(1)

    try:
        stats = batch_augment(
            input_dir=input_dir,
            output_dir=output,
            config=config,
            recursive=recursive,
            verbose=not quiet,
        )

        if quiet:
            click.echo(
                f"Created {stats['files_created']} augmented files from "
                f"{stats['files_processed']} source files in {stats['output_dir']}"
            )
    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        click.echo(f"Error during augmentation: {e}")
        raise SystemExit(1) from e


@dataset.command("download")
@click.argument("url", required=True)
@click.argument("output_dir", required=False, default=".")
def dataset_download(url: str, output_dir: str):
    """Download a file from the specified URL to the target directory."""
    import os
    from urllib.parse import urlparse

    from bioamla.core.utils import download_file

    if output_dir == ".":
        output_dir = os.getcwd()

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "downloaded_file"

    output_path = os.path.join(output_dir, filename)
    download_file(url, output_path)


@dataset.command("unzip")
@click.argument("file_path")
@click.argument("output_path", required=False, default=".")
def dataset_unzip(file_path: str, output_path: str):
    """Extract a ZIP archive to the specified output directory."""
    import os

    from bioamla.core.utils import extract_zip_file

    if output_path == ".":
        output_path = os.getcwd()

    extract_zip_file(file_path, output_path)


@dataset.command("zip")
@click.argument("source_path")
@click.argument("output_file")
def dataset_zip(source_path: str, output_file: str):
    """Create a ZIP archive from a file or directory."""
    import os

    from bioamla.core.utils import create_zip_file, zip_directory

    if os.path.isdir(source_path):
        zip_directory(source_path, output_file)
    else:
        create_zip_file([source_path], output_file)

    click.echo(f"Created {output_file}")

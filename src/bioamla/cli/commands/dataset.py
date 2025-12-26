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
    from bioamla.cli.service_helpers import handle_result, services

    result = services.dataset.merge(
        dataset_paths=list(dataset_paths),
        output_dir=output_dir,
        metadata_filename=metadata_filename,
        skip_existing=not overwrite,
        organize_by_category=not no_organize,
        target_format=target_format,
        verbose=not quiet,
    )
    stats = handle_result(result)

    if quiet:
        msg = f"Merged {stats.datasets_merged} datasets: {stats.total_files} total files"
        if target_format:
            msg += f", {stats.files_converted} converted"
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
) -> None:
    """Generate license/attribution file from dataset metadata."""
    from pathlib import Path as PathLib

    from bioamla.cli.service_helpers import handle_result, services

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

        result = services.dataset.generate_licenses_batch(
            directory=str(path_obj),
            template_path=str(template_path) if template_path else None,
            output_filename=output,
            metadata_filename=metadata_filename,
        )
        stats = handle_result(result)

        if stats.datasets_found == 0:
            click.echo("No datasets found (no directories with metadata.csv)")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"\nProcessed {stats.datasets_found} dataset(s):")
            click.echo(f"  Successful: {stats.datasets_processed}")
            click.echo(f"  Failed: {stats.datasets_failed}")

            for item in stats.results:
                if item["status"] == "success":
                    click.echo(
                        f"  - {item['dataset_name']}: {item['attributions_count']} attributions"
                    )
                else:
                    click.echo(
                        f"  - {item['dataset_name']}: FAILED - {item.get('error', 'Unknown error')}"
                    )
        else:
            click.echo(f"Generated {stats.datasets_processed} license files")

        if stats.datasets_failed > 0:
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

        result = services.dataset.generate_license(
            dataset_path=str(path_obj),
            template_path=str(template_path) if template_path else None,
            output_filename=output,
            metadata_filename=metadata_filename,
        )
        stats = handle_result(result)

        if not quiet:
            click.echo(f"License file generated: {stats.output_path}")
            click.echo(f"  Attributions: {stats.attributions_count}")
            click.echo(f"  File size: {stats.file_size:,} bytes")
        else:
            click.echo(f"Generated {output} with {stats.attributions_count} attributions")


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
    from bioamla.cli.service_helpers import handle_result, services

    # Parse augmentation parameters
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

    result = services.dataset.augment(
        input_dir=input_dir,
        output_dir=output,
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
        multiply=multiply,
        sample_rate=sample_rate,
        recursive=recursive,
        verbose=not quiet,
    )
    stats = handle_result(result)

    if quiet:
        click.echo(
            f"Created {stats.files_created} augmented files from "
            f"{stats.files_processed} source files in {stats.output_dir}"
        )


@dataset.command("download")
@click.argument("url", required=True)
@click.argument("output_dir", required=False, default=".")
def dataset_download(url: str, output_dir: str) -> None:
    """Download a file from the specified URL to the target directory."""
    import os
    from urllib.parse import urlparse

    from bioamla.cli.service_helpers import handle_result, services

    if output_dir == ".":
        output_dir = os.getcwd()

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "downloaded_file"

    output_path = os.path.join(output_dir, filename)

    result = services.dataset.download(url, output_path)
    handle_result(result)


@dataset.command("unzip")
@click.argument("file_path")
@click.argument("output_path", required=False, default=".")
def dataset_unzip(file_path: str, output_path: str) -> None:
    """Extract a ZIP archive to the specified output directory."""
    import os

    from bioamla.cli.service_helpers import handle_result, services

    if output_path == ".":
        output_path = os.getcwd()

    result = services.dataset.extract_zip(file_path, output_path)
    handle_result(result)


@dataset.command("zip")
@click.argument("source_path")
@click.argument("output_file")
def dataset_zip(source_path: str, output_file: str) -> None:
    """Create a ZIP archive from a file or directory."""
    from bioamla.cli.service_helpers import handle_result, services

    result = services.dataset.create_zip(source_path, output_file)
    handle_result(result)

    click.echo(f"Created {output_file}")

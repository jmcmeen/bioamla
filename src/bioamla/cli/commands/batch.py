"""Batch processing operations."""

import click


@click.group()
def batch() -> None:
    """Batch processing operations for multiple files."""
    pass


@batch.group()
def audio() -> None:
    """Batch audio processing operations."""
    pass


@audio.command("convert")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--sample-rate", "-r", default=None, type=int, help="Target sample rate")
@click.option("--channels", "-c", default=None, type=int, help="Target number of channels")
@click.option(
    "--format",
    "-f",
    default="wav",
    type=click.Choice(["wav", "mp3", "flac", "ogg"]),
    help="Output format",
)
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_convert(input_dir: str, output_dir: str, sample_rate: int, channels: int, format: str, recursive: bool, quiet: bool) -> None:
    """Batch convert audio files in a directory."""
    from bioamla.repository.local import LocalFileRepository
    from bioamla.services.batch import BatchService

    if not quiet:
        click.echo(f"Converting audio files from {input_dir} to {output_dir}...")

    controller = BatchService()
    result = controller.audio_convert(
        input_dir=input_dir,
        output_dir=output_dir,
        format=format,
        sample_rate=sample_rate,
        channels=channels,
        recursive=recursive,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    data = result.data
    click.echo(f"Converted {data.files_processed} files, {data.files_failed} errors")

    if data.errors and not quiet:
        for error in data.errors[:5]:  # Show first 5 errors
            click.echo(f"  {error}")
        if len(data.errors) > 5:
            click.echo(f"  ... and {len(data.errors) - 5} more errors")

"""Audio file operations (info, convert, segment, visualize)."""

import click


@click.group()
def audio() -> None:
    """Audio file operations (info, convert, segment, visualize)."""
    pass


@audio.command("info")
@click.argument("path")
def audio_info(path: str) -> None:
    """Display audio file information."""
    from bioamla.repository.local import LocalFileRepository
    from bioamla.services.audio_file import AudioFileService

    repository = LocalFileRepository()
    controller = AudioFileService(file_repository=repository)
    result = controller.open(path)

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    audio_data = result.data
    click.echo(f"File: {path}")
    click.echo(f"Duration: {audio_data.duration:.2f}s")
    click.echo(f"Sample rate: {audio_data.sample_rate} Hz")
    click.echo(f"Channels: {audio_data.channels}")
    click.echo(f"Samples: {audio_data.num_samples}")


@audio.command("list")
@click.argument("path")
@click.option("--recursive/--no-recursive", "-r", default=True, help="Search subdirectories (default: recursive)")
def audio_list(path: str, recursive: bool) -> None:
    """List audio files in a directory."""
    from bioamla.repository.local import LocalFileRepository
    from bioamla.services.audio_transform import AudioTransformService

    repository = LocalFileRepository()
    result = AudioTransformService(file_repository=repository).list_files(path, recursive=recursive)

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    audio_files = result.data
    if not audio_files:
        click.echo("No audio files found")
        return

    click.echo(f"Found {len(audio_files)} audio file(s):")
    for f in audio_files:
        click.echo(f"  {f}")


@audio.command("convert")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--sample-rate", "-r", default=None, type=int, help="Target sample rate")
@click.option("--channels", "-c", default=None, type=int, help="Target number of channels")
@click.option("--bit-depth", "-b", default=None, type=int, help="Target bit depth")
@click.option(
    "--format",
    "-f",
    default=None,
    type=click.Choice(["wav", "mp3", "flac", "ogg"]),
    help="Output format",
)
def audio_convert(input_path: str, output_path: str, sample_rate: int, channels: int, bit_depth: int, format: str) -> None:
    """Convert audio file format or properties."""
    from bioamla.repository.local import LocalFileRepository
    from bioamla.services.audio_file import AudioFileService

    repository = LocalFileRepository()
    controller = AudioFileService(file_repository=repository)

    # Load the audio file
    open_result = controller.open(input_path)
    if not open_result.success:
        click.echo(f"Error: {open_result.error}")
        raise SystemExit(1)

    audio_data = open_result.data

    # Handle channel conversion if requested
    if channels is not None and channels != audio_data.channels:
        import numpy as np

        if channels == 1 and audio_data.channels == 2:
            # Stereo to mono: average channels
            if audio_data.samples.ndim == 2:
                audio_data.samples = audio_data.samples.mean(axis=1)
            audio_data.channels = 1
        elif channels == 2 and audio_data.channels == 1:
            # Mono to stereo: duplicate channel
            audio_data.samples = np.column_stack([audio_data.samples, audio_data.samples])
            audio_data.channels = 2

    # Save with optional resampling and format conversion
    result = controller.save_as(
        audio_data=audio_data,
        output_path=output_path,
        target_sample_rate=sample_rate,
        format=format,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    click.echo(f"Converted: {output_path}")


@audio.command("segment")
@click.argument("input_path")
@click.argument("output_dir")
@click.option(
    "--duration", "-d", default=3.0, type=float, help="Segment duration in seconds (default: 3.0)"
)
@click.option(
    "--overlap",
    "-o",
    default=0.0,
    type=float,
    help="Overlap between segments in seconds (default: 0.0)",
)
@click.option(
    "--format",
    "-f",
    default="wav",
    type=click.Choice(["wav", "mp3", "flac", "ogg"]),
    help="Output format (default: wav)",
)
@click.option(
    "--prefix", "-p", default=None, help="Prefix for output filenames (default: input filename)"
)
def audio_segment(input_path: str, output_dir: str, duration: float, overlap: float, format: str, prefix: str) -> None:
    """Segment audio file into fixed-duration clips."""
    from bioamla.repository.local import LocalFileRepository
    from bioamla.services.audio_transform import AudioTransformService

    repository = LocalFileRepository()
    result = AudioTransformService(file_repository=repository).segment_file(
        input_path=input_path,
        output_dir=output_dir,
        duration=duration,
        overlap=overlap,
        format=format,
        prefix=prefix,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    click.echo(f"Created {result.data.processed} segments in {output_dir}")


@audio.command("trim")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--start", "-s", default=0.0, type=float, help="Start time in seconds")
@click.option("--end", "-e", default=None, type=float, help="End time in seconds")
@click.option("--duration", "-d", default=None, type=float, help="Duration in seconds")
def audio_trim(input_path: str, output_path: str, start: float, end: float, duration: float) -> None:
    """Trim audio file to specified time range."""
    from bioamla.repository.local import LocalFileRepository
    from bioamla.services.audio_transform import AudioTransformService

    if end is not None and duration is not None:
        click.echo("Error: Cannot specify both --end and --duration")
        raise SystemExit(1)

    # Calculate end time from duration if specified
    if duration is not None:
        end = start + duration

    repository = LocalFileRepository()
    controller = AudioTransformService(file_repository=repository)
    result = controller.trim_file(
        input_path=input_path,
        output_path=output_path,
        start=start if start != 0.0 else None,
        end=end,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    click.echo(f"Trimmed audio saved to: {output_path}")


@audio.command("normalize")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--target-db", "-t", default=-3.0, type=float, help="Target peak level in dB")
@click.option("--method", "-m", type=click.Choice(["peak", "rms"]), default="peak", help="Method")
def audio_normalize(input_path: str, output_path: str, target_db: float, method: str) -> None:
    """Normalize audio amplitude."""
    from bioamla.repository.local import LocalFileRepository
    from bioamla.services.audio_transform import AudioTransformService

    repository = LocalFileRepository()
    result = AudioTransformService(file_repository=repository).normalize_file(
        input_path=input_path,
        output_path=output_path,
        target_db=target_db,
        peak=(method == "peak"),
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    click.echo(f"Normalized audio saved to: {output_path}")


@audio.command("resample")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--sample-rate", "-r", required=True, type=int, help="Target sample rate in Hz")
def audio_resample(input_path: str, output_path: str, sample_rate: int) -> None:
    """Resample audio to a different sample rate."""
    from bioamla.repository.local import LocalFileRepository
    from bioamla.services.audio_transform import AudioTransformService

    repository = LocalFileRepository()
    result = AudioTransformService(file_repository=repository).resample_file(
        input_path=input_path,
        output_path=output_path,
        target_rate=sample_rate,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    click.echo(f"Resampled audio saved to: {output_path}")


@audio.command("visualize")
@click.argument("path")
@click.option("--output", "-o", default=None, help="Output image file path")
@click.option(
    "--type", "-t",
    "viz_type",
    default="mel",
    type=click.Choice(["mel", "stft", "mfcc", "waveform"]),
    help="Visualization type",
)
@click.option("--n-fft", default=2048, type=int, help="FFT window size")
@click.option("--hop-length", default=512, type=int, help="Hop length")
@click.option("--n-mels", default=128, type=int, help="Number of mel bands")
@click.option("--n-mfcc", default=20, type=int, help="Number of MFCCs")
@click.option("--cmap", default="viridis", help="Colormap name")
@click.option("--dpi", default=100, type=int, help="Output DPI")
def audio_visualize(path: str, output: str, viz_type: str, n_fft: int, hop_length: int, n_mels: int, n_mfcc: int, cmap: str, dpi: int) -> None:
    """Generate audio visualization (spectrogram, waveform, MFCC) for a single file."""
    from pathlib import Path

    from bioamla.repository.local import LocalFileRepository
    from bioamla.services.audio_transform import AudioTransformService

    repository = LocalFileRepository()
    output_path = output or f"{Path(path).stem}_{viz_type}.png"
    result = AudioTransformService(file_repository=repository).visualize_file(
        input_path=path,
        output_path=output_path,
        viz_type=viz_type,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        cmap=cmap,
        dpi=dpi,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    click.echo(f"Visualization saved to: {output_path}")



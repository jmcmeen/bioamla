"""Audio file operations (info, convert, segment, visualize)."""

import click


@click.group()
def audio():
    """Audio file operations (info, convert, segment, visualize)."""
    pass


@audio.command("info")
@click.argument("path")
def audio_info(path: str):
    """Display audio file information."""
    from bioamla.services.audio_file import AudioFileController

    controller = AudioFileController()
    result = controller.open(path)

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    audio_data = result.data
    click.echo(f"File: {audio_data.path}")
    click.echo(f"Duration: {audio_data.duration:.2f}s")
    click.echo(f"Sample rate: {audio_data.sample_rate} Hz")
    click.echo(f"Channels: {audio_data.channels}")
    click.echo(f"Samples: {audio_data.samples}")
    click.echo(f"Format: {audio_data.format}")


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
def audio_convert(input_path, output_path, sample_rate, channels, bit_depth, format):
    """Convert audio file format or properties."""
    from bioamla.services.audio_file import AudioFileController

    controller = AudioFileController()
    result = controller.convert(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=sample_rate,
        target_channels=channels,
        target_bit_depth=bit_depth,
        target_format=format,
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
def audio_segment(input_path, output_dir, duration, overlap, format, prefix):
    """Segment audio file into fixed-duration clips."""
    from bioamla.services.audio_file import AudioFileController

    controller = AudioFileController()
    result = controller.segment(
        input_path=input_path,
        output_dir=output_dir,
        segment_duration=duration,
        overlap=overlap,
        output_format=format,
        prefix=prefix,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    click.echo(f"Created {result.data.segments_created} segments in {output_dir}")


@audio.command("trim")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--start", "-s", default=0.0, type=float, help="Start time in seconds")
@click.option("--end", "-e", default=None, type=float, help="End time in seconds")
@click.option("--duration", "-d", default=None, type=float, help="Duration in seconds")
def audio_trim(input_path, output_path, start, end, duration):
    """Trim audio file to specified time range."""
    from bioamla.services.audio_file import AudioFileController

    if end is not None and duration is not None:
        click.echo("Error: Cannot specify both --end and --duration")
        raise SystemExit(1)

    controller = AudioFileController()
    result = controller.trim(
        input_path=input_path,
        output_path=output_path,
        start_time=start,
        end_time=end,
        duration=duration,
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
def audio_normalize(input_path, output_path, target_db, method):
    """Normalize audio amplitude."""
    from bioamla.services.audio_file import AudioFileController

    controller = AudioFileController()
    result = controller.normalize(
        input_path=input_path,
        output_path=output_path,
        target_db=target_db,
        method=method,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    click.echo(f"Normalized audio saved to: {output_path}")


@audio.command("resample")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--sample-rate", "-r", required=True, type=int, help="Target sample rate in Hz")
def audio_resample(input_path, output_path, sample_rate):
    """Resample audio to a different sample rate."""
    from bioamla.services.audio_file import AudioFileController

    controller = AudioFileController()
    result = controller.resample(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=sample_rate,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    click.echo(f"Resampled audio saved to: {output_path}")


@audio.command("spectrogram")
@click.argument("path")
@click.option("--output", "-o", default=None, help="Output image file path")
@click.option("--width", "-w", default=800, type=int, help="Image width in pixels")
@click.option("--height", "-h", default=400, type=int, help="Image height in pixels")
@click.option("--n-fft", default=2048, type=int, help="FFT window size")
@click.option("--hop-length", default=512, type=int, help="Hop length")
@click.option("--colormap", default="viridis", help="Colormap name")
@click.option("--mel/--no-mel", default=True, help="Use mel spectrogram")
@click.option("--n-mels", default=128, type=int, help="Number of mel bands")
@click.option("--fmin", default=0.0, type=float, help="Minimum frequency")
@click.option("--fmax", default=None, type=float, help="Maximum frequency")
@click.option("--db/--linear", default=True, help="Use decibel scale")
@click.option("--show", is_flag=True, help="Display the plot interactively")
def audio_spectrogram(
    path, output, width, height, n_fft, hop_length, colormap, mel, n_mels, fmin, fmax, db, show
):
    """Generate spectrogram visualization."""
    from bioamla.services.audio_file import AudioFileController
    from bioamla.services.visualize import VisualizeController

    audio_ctrl = AudioFileController()
    result = audio_ctrl.open(path)

    if not result.success:
        click.echo(f"Error loading audio: {result.error}")
        raise SystemExit(1)

    vis_ctrl = VisualizeController()
    vis_result = vis_ctrl.spectrogram(
        audio_data=result.data,
        output_path=output,
        width=width,
        height=height,
        n_fft=n_fft,
        hop_length=hop_length,
        colormap=colormap,
        use_mel=mel,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        use_db=db,
        show=show,
    )

    if not vis_result.success:
        click.echo(f"Error generating spectrogram: {vis_result.error}")
        raise SystemExit(1)

    if output:
        click.echo(f"Spectrogram saved to: {output}")
    elif show:
        click.echo("Displaying spectrogram...")


@audio.command("waveform")
@click.argument("path")
@click.option("--output", "-o", default=None, help="Output image file path")
@click.option("--width", "-w", default=800, type=int, help="Image width in pixels")
@click.option("--height", "-h", default=200, type=int, help="Image height in pixels")
@click.option("--color", default="blue", help="Waveform color")
@click.option("--show", is_flag=True, help="Display the plot interactively")
def audio_waveform(path, output, width, height, color, show):
    """Generate waveform visualization."""
    from bioamla.services.audio_file import AudioFileController
    from bioamla.services.visualize import VisualizeController

    audio_ctrl = AudioFileController()
    result = audio_ctrl.open(path)

    if not result.success:
        click.echo(f"Error loading audio: {result.error}")
        raise SystemExit(1)

    vis_ctrl = VisualizeController()
    vis_result = vis_ctrl.waveform(
        audio_data=result.data,
        output_path=output,
        width=width,
        height=height,
        color=color,
        show=show,
    )

    if not vis_result.success:
        click.echo(f"Error generating waveform: {vis_result.error}")
        raise SystemExit(1)

    if output:
        click.echo(f"Waveform saved to: {output}")
    elif show:
        click.echo("Displaying waveform...")


@audio.command("batch-convert")
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
def audio_batch_convert(input_dir, output_dir, sample_rate, channels, format, recursive, quiet):
    """Batch convert audio files in a directory."""
    from pathlib import Path

    from bioamla.core.progress import ProgressBar
    from bioamla.core.utils import get_audio_files
    from bioamla.services.audio_file import AudioFileController

    audio_files = get_audio_files(input_dir, recursive=recursive)
    if not audio_files:
        click.echo("No audio files found")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    controller = AudioFileController()
    success_count = 0
    error_count = 0

    with ProgressBar(total=len(audio_files), description="Converting files") as progress:
        for filepath in audio_files:
            input_path = Path(filepath)
            rel_path = input_path.relative_to(input_dir) if recursive else input_path.name
            output_path = Path(output_dir) / rel_path.with_suffix(f".{format}")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            result = controller.convert(
                input_path=str(input_path),
                output_path=str(output_path),
                target_sample_rate=sample_rate,
                target_channels=channels,
                target_format=format,
            )

            if result.success:
                success_count += 1
            else:
                error_count += 1
                if not quiet:
                    click.echo(f"Error: {filepath} - {result.error}")

            progress.advance()

    click.echo(f"Converted {success_count} files, {error_count} errors")

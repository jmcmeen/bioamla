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
    from bioamla.cli.service_helpers import handle_result, services

    result = services.audio_file.open(path)
    audio_data = handle_result(result)

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
    from bioamla.cli.service_helpers import handle_result, services

    result = services.audio_transform.list_files(path, recursive=recursive)
    audio_files = handle_result(result)

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
    from bioamla.cli.service_helpers import check_result, handle_result, services

    # Load the audio file
    audio_data = handle_result(services.audio_file.open(input_path))

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
    result = services.audio_file.save_as(
        audio_data=audio_data,
        output_path=output_path,
        target_sample_rate=sample_rate,
        format=format,
    )
    check_result(result)

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
    from bioamla.cli.service_helpers import handle_result, services

    result = services.audio_transform.segment_file(
        input_path=input_path,
        output_dir=output_dir,
        duration=duration,
        overlap=overlap,
        format=format,
        prefix=prefix,
    )
    batch_result = handle_result(result)

    click.echo(f"Created {batch_result.processed} segments in {output_dir}")


@audio.command("trim")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--start", "-s", default=0.0, type=float, help="Start time in seconds")
@click.option("--end", "-e", default=None, type=float, help="End time in seconds")
@click.option("--duration", "-d", default=None, type=float, help="Duration in seconds")
def audio_trim(input_path: str, output_path: str, start: float, end: float, duration: float) -> None:
    """Trim audio file to specified time range."""
    from bioamla.cli.service_helpers import check_result, exit_with_error, services

    if end is not None and duration is not None:
        exit_with_error("Cannot specify both --end and --duration")

    # Calculate end time from duration if specified
    if duration is not None:
        end = start + duration

    result = services.audio_transform.trim_file(
        input_path=input_path,
        output_path=output_path,
        start=start if start != 0.0 else None,
        end=end,
    )
    check_result(result)

    click.echo(f"Trimmed audio saved to: {output_path}")


@audio.command("normalize")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--target-db", "-t", default=-3.0, type=float, help="Target peak level in dB")
@click.option("--method", "-m", type=click.Choice(["peak", "rms"]), default="peak", help="Method")
def audio_normalize(input_path: str, output_path: str, target_db: float, method: str) -> None:
    """Normalize audio amplitude."""
    from bioamla.cli.service_helpers import check_result, services

    result = services.audio_transform.normalize_file(
        input_path=input_path,
        output_path=output_path,
        target_db=target_db,
        peak=(method == "peak"),
    )
    check_result(result)

    click.echo(f"Normalized audio saved to: {output_path}")


@audio.command("resample")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--sample-rate", "-r", required=True, type=int, help="Target sample rate in Hz")
def audio_resample(input_path: str, output_path: str, sample_rate: int) -> None:
    """Resample audio to a different sample rate."""
    from bioamla.cli.service_helpers import check_result, services

    result = services.audio_transform.resample_file(
        input_path=input_path,
        output_path=output_path,
        target_rate=sample_rate,
    )
    check_result(result)

    click.echo(f"Resampled audio saved to: {output_path}")


@audio.command("filter")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--lowpass", default=None, type=float, help="Lowpass cutoff frequency in Hz")
@click.option("--highpass", default=None, type=float, help="Highpass cutoff frequency in Hz")
@click.option("--bandpass-low", default=None, type=float, help="Bandpass low frequency in Hz")
@click.option("--bandpass-high", default=None, type=float, help="Bandpass high frequency in Hz")
@click.option("--order", default=5, type=int, help="Filter order (default: 5)")
def audio_filter(input_path: str, output_path: str, lowpass: float, highpass: float, bandpass_low: float, bandpass_high: float, order: int) -> None:
    """Apply frequency filter to audio file."""
    from bioamla.cli.service_helpers import check_result, services

    # Validate filter options
    if not any([lowpass, highpass, bandpass_low]):
        click.echo("Error: Must specify --lowpass, --highpass, or --bandpass-low/--bandpass-high")
        return

    if (bandpass_low is not None) != (bandpass_high is not None):
        click.echo("Error: Both --bandpass-low and --bandpass-high must be specified together")
        return

    bandpass = (bandpass_low, bandpass_high) if bandpass_low is not None else None

    result = services.audio_transform.filter_file(
        input_path=input_path,
        output_path=output_path,
        lowpass=lowpass,
        highpass=highpass,
        bandpass=bandpass,
        order=order,
    )
    check_result(result)

    if bandpass:
        click.echo(f"Applied bandpass filter ({bandpass[0]}-{bandpass[1]} Hz) to: {output_path}")
    elif lowpass:
        click.echo(f"Applied lowpass filter ({lowpass} Hz) to: {output_path}")
    else:
        click.echo(f"Applied highpass filter ({highpass} Hz) to: {output_path}")


@audio.command("denoise")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--strength", default=1.0, type=float, help="Noise reduction strength (0-2, default: 1.0)")
def audio_denoise(input_path: str, output_path: str, strength: float) -> None:
    """Apply spectral noise reduction to audio file."""
    from bioamla.cli.service_helpers import check_result, services

    result = services.audio_transform.denoise_file(
        input_path=input_path,
        output_path=output_path,
        strength=strength,
    )
    check_result(result)

    click.echo(f"Denoised audio saved to: {output_path}")


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
@click.option("--legend/--no-legend", default=True, help="Show axes, title, and colorbar (default: True)")
def audio_visualize(path: str, output: str, viz_type: str, n_fft: int, hop_length: int, n_mels: int, n_mfcc: int, cmap: str, dpi: int, legend: bool) -> None:
    """Generate audio visualization (spectrogram, waveform, MFCC) for a single file."""
    from pathlib import Path

    from bioamla.cli.service_helpers import check_result, services

    output_path = output or f"{Path(path).stem}_{viz_type}.png"
    result = services.audio_transform.visualize_file(
        input_path=path,
        output_path=output_path,
        viz_type=viz_type,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        cmap=cmap,
        dpi=dpi,
        show_legend=legend,
    )
    check_result(result)

    click.echo(f"Visualization saved to: {output_path}")

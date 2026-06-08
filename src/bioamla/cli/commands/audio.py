"""Audio file operations (info, convert, segment, visualize)."""

from pathlib import Path

import click

from bioamla.exceptions import BioamlaError


@click.group()
def audio() -> None:
    """Audio file operations (info, convert, segment, visualize)."""
    pass


@audio.command("info")
@click.argument("path")
def audio_info(path: str) -> None:
    """Display audio file information."""
    from bioamla.audio import load_audio_data

    try:
        audio_data = load_audio_data(path)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"File: {path}")
    click.echo(f"Duration: {audio_data.duration:.2f}s")
    click.echo(f"Sample rate: {audio_data.sample_rate} Hz")
    click.echo(f"Channels: {audio_data.channels}")
    click.echo(f"Samples: {audio_data.num_samples}")


@audio.command("list")
@click.argument("path")
@click.option(
    "--recursive/--no-recursive",
    "-r",
    default=True,
    help="Search subdirectories (default: recursive)",
)
def audio_list(path: str, recursive: bool) -> None:
    """List audio files in a directory."""
    from bioamla.audio import list_audio_files

    try:
        audio_files = list_audio_files(path, recursive=recursive)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

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
def audio_convert(
    input_path: str,
    output_path: str,
    sample_rate: int,
    channels: int,
    bit_depth: int,
    format: str,
) -> None:
    """Convert audio file format or properties."""
    import numpy as np

    from bioamla.audio import load_audio_data, save_audio_data_as

    try:
        audio_data = load_audio_data(input_path)

        # Handle channel conversion if requested
        if channels is not None and channels != audio_data.channels:
            if channels == 1 and audio_data.channels == 2:
                # Stereo to mono: average channels
                if audio_data.samples.ndim == 2:
                    audio_data.samples = audio_data.samples.mean(axis=1)
                audio_data.channels = 1
            elif channels == 2 and audio_data.channels == 1:
                # Mono to stereo: duplicate channel
                audio_data.samples = np.column_stack(
                    [audio_data.samples, audio_data.samples]
                )
                audio_data.channels = 2

        save_audio_data_as(
            audio_data,
            output_path,
            target_sample_rate=sample_rate,
            format=format,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

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
def audio_segment(
    input_path: str,
    output_dir: str,
    duration: float,
    overlap: float,
    format: str,
    prefix: str,
) -> None:
    """Segment audio file into fixed-duration clips."""
    from bioamla.audio import load_audio, save_audio
    from bioamla.exceptions import InvalidInputError

    try:
        audio, sr = load_audio(input_path)

        segment_samples = int(duration * sr)
        overlap_samples = int(overlap * sr)
        step_samples = segment_samples - overlap_samples

        if step_samples <= 0:
            raise InvalidInputError("Overlap must be less than duration")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            prefix = Path(input_path).stem

        segments_created = 0
        position = 0
        while position + segment_samples <= len(audio):
            segment = audio[position : position + segment_samples]
            segment_file = out_dir / f"{prefix}_{segments_created:04d}.{format}"
            save_audio(str(segment_file), segment, sr)
            segments_created += 1
            position += step_samples
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Created {segments_created} segments in {output_dir}")


@audio.command("trim")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--start", "-s", default=0.0, type=float, help="Start time in seconds")
@click.option("--end", "-e", default=None, type=float, help="End time in seconds")
@click.option("--duration", "-d", default=None, type=float, help="Duration in seconds")
def audio_trim(
    input_path: str, output_path: str, start: float, end: float, duration: float
) -> None:
    """Trim audio file to specified time range."""
    from bioamla.audio import load_audio, save_audio, trim_audio

    if end is not None and duration is not None:
        raise click.ClickException("Cannot specify both --end and --duration")

    # Calculate end time from duration if specified
    if duration is not None:
        end = start + duration

    try:
        audio, sr = load_audio(input_path)
        trimmed = trim_audio(
            audio, sr, start_time=start if start != 0.0 else None, end_time=end
        )
        save_audio(output_path, trimmed, sr)
    except (BioamlaError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Trimmed audio saved to: {output_path}")


@audio.command("normalize")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--target-db", "-t", default=-3.0, type=float, help="Target peak level in dB")
@click.option("--method", "-m", type=click.Choice(["peak", "rms"]), default="peak", help="Method")
def audio_normalize(
    input_path: str, output_path: str, target_db: float, method: str
) -> None:
    """Normalize audio amplitude."""
    from bioamla.audio import (
        load_audio,
        normalize_loudness,
        peak_normalize,
        save_audio,
    )

    try:
        audio, sr = load_audio(input_path)

        if method == "peak":
            target_linear = 10 ** (target_db / 20)
            normalized = peak_normalize(audio, target_peak=min(target_linear, 0.99))
        else:
            normalized = normalize_loudness(audio, sr, target_db=target_db)

        save_audio(output_path, normalized, sr)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Normalized audio saved to: {output_path}")


@audio.command("resample")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--sample-rate", "-r", required=True, type=int, help="Target sample rate in Hz")
def audio_resample(input_path: str, output_path: str, sample_rate: int) -> None:
    """Resample audio to a different sample rate."""
    from bioamla.audio import load_audio, resample_audio, save_audio

    try:
        audio, sr = load_audio(input_path)
        resampled = resample_audio(audio, sr, sample_rate)
        save_audio(output_path, resampled, sample_rate)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Resampled audio saved to: {output_path}")


@audio.command("filter")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--lowpass", default=None, type=float, help="Lowpass cutoff frequency in Hz")
@click.option("--highpass", default=None, type=float, help="Highpass cutoff frequency in Hz")
@click.option("--bandpass-low", default=None, type=float, help="Bandpass low frequency in Hz")
@click.option("--bandpass-high", default=None, type=float, help="Bandpass high frequency in Hz")
@click.option("--order", default=5, type=int, help="Filter order (default: 5)")
def audio_filter(
    input_path: str,
    output_path: str,
    lowpass: float,
    highpass: float,
    bandpass_low: float,
    bandpass_high: float,
    order: int,
) -> None:
    """Apply frequency filter to audio file."""
    from bioamla.audio import (
        bandpass_filter,
        highpass_filter,
        load_audio,
        lowpass_filter,
        save_audio,
    )

    # Validate filter options
    if not any([lowpass, highpass, bandpass_low]):
        raise click.ClickException(
            "Must specify --lowpass, --highpass, or --bandpass-low/--bandpass-high"
        )

    if (bandpass_low is not None) != (bandpass_high is not None):
        raise click.ClickException(
            "Both --bandpass-low and --bandpass-high must be specified together"
        )

    try:
        audio, sr = load_audio(input_path)

        if bandpass_low is not None:
            filtered = bandpass_filter(audio, sr, bandpass_low, bandpass_high, order)
            desc = f"bandpass filter ({bandpass_low}-{bandpass_high} Hz)"
        elif lowpass:
            filtered = lowpass_filter(audio, sr, lowpass, order)
            desc = f"lowpass filter ({lowpass} Hz)"
        else:
            filtered = highpass_filter(audio, sr, highpass, order)
            desc = f"highpass filter ({highpass} Hz)"

        save_audio(output_path, filtered, sr)
    except (BioamlaError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Applied {desc} to: {output_path}")


@audio.command("denoise")
@click.argument("input_path")
@click.argument("output_path")
@click.option(
    "--strength", default=1.0, type=float, help="Noise reduction strength (0-2, default: 1.0)"
)
def audio_denoise(input_path: str, output_path: str, strength: float) -> None:
    """Apply spectral noise reduction to audio file."""
    from bioamla.audio import load_audio, save_audio, spectral_denoise

    try:
        audio, sr = load_audio(input_path)
        denoised = spectral_denoise(audio, sr, noise_reduce_factor=strength)
        save_audio(output_path, denoised, sr)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Denoised audio saved to: {output_path}")


@audio.command("visualize")
@click.argument("path")
@click.option("--output", "-o", default=None, help="Output image file path")
@click.option(
    "--type",
    "-t",
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
@click.option(
    "--legend/--no-legend",
    default=True,
    help="Show axes, title, and colorbar (default: True)",
)
def audio_visualize(
    path: str,
    output: str,
    viz_type: str,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    n_mfcc: int,
    cmap: str,
    dpi: int,
    legend: bool,
) -> None:
    """Generate audio visualization (spectrogram, waveform, MFCC) for a single file."""
    from bioamla.viz import generate_spectrogram

    output_path = output or f"{Path(path).stem}_{viz_type}.png"

    try:
        generate_spectrogram(
            audio_path=path,
            output_path=output_path,
            viz_type=viz_type,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            cmap=cmap,
            dpi=dpi,
            show_colorbar=legend,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Visualization saved to: {output_path}")

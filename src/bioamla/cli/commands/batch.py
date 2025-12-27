"""Batch processing operations."""

from typing import Optional

import click

from bioamla.cli.batch_options import batch_input_options, batch_output_options


@click.group()
def batch() -> None:
    """Batch processing operations for multiple files."""
    pass


@batch.group()
def audio() -> None:
    """Batch audio processing operations."""
    pass


@audio.command("info")
@batch_input_options
@batch_output_options
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_info(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], max_workers: int, recursive: bool, quiet: bool) -> None:
    """Extract audio file metadata (duration, sample rate, channels, format, etc.)."""
    import subprocess
    import sys

    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo("Extracting audio metadata...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    try:
        batch_result = services.batch_audio_info.info_batch(config)

        if not quiet:
            click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
            if output_dir:
                click.echo(f"Results saved to {output_dir}/audio_info.csv")
            elif input_file:
                # CSV in-place mode
                click.echo(f"Results merged into {input_file}")
            if batch_result.errors:
                for error in batch_result.errors:
                    click.echo(f"  Error: {error}")
    finally:
        # Force flush all output and reset terminal state
        sys.stdout.flush()
        sys.stderr.flush()
        # Reset terminal to sane state (fixes terminal freeze from parallel processing)
        if sys.stdout.isatty():
            try:
                subprocess.run(["stty", "sane"], check=False, stderr=subprocess.DEVNULL, timeout=1)
            except Exception:
                pass  # Silently fail if stty not available


@audio.command("convert")
@batch_input_options
@batch_output_options
@click.option("--sample-rate", "-r", default=None, type=int, help="Target sample rate")
@click.option("--channels", "-c", default=None, type=int, help="Target number of channels")
@click.option(
    "--format",
    "-f",
    default="wav",
    type=click.Choice(["wav", "mp3", "flac", "ogg"]),
    help="Output format",
)
@click.option("--delete-original", is_flag=True, default=False, help="Delete original files after successful conversion")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_convert(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], sample_rate: int, channels: int, format: str, delete_original: bool, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch convert audio files."""
    import sys

    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo("Converting audio files...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_audio_transform.convert_batch(
        config,
        target_format=format,
        target_sr=sample_rate,
        target_channels=channels,
        delete_original=delete_original,
    )

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")

    # Exit with non-zero status if any files failed
    if batch_result.failed > 0 or batch_result.errors:
        sys.exit(1)


# ==============================================================================
# Audio Transform Batch Commands
# ==============================================================================


@audio.command("resample")
@batch_input_options
@batch_output_options
@click.option("--sample-rate", "-r", required=True, type=int, help="Target sample rate in Hz")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_resample(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], sample_rate: int, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch resample audio files to a target sample rate."""
    import sys

    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo("Resampling audio files...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_audio_transform.resample_batch(config, target_sr=sample_rate)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")

    # Exit with non-zero status if any files failed
    if batch_result.failed > 0 or batch_result.errors:
        sys.exit(1)


@audio.command("normalize")
@batch_input_options
@batch_output_options
@click.option("--target-db", "-d", default=-20.0, type=float, help="Target loudness in dB")
@click.option("--peak", is_flag=True, help="Use peak normalization instead of RMS")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_normalize(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], target_db: float, peak: bool, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch normalize audio levels."""
    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo("Normalizing audio files...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_audio_transform.normalize_batch(config, target_db=target_db, peak=peak)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")


@audio.command("trim")
@batch_input_options
@batch_output_options
@click.option("--start", "-s", default=None, type=float, help="Start time in seconds")
@click.option("--end", "-e", default=None, type=float, help="End time in seconds")
@click.option("--trim-silence", is_flag=True, help="Trim silence from start/end instead of using time range")
@click.option("--silence-threshold-db", default=-40.0, type=float, help="Silence threshold in dB (when using --trim-silence)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_trim(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], start: Optional[float], end: Optional[float], trim_silence: bool, silence_threshold_db: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch trim audio files by time range or remove silence."""
    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo("Trimming audio files...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_audio_transform.trim_batch(
        config,
        start=start,
        end=end,
        trim_silence=trim_silence,
        silence_threshold_db=silence_threshold_db,
    )

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")


@audio.command("filter")
@batch_input_options
@batch_output_options
@click.option("--lowpass", default=None, type=float, help="Lowpass cutoff frequency in Hz")
@click.option("--highpass", default=None, type=float, help="Highpass cutoff frequency in Hz")
@click.option("--bandpass-low", default=None, type=float, help="Bandpass low frequency in Hz (requires --bandpass-high)")
@click.option("--bandpass-high", default=None, type=float, help="Bandpass high frequency in Hz (requires --bandpass-low)")
@click.option("--order", default=5, type=int, help="Filter order (default: 5)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_filter(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], lowpass: Optional[float], highpass: Optional[float], bandpass_low: Optional[float], bandpass_high: Optional[float], order: int, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch apply frequency filters to audio files."""
    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    # Validate filter options
    if not any([lowpass, highpass, bandpass_low]):
        click.echo("Error: Must specify --lowpass, --highpass, or --bandpass-low/--bandpass-high")
        return

    if (bandpass_low is not None) != (bandpass_high is not None):
        click.echo("Error: Both --bandpass-low and --bandpass-high must be specified together")
        return

    bandpass = (bandpass_low, bandpass_high) if bandpass_low is not None else None

    if not quiet:
        if bandpass:
            click.echo(f"Applying bandpass filter ({bandpass[0]}-{bandpass[1]} Hz)...")
        elif lowpass:
            click.echo(f"Applying lowpass filter ({lowpass} Hz)...")
        else:
            click.echo(f"Applying highpass filter ({highpass} Hz)...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_audio_transform.filter_batch(
        config,
        lowpass=lowpass,
        highpass=highpass,
        bandpass=bandpass,
        order=order,
    )

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")


@audio.command("denoise")
@batch_input_options
@batch_output_options
@click.option("--strength", default=1.0, type=float, help="Noise reduction strength (0-2, default: 1.0)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_denoise(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], strength: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch apply spectral noise reduction to audio files."""
    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Applying spectral noise reduction (strength={strength})...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_audio_transform.denoise_batch(
        config,
        strength=strength,
    )

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")


@audio.command("segment")
@batch_input_options
@batch_output_options
@click.option("--duration", "-d", required=True, type=float, help="Segment duration in seconds")
@click.option("--overlap", "-o", default=0.0, type=float, help="Overlap between segments in seconds")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_segment(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], duration: float, overlap: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch segment audio files into fixed-duration chunks."""
    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo("Segmenting audio files...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_audio_transform.segment_batch(config, segment_duration=duration, overlap=overlap)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")


@audio.command("visualize")
@batch_input_options
@batch_output_options
@click.option("--plot-type", "-t", default="mel", type=click.Choice(["mel", "stft", "mfcc", "waveform"]), help="Visualization type")
@click.option("--legend/--no-legend", default=True, help="Show axes, title, and colorbar (default: True)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_visualize(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], plot_type: str, legend: bool, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch generate audio visualizations."""
    import subprocess
    import sys

    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Generating {plot_type} visualizations...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    try:
        batch_result = services.batch_audio_transform.visualize_batch(config, plot_type=plot_type, show_legend=legend)

        if not quiet:
            click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
            if batch_result.errors:
                for error in batch_result.errors:
                    click.echo(f"  Error: {error}")
    finally:
        # Force flush all output and reset terminal state
        sys.stdout.flush()
        sys.stderr.flush()
        # Reset terminal to sane state (fixes terminal corruption from parallel processing)
        if sys.stdout.isatty():
            try:
                subprocess.run(["stty", "sane"], check=False, stderr=subprocess.DEVNULL, timeout=1)
            except Exception:
                pass  # Silently fail if stty not available


# ==============================================================================
# Detection Batch Commands
# ==============================================================================


@batch.group()
def detect() -> None:
    """Batch detection operations."""
    pass


@detect.command("energy")
@batch_input_options
@batch_output_options
@click.option("--low-freq", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option("--threshold-db", default=-20.0, type=float, help="Detection threshold (dB)")
@click.option("--min-duration", default=0.05, type=float, help="Minimum detection duration (s)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def detect_energy(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], low_freq: float, high_freq: float, threshold_db: float, min_duration: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch detect sounds using band-limited energy detection."""
    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo("Detecting energy...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_detection.detect_energy_batch(
        config,
        low_freq=low_freq,
        high_freq=high_freq,
        threshold_db=threshold_db,
        min_duration=min_duration,
    )

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if output_dir:
            click.echo(f"Results saved to {output_dir}/detections.json")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")


@detect.command("ribbit")
@batch_input_options
@batch_output_options
@click.option("--pulse-rate", default=10.0, type=float, help="Expected pulse rate (Hz)")
@click.option("--pulse-tolerance", default=0.2, type=float, help="Pulse rate tolerance")
@click.option("--low-freq", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option("--window-duration", default=2.0, type=float, help="Analysis window duration (s)")
@click.option("--min-score", default=0.3, type=float, help="Minimum detection score")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def detect_ribbit(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], pulse_rate: float, pulse_tolerance: float, low_freq: float, high_freq: float, window_duration: float, min_score: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch detect periodic calls using RIBBIT algorithm."""
    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo("Detecting RIBBIT calls...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_detection.detect_ribbit_batch(
        config,
        pulse_rate_hz=pulse_rate,
        pulse_rate_tolerance=pulse_tolerance,
        low_freq=low_freq,
        high_freq=high_freq,
        window_duration=window_duration,
        min_score=min_score,
    )

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if output_dir:
            click.echo(f"Results saved to {output_dir}/detections.json")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")


@detect.command("peaks")
@batch_input_options
@batch_output_options
@click.option("--snr-threshold", default=2.0, type=float, help="Signal-to-noise ratio threshold")
@click.option("--min-peak-distance", default=0.01, type=float, help="Minimum peak distance (s)")
@click.option("--low-freq", default=None, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=None, type=float, help="High frequency bound (Hz)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def detect_peaks(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], snr_threshold: float, min_peak_distance: float, low_freq: float, high_freq: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch detect peaks using Continuous Wavelet Transform."""
    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo("Detecting peaks...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_detection.detect_peaks_batch(
        config,
        snr_threshold=snr_threshold,
        min_peak_distance=min_peak_distance,
        low_freq=low_freq,
        high_freq=high_freq,
    )

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if output_dir:
            click.echo(f"Results saved to {output_dir}/detections.json")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")


@detect.command("accelerating")
@batch_input_options
@batch_output_options
@click.option("--min-pulses", default=5, type=int, help="Minimum pulses to detect pattern")
@click.option("--accel-threshold", default=1.5, type=float, help="Acceleration threshold")
@click.option("--decel-threshold", default=None, type=float, help="Deceleration threshold")
@click.option("--low-freq", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option("--window-duration", default=3.0, type=float, help="Analysis window duration (s)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def detect_accelerating(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], min_pulses: int, accel_threshold: float, decel_threshold: float, low_freq: float, high_freq: float, window_duration: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch detect accelerating or decelerating call patterns."""
    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo("Detecting accelerating patterns...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_detection.detect_accelerating_batch(
        config,
        min_pulses=min_pulses,
        acceleration_threshold=accel_threshold,
        deceleration_threshold=decel_threshold,
        low_freq=low_freq,
        high_freq=high_freq,
        window_duration=window_duration,
    )

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if output_dir:
            click.echo(f"Results saved to {output_dir}/detections.json")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")


# ==============================================================================
# Indices Batch Commands
# ==============================================================================


@batch.group()
def indices() -> None:
    """Batch acoustic indices operations."""
    pass


@indices.command("calculate")
@batch_input_options
@batch_output_options
@click.option("--indices", default="aci,adi,aei,bio,ndsi,h_spectral,h_temporal", help="Comma-separated list of indices to calculate")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def indices_calculate(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], indices: str, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch calculate acoustic indices for audio files."""
    import subprocess
    import sys

    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo("Calculating acoustic indices...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    # Parse indices list
    indices_list = [idx.strip() for idx in indices.split(",")]

    try:
        batch_result = services.batch_indices.calculate_batch(config, indices=indices_list)

        if not quiet:
            click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
            if output_dir:
                click.echo(f"Results saved to {output_dir}/indices.csv")
            if batch_result.errors:
                for error in batch_result.errors:
                    click.echo(f"  Error: {error}")
    finally:
        # Force flush all output and reset terminal state
        sys.stdout.flush()
        sys.stderr.flush()
        # Reset terminal to sane state (fixes terminal freeze from parallel processing)
        if sys.stdout.isatty():
            try:
                subprocess.run(["stty", "sane"], check=False, stderr=subprocess.DEVNULL, timeout=1)
            except Exception:
                pass  # Silently fail if stty not available


# ==============================================================================
# Model Inference Batch Commands
# ==============================================================================


@batch.group()
def models() -> None:
    """Batch model operations (AST-only)."""
    pass


@models.command("predict")
@batch_input_options
@batch_output_options
@click.option("--model", "-m", required=True, help="Model path (HuggingFace ID or local path)")
@click.option("--top-k", default=5, type=int, help="Number of top predictions to return")
@click.option("--min-confidence", default=0.0, type=float, help="Minimum confidence threshold")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def models_predict(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], model: str, top_k: int, min_confidence: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch run model predictions on audio files."""
    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Running AST predictions with model {model}...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_inference.predict_batch(
        config,
        model_path=model,
        top_k=top_k,
        min_confidence=min_confidence,
    )

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if output_dir:
            click.echo(f"Results saved to {output_dir}/predictions.json")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")


@models.command("embed")
@batch_input_options
@batch_output_options
@click.option("--model", "-m", required=True, help="Model path (HuggingFace ID or local path)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def models_embed(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], model: str, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch extract embeddings from audio files."""
    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Extracting AST embeddings with model {model}...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_inference.embed_batch(config, model_path=model)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if output_dir:
            click.echo(f"Embeddings saved to {output_dir}/")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")


# ==============================================================================
# Clustering Batch Commands
# ==============================================================================


@batch.command("cluster")
@batch_input_options
@batch_output_options
@click.option("--method", default="hdbscan", type=click.Choice(["hdbscan", "kmeans", "dbscan", "agglomerative"]), help="Clustering method")
@click.option("--n-clusters", default=None, type=int, help="Number of clusters (for k-means/agglomerative)")
@click.option("--min-cluster-size", default=5, type=int, help="Minimum cluster size (for HDBSCAN)")
@click.option("--min-samples", default=3, type=int, help="Minimum samples per cluster")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def cluster_batch(input_dir: Optional[str], input_file: Optional[str], output_dir: Optional[str], method: str, n_clusters: int, min_cluster_size: int, min_samples: int, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch cluster embeddings from files."""
    from bioamla.cli.service_helpers import services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Clustering embeddings using {method}...")

    config = BatchConfig(
        input_dir=input_dir,
        input_file=input_file,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    batch_result = services.batch_clustering.cluster_batch(
        config,
        method=method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if output_dir:
            click.echo(f"Cluster results saved to {output_dir}/clusters.json")
        if batch_result.errors:
            for error in batch_result.errors:
                click.echo(f"  Error: {error}")

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


# ==============================================================================
# Audio Transform Batch Commands
# ==============================================================================


@audio.command("resample")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--sample-rate", "-r", required=True, type=int, help="Target sample rate in Hz")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_resample(input_dir: str, output_dir: str, sample_rate: int, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch resample audio files to a target sample rate."""
    from bioamla.cli.service_helpers import handle_result, services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Resampling audio files from {input_dir} to {output_dir}...")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    result = services.batch_audio_transform.resample_batch(config, target_sr=sample_rate)
    batch_result = handle_result(result)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if batch_result.errors:
            for error in batch_result.errors[:5]:
                click.echo(f"  Error: {error}")
            if len(batch_result.errors) > 5:
                click.echo(f"  ... and {len(batch_result.errors) - 5} more errors")


@audio.command("normalize")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--target-db", "-d", default=-20.0, type=float, help="Target loudness in dB")
@click.option("--peak", is_flag=True, help="Use peak normalization instead of RMS")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_normalize(input_dir: str, output_dir: str, target_db: float, peak: bool, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch normalize audio levels."""
    from bioamla.cli.service_helpers import handle_result, services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Normalizing audio files from {input_dir} to {output_dir}...")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    result = services.batch_audio_transform.normalize_batch(config, target_db=target_db, peak=peak)
    batch_result = handle_result(result)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if batch_result.errors:
            for error in batch_result.errors[:5]:
                click.echo(f"  Error: {error}")
            if len(batch_result.errors) > 5:
                click.echo(f"  ... and {len(batch_result.errors) - 5} more errors")


@audio.command("segment")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--duration", "-d", required=True, type=float, help="Segment duration in seconds")
@click.option("--overlap", "-o", default=0.0, type=float, help="Overlap between segments in seconds")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_segment(input_dir: str, output_dir: str, duration: float, overlap: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch segment audio files into fixed-duration chunks."""
    from bioamla.cli.service_helpers import handle_result, services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Segmenting audio files from {input_dir} to {output_dir}...")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    result = services.batch_audio_transform.segment_batch(config, segment_duration=duration, overlap=overlap)
    batch_result = handle_result(result)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if batch_result.errors:
            for error in batch_result.errors[:5]:
                click.echo(f"  Error: {error}")
            if len(batch_result.errors) > 5:
                click.echo(f"  ... and {len(batch_result.errors) - 5} more errors")


@audio.command("visualize")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--plot-type", "-t", default="mel", type=click.Choice(["mel", "stft", "mfcc", "waveform"]), help="Visualization type")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def audio_visualize(input_dir: str, output_dir: str, plot_type: str, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch generate audio visualizations."""
    from bioamla.cli.service_helpers import handle_result, services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Generating {plot_type} visualizations from {input_dir} to {output_dir}...")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    result = services.batch_audio_transform.visualize_batch(config, plot_type=plot_type)
    batch_result = handle_result(result)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        if batch_result.errors:
            for error in batch_result.errors[:5]:
                click.echo(f"  Error: {error}")
            if len(batch_result.errors) > 5:
                click.echo(f"  ... and {len(batch_result.errors) - 5} more errors")


# ==============================================================================
# Detection Batch Commands
# ==============================================================================


@batch.group()
def detect() -> None:
    """Batch detection operations."""
    pass


@detect.command("energy")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--low-freq", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option("--threshold-db", default=-20.0, type=float, help="Detection threshold (dB)")
@click.option("--min-duration", default=0.05, type=float, help="Minimum detection duration (s)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def detect_energy(input_dir: str, output_dir: str, low_freq: float, high_freq: float, threshold_db: float, min_duration: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch detect sounds using band-limited energy detection."""
    from bioamla.cli.service_helpers import handle_result, services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Detecting energy from {input_dir}...")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    result = services.batch_detection.detect_energy_batch(
        config,
        low_freq=low_freq,
        high_freq=high_freq,
        threshold_db=threshold_db,
        min_duration=min_duration,
    )
    batch_result = handle_result(result)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        click.echo(f"Results saved to {output_dir}/detections.json")
        if batch_result.errors:
            for error in batch_result.errors[:5]:
                click.echo(f"  Error: {error}")
            if len(batch_result.errors) > 5:
                click.echo(f"  ... and {len(batch_result.errors) - 5} more errors")


@detect.command("ribbit")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--pulse-rate", default=10.0, type=float, help="Expected pulse rate (Hz)")
@click.option("--pulse-tolerance", default=0.2, type=float, help="Pulse rate tolerance")
@click.option("--low-freq", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option("--window-duration", default=2.0, type=float, help="Analysis window duration (s)")
@click.option("--min-score", default=0.3, type=float, help="Minimum detection score")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def detect_ribbit(input_dir: str, output_dir: str, pulse_rate: float, pulse_tolerance: float, low_freq: float, high_freq: float, window_duration: float, min_score: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch detect periodic calls using RIBBIT algorithm."""
    from bioamla.cli.service_helpers import handle_result, services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Detecting RIBBIT calls from {input_dir}...")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    result = services.batch_detection.detect_ribbit_batch(
        config,
        pulse_rate_hz=pulse_rate,
        pulse_rate_tolerance=pulse_tolerance,
        low_freq=low_freq,
        high_freq=high_freq,
        window_duration=window_duration,
        min_score=min_score,
    )
    batch_result = handle_result(result)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        click.echo(f"Results saved to {output_dir}/detections.json")
        if batch_result.errors:
            for error in batch_result.errors[:5]:
                click.echo(f"  Error: {error}")
            if len(batch_result.errors) > 5:
                click.echo(f"  ... and {len(batch_result.errors) - 5} more errors")


@detect.command("peaks")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--snr-threshold", default=2.0, type=float, help="Signal-to-noise ratio threshold")
@click.option("--min-peak-distance", default=0.01, type=float, help="Minimum peak distance (s)")
@click.option("--low-freq", default=None, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=None, type=float, help="High frequency bound (Hz)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def detect_peaks(input_dir: str, output_dir: str, snr_threshold: float, min_peak_distance: float, low_freq: float, high_freq: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch detect peaks using Continuous Wavelet Transform."""
    from bioamla.cli.service_helpers import handle_result, services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Detecting peaks from {input_dir}...")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    result = services.batch_detection.detect_peaks_batch(
        config,
        snr_threshold=snr_threshold,
        min_peak_distance=min_peak_distance,
        low_freq=low_freq,
        high_freq=high_freq,
    )
    batch_result = handle_result(result)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        click.echo(f"Results saved to {output_dir}/detections.json")
        if batch_result.errors:
            for error in batch_result.errors[:5]:
                click.echo(f"  Error: {error}")
            if len(batch_result.errors) > 5:
                click.echo(f"  ... and {len(batch_result.errors) - 5} more errors")


@detect.command("accelerating")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--min-pulses", default=5, type=int, help="Minimum pulses to detect pattern")
@click.option("--accel-threshold", default=1.5, type=float, help="Acceleration threshold")
@click.option("--decel-threshold", default=None, type=float, help="Deceleration threshold")
@click.option("--low-freq", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option("--window-duration", default=3.0, type=float, help="Analysis window duration (s)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def detect_accelerating(input_dir: str, output_dir: str, min_pulses: int, accel_threshold: float, decel_threshold: float, low_freq: float, high_freq: float, window_duration: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch detect accelerating or decelerating call patterns."""
    from bioamla.cli.service_helpers import handle_result, services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Detecting accelerating patterns from {input_dir}...")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    result = services.batch_detection.detect_accelerating_batch(
        config,
        min_pulses=min_pulses,
        acceleration_threshold=accel_threshold,
        deceleration_threshold=decel_threshold,
        low_freq=low_freq,
        high_freq=high_freq,
        window_duration=window_duration,
    )
    batch_result = handle_result(result)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        click.echo(f"Results saved to {output_dir}/detections.json")
        if batch_result.errors:
            for error in batch_result.errors[:5]:
                click.echo(f"  Error: {error}")
            if len(batch_result.errors) > 5:
                click.echo(f"  ... and {len(batch_result.errors) - 5} more errors")


# ==============================================================================
# Indices Batch Commands
# ==============================================================================


@batch.group()
def indices() -> None:
    """Batch acoustic indices operations."""
    pass


@indices.command("calculate")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--indices", default="aci,adi,aei,bio,ndsi,h_spectral,h_temporal", help="Comma-separated list of indices to calculate")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def indices_calculate(input_dir: str, output_dir: str, indices: str, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch calculate acoustic indices for audio files."""
    from bioamla.cli.service_helpers import handle_result, services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Calculating acoustic indices from {input_dir}...")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    # Parse indices list
    indices_list = [idx.strip() for idx in indices.split(",")]

    result = services.batch_indices.calculate_batch(config, indices=indices_list)
    batch_result = handle_result(result)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        click.echo(f"Results saved to {output_dir}/indices.csv")
        if batch_result.errors:
            for error in batch_result.errors[:5]:
                click.echo(f"  Error: {error}")
            if len(batch_result.errors) > 5:
                click.echo(f"  ... and {len(batch_result.errors) - 5} more errors")


# ==============================================================================
# Model Inference Batch Commands
# ==============================================================================


@batch.group()
def models() -> None:
    """Batch model operations (AST-only)."""
    pass


@models.command("predict")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--model", "-m", required=True, help="Model path (HuggingFace ID or local path)")
@click.option("--top-k", default=5, type=int, help="Number of top predictions to return")
@click.option("--min-confidence", default=0.0, type=float, help="Minimum confidence threshold")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def models_predict(input_dir: str, output_dir: str, model: str, top_k: int, min_confidence: float, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch run model predictions on audio files."""
    from bioamla.cli.service_helpers import handle_result, services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Running AST predictions on {input_dir} with model {model}...")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    result = services.batch_inference.predict_batch(
        config,
        model_path=model,
        top_k=top_k,
        min_confidence=min_confidence,
    )
    batch_result = handle_result(result)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        click.echo(f"Results saved to {output_dir}/predictions.json")
        if batch_result.errors:
            for error in batch_result.errors[:5]:
                click.echo(f"  Error: {error}")
            if len(batch_result.errors) > 5:
                click.echo(f"  ... and {len(batch_result.errors) - 5} more errors")


@models.command("embed")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--model", "-m", required=True, help="Model path (HuggingFace ID or local path)")
@click.option("--max-workers", "-w", default=1, type=int, help="Number of parallel workers")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def models_embed(input_dir: str, output_dir: str, model: str, max_workers: int, recursive: bool, quiet: bool) -> None:
    """Batch extract embeddings from audio files."""
    from bioamla.cli.service_helpers import handle_result, services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Extracting AST embeddings from {input_dir} with model {model}...")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=max_workers,
        quiet=quiet,
    )

    result = services.batch_inference.embed_batch(config, model_path=model)
    batch_result = handle_result(result)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        click.echo(f"Embeddings saved to {output_dir}/")
        if batch_result.errors:
            for error in batch_result.errors[:5]:
                click.echo(f"  Error: {error}")
            if len(batch_result.errors) > 5:
                click.echo(f"  ... and {len(batch_result.errors) - 5} more errors")


# ==============================================================================
# Clustering Batch Commands
# ==============================================================================


@batch.command("cluster")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--method", default="hdbscan", type=click.Choice(["hdbscan", "kmeans", "dbscan", "agglomerative"]), help="Clustering method")
@click.option("--n-clusters", default=None, type=int, help="Number of clusters (for k-means/agglomerative)")
@click.option("--min-cluster-size", default=5, type=int, help="Minimum cluster size (for HDBSCAN)")
@click.option("--min-samples", default=3, type=int, help="Minimum samples per cluster")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def cluster_batch(input_dir: str, output_dir: str, method: str, n_clusters: int, min_cluster_size: int, min_samples: int, recursive: bool, quiet: bool) -> None:
    """Batch cluster embeddings from files."""
    from bioamla.cli.service_helpers import handle_result, services
    from bioamla.models.batch import BatchConfig

    if not quiet:
        click.echo(f"Clustering embeddings from {input_dir} using {method}...")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        max_workers=1,  # Clustering is not parallelized
        quiet=quiet,
    )

    result = services.batch_clustering.cluster_batch(
        config,
        method=method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    batch_result = handle_result(result)

    if not quiet:
        click.echo(f"Processed {batch_result.total_files} files: {batch_result.successful} successful, {batch_result.failed} failed")
        click.echo(f"Cluster results saved to {output_dir}/clusters.json")
        if batch_result.errors:
            for error in batch_result.errors[:5]:
                click.echo(f"  Error: {error}")
            if len(batch_result.errors) > 5:
                click.echo(f"  ... and {len(batch_result.errors) - 5} more errors")

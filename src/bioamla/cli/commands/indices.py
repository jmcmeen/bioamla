"""Acoustic indices for soundscape ecology analysis."""

import click


@click.group()
def indices() -> None:
    """Acoustic indices for soundscape ecology analysis."""
    pass


@indices.command("compute")
@click.argument("file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output CSV file for results")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
@click.option("--n-fft", default=512, type=int, help="FFT window size")
@click.option("--aci-min-freq", default=0.0, type=float, help="ACI minimum frequency (Hz)")
@click.option("--aci-max-freq", default=None, type=float, help="ACI maximum frequency (Hz)")
@click.option("--bio-min-freq", default=2000.0, type=float, help="BIO minimum frequency (Hz)")
@click.option("--bio-max-freq", default=8000.0, type=float, help="BIO maximum frequency (Hz)")
@click.option("--db-threshold", default=-50.0, type=float, help="dB threshold for ADI/AEI")
def indices_compute(
    file: str,
    output: str,
    output_format: str,
    n_fft: int,
    aci_min_freq: float,
    aci_max_freq: float,
    bio_min_freq: float,
    bio_max_freq: float,
    db_threshold: float,
) -> None:
    """Compute all acoustic indices for a single audio file."""
    import json as json_lib

    from bioamla.cli.service_helpers import handle_result, services

    # Load audio file
    audio_data = handle_result(services.audio_file.open(file))

    # Calculate indices
    kwargs = {
        "n_fft": n_fft,
        "aci_min_freq": aci_min_freq,
        "bio_min_freq": bio_min_freq,
        "bio_max_freq": bio_max_freq,
        "db_threshold": db_threshold,
    }
    if aci_max_freq:
        kwargs["aci_max_freq"] = aci_max_freq

    result = services.indices.calculate(audio_data, include_entropy=True, **kwargs)
    indices_result = handle_result(result)

    # Format output
    output_dict = {"filepath": file}
    output_dict.update(indices_result.to_dict())

    if output:
        services.file.write_json(output, output_dict)
        click.echo(f"Results saved to {output}")
    elif output_format == "json":
        click.echo(json_lib.dumps(output_dict, indent=2))
    elif output_format == "csv":
        import csv
        import sys

        writer = csv.DictWriter(sys.stdout, fieldnames=list(output_dict.keys()))
        writer.writeheader()
        writer.writerow(output_dict)
    else:
        click.echo(f"\n{file}:")
        click.echo(f"  ACI:  {indices_result.indices.aci:.2f}")
        click.echo(f"  ADI:  {indices_result.indices.adi:.3f}")
        click.echo(f"  AEI:  {indices_result.indices.aei:.3f}")
        click.echo(f"  BIO:  {indices_result.indices.bio:.2f}")
        click.echo(f"  NDSI: {indices_result.indices.ndsi:.3f}")
        if indices_result.h_spectral is not None:
            click.echo(f"  H (spectral): {indices_result.h_spectral:.3f}")
        if indices_result.h_temporal is not None:
            click.echo(f"  H (temporal): {indices_result.h_temporal:.3f}")


@indices.command("temporal")
@click.argument("file", type=click.Path(exists=True))
@click.option("--segment-duration", default=60.0, type=float, help="Segment duration in seconds")
@click.option("--output", "-o", type=click.Path(), help="Output CSV file for results")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_temporal(file: str, segment_duration: float, output: str, n_fft: int) -> None:
    """Compute temporal acoustic indices (indices over time segments)."""
    from bioamla.cli.service_helpers import handle_result, services

    # Load audio file
    audio_data = handle_result(services.audio_file.open(file))

    # Calculate temporal indices
    result = services.indices.calculate_temporal(
        audio_data, segment_duration=segment_duration, n_fft=n_fft
    )
    temporal_result = handle_result(result)

    if output:
        rows = []
        for i, window_indices in enumerate(temporal_result.windows):
            row = {
                "segment": i,
                "start_time": i * segment_duration,
                "aci": window_indices["aci"],
                "adi": window_indices["adi"],
                "aei": window_indices["aei"],
                "bio": window_indices["bio"],
                "ndsi": window_indices["ndsi"],
            }
            rows.append(row)
        services.file.write_json(output, rows)
        click.echo(f"Temporal indices saved to {output}")
    else:
        click.echo(f"\n{file} - Temporal indices ({segment_duration}s segments):")
        for i, window_indices in enumerate(temporal_result.windows[:10]):
            start_time = i * segment_duration
            click.echo(
                f"  Segment {i} ({start_time:.1f}s): "
                f"ACI={window_indices['aci']:.2f}, ADI={window_indices['adi']:.3f}, "
                f"AEI={window_indices['aei']:.3f}, BIO={window_indices['bio']:.2f}, "
                f"NDSI={window_indices['ndsi']:.3f}"
            )
        if len(temporal_result.windows) > 10:
            click.echo(f"  ... and {len(temporal_result.windows) - 10} more segments")
        click.echo(f"\nTotal segments: {len(temporal_result.windows)}")


# Individual index commands for convenience
@indices.command("aci")
@click.argument("file", type=click.Path(exists=True))
@click.option("--min-freq", default=0.0, type=float, help="Minimum frequency (Hz)")
@click.option("--max-freq", default=None, type=float, help="Maximum frequency (Hz)")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_aci(file: str, min_freq: float, max_freq: float, n_fft: int) -> None:
    """Compute Acoustic Complexity Index (ACI) only."""
    from bioamla.cli.service_helpers import handle_result, services

    audio_data = handle_result(services.audio_file.open(file))
    result = services.indices.calculate(
        audio_data, n_fft=n_fft, aci_min_freq=min_freq, aci_max_freq=max_freq
    )
    indices_result = handle_result(result)
    click.echo(f"ACI: {indices_result.indices.aci:.2f}")


@indices.command("adi")
@click.argument("file", type=click.Path(exists=True))
@click.option("--db-threshold", default=-50.0, type=float, help="dB threshold")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_adi(file: str, db_threshold: float, n_fft: int) -> None:
    """Compute Acoustic Diversity Index (ADI) only."""
    from bioamla.cli.service_helpers import handle_result, services

    audio_data = handle_result(services.audio_file.open(file))
    result = services.indices.calculate(audio_data, n_fft=n_fft, db_threshold=db_threshold)
    indices_result = handle_result(result)
    click.echo(f"ADI: {indices_result.indices.adi:.3f}")


@indices.command("aei")
@click.argument("file", type=click.Path(exists=True))
@click.option("--db-threshold", default=-50.0, type=float, help="dB threshold")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_aei(file: str, db_threshold: float, n_fft: int) -> None:
    """Compute Acoustic Evenness Index (AEI) only."""
    from bioamla.cli.service_helpers import handle_result, services

    audio_data = handle_result(services.audio_file.open(file))
    result = services.indices.calculate(audio_data, n_fft=n_fft, db_threshold=db_threshold)
    indices_result = handle_result(result)
    click.echo(f"AEI: {indices_result.indices.aei:.3f}")


@indices.command("bio")
@click.argument("file", type=click.Path(exists=True))
@click.option("--min-freq", default=2000.0, type=float, help="Minimum frequency (Hz)")
@click.option("--max-freq", default=8000.0, type=float, help="Maximum frequency (Hz)")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_bio(file: str, min_freq: float, max_freq: float, n_fft: int) -> None:
    """Compute Bioacoustic Index (BIO) only."""
    from bioamla.cli.service_helpers import handle_result, services

    audio_data = handle_result(services.audio_file.open(file))
    result = services.indices.calculate(
        audio_data, n_fft=n_fft, bio_min_freq=min_freq, bio_max_freq=max_freq
    )
    indices_result = handle_result(result)
    click.echo(f"BIO: {indices_result.indices.bio:.2f}")


@indices.command("ndsi")
@click.argument("file", type=click.Path(exists=True))
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_ndsi(file: str, n_fft: int) -> None:
    """Compute Normalized Difference Soundscape Index (NDSI) only."""
    from bioamla.cli.service_helpers import handle_result, services

    audio_data = handle_result(services.audio_file.open(file))
    result = services.indices.calculate(audio_data, n_fft=n_fft)
    indices_result = handle_result(result)
    click.echo(f"NDSI: {indices_result.indices.ndsi:.3f}")


@indices.command("entropy")
@click.argument("file", type=click.Path(exists=True))
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_entropy(file: str, n_fft: int) -> None:
    """Compute spectral and temporal entropy indices only."""
    from bioamla.cli.service_helpers import handle_result, services

    audio_data = handle_result(services.audio_file.open(file))
    result = services.indices.calculate(audio_data, n_fft=n_fft, include_entropy=True)
    indices_result = handle_result(result)

    if hasattr(indices_result, "h_spectral") and indices_result.h_spectral is not None:
        click.echo(f"H (spectral): {indices_result.h_spectral:.3f}")
    if hasattr(indices_result, "h_temporal") and indices_result.h_temporal is not None:
        click.echo(f"H (temporal): {indices_result.h_temporal:.3f}")

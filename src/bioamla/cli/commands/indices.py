"""Acoustic indices for soundscape ecology analysis."""

import click

from bioamla.exceptions import BioamlaError


@click.group()
def indices() -> None:
    """Acoustic indices for soundscape ecology analysis."""
    pass


@indices.command("compute")
@click.argument("file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON file for results")
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

    from bioamla.audio import load_audio_data
    from bioamla.indices import compute_all_indices

    try:
        audio = load_audio_data(file)
        kwargs = {
            "n_fft": n_fft,
            "aci_min_freq": aci_min_freq,
            "bio_min_freq": bio_min_freq,
            "bio_max_freq": bio_max_freq,
            "db_threshold": db_threshold,
        }
        if aci_max_freq:
            kwargs["aci_max_freq"] = aci_max_freq
        result = compute_all_indices(
            audio.samples, audio.sample_rate, include_entropy=True, **kwargs
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    output_dict = {"filepath": file}
    output_dict.update(result.to_dict())

    if output:
        from pathlib import Path

        try:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json_lib.dumps(output_dict, indent=2))
        except OSError as e:
            raise click.ClickException(f"Failed to write {output}: {e}") from e
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
        click.echo(f"  ACI:  {result.aci:.2f}")
        click.echo(f"  ADI:  {result.adi:.3f}")
        click.echo(f"  AEI:  {result.aei:.3f}")
        click.echo(f"  BIO:  {result.bio:.2f}")
        click.echo(f"  NDSI: {result.ndsi:.3f}")
        if result.h_spectral is not None:
            click.echo(f"  H (spectral): {result.h_spectral:.3f}")
        if result.h_temporal is not None:
            click.echo(f"  H (temporal): {result.h_temporal:.3f}")


@indices.command("temporal")
@click.argument("file", type=click.Path(exists=True))
@click.option("--segment-duration", default=60.0, type=float, help="Segment duration in seconds")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file for results")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_temporal(file: str, segment_duration: float, output: str, n_fft: int) -> None:
    """Compute temporal acoustic indices (indices over time segments)."""
    import json as json_lib

    from bioamla.audio import load_audio_data
    from bioamla.indices import temporal_indices

    try:
        audio = load_audio_data(file)
        windows = temporal_indices(
            audio.samples,
            audio.sample_rate,
            window_duration=segment_duration,
            hop_duration=segment_duration,
            n_fft=n_fft,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if output:
        from pathlib import Path

        rows = [
            {
                "segment": i,
                "start_time": w.get("start_time", i * segment_duration),
                "aci": w["aci"],
                "adi": w["adi"],
                "aei": w["aei"],
                "bio": w["bio"],
                "ndsi": w["ndsi"],
            }
            for i, w in enumerate(windows)
        ]
        try:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json_lib.dumps(rows, indent=2))
        except OSError as e:
            raise click.ClickException(f"Failed to write {output}: {e}") from e
        click.echo(f"Temporal indices saved to {output}")
    else:
        click.echo(f"\n{file} - Temporal indices ({segment_duration}s segments):")
        for i, w in enumerate(windows[:10]):
            start_time = w.get("start_time", i * segment_duration)
            click.echo(
                f"  Segment {i} ({start_time:.1f}s): "
                f"ACI={w['aci']:.2f}, ADI={w['adi']:.3f}, "
                f"AEI={w['aei']:.3f}, BIO={w['bio']:.2f}, NDSI={w['ndsi']:.3f}"
            )
        if len(windows) > 10:
            click.echo(f"  ... and {len(windows) - 10} more segments")
        click.echo(f"\nTotal segments: {len(windows)}")


@indices.command("aci")
@click.argument("file", type=click.Path(exists=True))
@click.option("--min-freq", default=0.0, type=float, help="Minimum frequency (Hz)")
@click.option("--max-freq", default=None, type=float, help="Maximum frequency (Hz)")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_aci(file: str, min_freq: float, max_freq: float, n_fft: int) -> None:
    """Compute Acoustic Complexity Index (ACI) only."""
    from bioamla.audio import load_audio_data
    from bioamla.indices import compute_index

    try:
        audio = load_audio_data(file)
        value = compute_index(
            audio.samples,
            audio.sample_rate,
            "aci",
            n_fft=n_fft,
            min_freq=min_freq,
            max_freq=max_freq,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"ACI: {value:.2f}")


@indices.command("adi")
@click.argument("file", type=click.Path(exists=True))
@click.option("--db-threshold", default=-50.0, type=float, help="dB threshold")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_adi(file: str, db_threshold: float, n_fft: int) -> None:
    """Compute Acoustic Diversity Index (ADI) only."""
    from bioamla.audio import load_audio_data
    from bioamla.indices import compute_index

    try:
        audio = load_audio_data(file)
        value = compute_index(
            audio.samples, audio.sample_rate, "adi", n_fft=n_fft, db_threshold=db_threshold
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"ADI: {value:.3f}")


@indices.command("aei")
@click.argument("file", type=click.Path(exists=True))
@click.option("--db-threshold", default=-50.0, type=float, help="dB threshold")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_aei(file: str, db_threshold: float, n_fft: int) -> None:
    """Compute Acoustic Evenness Index (AEI) only."""
    from bioamla.audio import load_audio_data
    from bioamla.indices import compute_index

    try:
        audio = load_audio_data(file)
        value = compute_index(
            audio.samples, audio.sample_rate, "aei", n_fft=n_fft, db_threshold=db_threshold
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"AEI: {value:.3f}")


@indices.command("bio")
@click.argument("file", type=click.Path(exists=True))
@click.option("--min-freq", default=2000.0, type=float, help="Minimum frequency (Hz)")
@click.option("--max-freq", default=8000.0, type=float, help="Maximum frequency (Hz)")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_bio(file: str, min_freq: float, max_freq: float, n_fft: int) -> None:
    """Compute Bioacoustic Index (BIO) only."""
    from bioamla.audio import load_audio_data
    from bioamla.indices import compute_index

    try:
        audio = load_audio_data(file)
        value = compute_index(
            audio.samples,
            audio.sample_rate,
            "bio",
            n_fft=n_fft,
            min_freq=min_freq,
            max_freq=max_freq,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"BIO: {value:.2f}")


@indices.command("ndsi")
@click.argument("file", type=click.Path(exists=True))
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_ndsi(file: str, n_fft: int) -> None:
    """Compute Normalized Difference Soundscape Index (NDSI) only."""
    from bioamla.audio import load_audio_data
    from bioamla.indices import compute_index

    try:
        audio = load_audio_data(file)
        value = compute_index(audio.samples, audio.sample_rate, "ndsi", n_fft=n_fft)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"NDSI: {value:.3f}")


@indices.command("entropy")
@click.argument("file", type=click.Path(exists=True))
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_entropy(file: str, n_fft: int) -> None:
    """Compute spectral and temporal entropy indices only."""
    from bioamla.audio import load_audio_data
    from bioamla.indices import compute_index

    try:
        audio = load_audio_data(file)
        h_spectral = compute_index(audio.samples, audio.sample_rate, "h_spectral", n_fft=n_fft)
        h_temporal = compute_index(audio.samples, audio.sample_rate, "h_temporal", n_fft=n_fft)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"H (spectral): {h_spectral:.3f}")
    click.echo(f"H (temporal): {h_temporal:.3f}")

"""Acoustic indices for soundscape ecology analysis."""

import click

from bioamla.core.files import TextFile


@click.group()
def indices():
    """Acoustic indices for soundscape ecology analysis."""
    pass


@indices.command("compute")
@click.argument("path", type=click.Path(exists=True))
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
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def indices_compute(
    path,
    output,
    output_format,
    n_fft,
    aci_min_freq,
    aci_max_freq,
    bio_min_freq,
    bio_max_freq,
    db_threshold,
    quiet,
):
    """Compute acoustic indices for audio file(s)."""
    import json as json_lib
    from pathlib import Path as PathLib

    from bioamla.services.audio_file import AudioFileController
    from bioamla.services.indices import IndicesController

    path_obj = PathLib(path)

    kwargs = {
        "n_fft": n_fft,
        "aci_min_freq": aci_min_freq,
        "bio_min_freq": bio_min_freq,
        "bio_max_freq": bio_max_freq,
        "db_threshold": db_threshold,
    }
    if aci_max_freq:
        kwargs["aci_max_freq"] = aci_max_freq

    indices_ctrl = IndicesController()

    if path_obj.is_file():
        file_ctrl = AudioFileController()
        load_result = file_ctrl.open(str(path_obj))

        if not load_result.success:
            click.echo(f"Error loading {path}: {load_result.error}")
            raise SystemExit(1)

        calc_result = indices_ctrl.calculate(load_result.data, include_entropy=True, **kwargs)

        if not calc_result.success:
            click.echo(f"Error computing indices: {calc_result.error}")
            raise SystemExit(1)

        result = {"filepath": str(path_obj)}
        result.update(calc_result.data.to_dict())
        result["success"] = True
        results = [result]
    else:
        batch_result = indices_ctrl.calculate_batch(
            str(path_obj),
            output_path=output if output else None,
            recursive=True,
            include_entropy=True,
            **kwargs,
        )

        if not batch_result.success:
            click.echo(f"Error: {batch_result.error}")
            raise SystemExit(1)

        results = batch_result.data.results

        if output and batch_result.data.output_path:
            click.echo(f"Results saved to {batch_result.data.output_path}")
            return

    successful = [r for r in results if r.get("success", False)]
    failed = len(results) - len(successful)

    if output_format == "json":
        click.echo(json_lib.dumps(results, indent=2))
    elif output_format == "csv" or output:
        import csv
        import sys

        if successful:
            fieldnames = list(successful[0].keys())
            if output:
                with TextFile(output, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(successful)
                click.echo(f"Results saved to {output}")
            else:
                writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(successful)
    else:
        for r in results:
            if r.get("success"):
                click.echo(f"\n{r.get('filepath', 'Unknown')}:")
                click.echo(f"  ACI:  {r['aci']:.2f}")
                click.echo(f"  ADI:  {r['adi']:.3f}")
                click.echo(f"  AEI:  {r['aei']:.3f}")
                click.echo(f"  BIO:  {r['bio']:.2f}")
                click.echo(f"  NDSI: {r['ndsi']:.3f}")
                if r.get("h_spectral"):
                    click.echo(f"  H (spectral): {r['h_spectral']:.3f}")
                if r.get("h_temporal"):
                    click.echo(f"  H (temporal): {r['h_temporal']:.3f}")
            else:
                click.echo(
                    f"\n{r.get('filepath', 'Unknown')}: Error - {r.get('error', 'Unknown error')}"
                )

    if not quiet:
        click.echo(
            f"\nProcessed {len(results)} file(s): {len(successful)} successful, {failed} failed"
        )


@indices.command("temporal")
@click.argument("path", type=click.Path(exists=True))
@click.option("--window", "-w", default=60.0, type=float, help="Window duration in seconds")
@click.option("--hop", default=None, type=float, help="Hop duration in seconds (default: window)")
@click.option("--output", "-o", type=click.Path(), help="Output CSV file")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def indices_temporal(path, window, hop, output, output_format, quiet):
    """Compute acoustic indices over time windows."""
    import json as json_lib

    from bioamla.services.audio_file import AudioFileController
    from bioamla.services.indices import IndicesController

    file_ctrl = AudioFileController()
    indices_ctrl = IndicesController()

    load_result = file_ctrl.open(path)
    if not load_result.success:
        click.echo(f"Error loading audio: {load_result.error}")
        raise SystemExit(1)

    audio_data = load_result.data

    if not quiet:
        click.echo(f"Processing {path}")
        click.echo(f"Duration: {audio_data.duration:.1f}s, Sample rate: {audio_data.sample_rate} Hz")
        click.echo(f"Window: {window}s, Hop: {hop or window}s")

    temporal_result = indices_ctrl.calculate_temporal(
        audio_data,
        window_duration=window,
        hop_duration=hop,
    )

    if not temporal_result.success:
        click.echo(f"Error: {temporal_result.error}")
        raise SystemExit(1)

    results = temporal_result.data.windows

    if not results:
        click.echo("No complete windows in recording (audio shorter than window duration)")
        raise SystemExit(1)

    if output_format == "json":
        click.echo(json_lib.dumps(results, indent=2))
    elif output_format == "csv" or output:
        import csv
        import sys

        fieldnames = list(results[0].keys())
        if output:
            with TextFile(output, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            click.echo(f"Results saved to {output}")
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    else:
        click.echo(f"\nTemporal analysis ({len(results)} windows):")
        click.echo("-" * 70)
        click.echo(f"{'Time':>12}  {'ACI':>8}  {'ADI':>6}  {'AEI':>6}  {'BIO':>8}  {'NDSI':>6}")
        click.echo("-" * 70)
        for r in results:
            time_str = f"{r['start_time']:.0f}-{r['end_time']:.0f}s"
            click.echo(
                f"{time_str:>12}  {r['aci']:>8.2f}  {r['adi']:>6.3f}  "
                f"{r['aei']:>6.3f}  {r['bio']:>8.2f}  {r['ndsi']:>6.3f}"
            )


@indices.command("aci")
@click.argument("path", type=click.Path(exists=True))
@click.option("--min-freq", default=0.0, type=float, help="Minimum frequency (Hz)")
@click.option("--max-freq", default=None, type=float, help="Maximum frequency (Hz)")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_aci(path, min_freq, max_freq, n_fft):
    """Compute Acoustic Complexity Index (ACI) for an audio file."""
    import librosa

    from bioamla.core.analysis.indices import compute_aci

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1) from e

    kwargs = {"n_fft": n_fft, "min_freq": min_freq}
    if max_freq:
        kwargs["max_freq"] = max_freq

    aci = compute_aci(audio, sample_rate, **kwargs)
    click.echo(f"ACI: {aci:.2f}")


@indices.command("adi")
@click.argument("path", type=click.Path(exists=True))
@click.option("--max-freq", default=10000.0, type=float, help="Maximum frequency (Hz)")
@click.option("--freq-step", default=1000.0, type=float, help="Frequency band width (Hz)")
@click.option("--db-threshold", default=-50.0, type=float, help="dB threshold")
def indices_adi(path, max_freq, freq_step, db_threshold):
    """Compute Acoustic Diversity Index (ADI) for an audio file."""
    import librosa

    from bioamla.core.analysis.indices import compute_adi

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1) from e

    adi = compute_adi(
        audio, sample_rate, max_freq=max_freq, freq_step=freq_step, db_threshold=db_threshold
    )
    click.echo(f"ADI: {adi:.3f}")


@indices.command("aei")
@click.argument("path", type=click.Path(exists=True))
@click.option("--max-freq", default=10000.0, type=float, help="Maximum frequency (Hz)")
@click.option("--freq-step", default=1000.0, type=float, help="Frequency band width (Hz)")
@click.option("--db-threshold", default=-50.0, type=float, help="dB threshold")
def indices_aei(path, max_freq, freq_step, db_threshold):
    """Compute Acoustic Evenness Index (AEI) for an audio file."""
    import librosa

    from bioamla.core.analysis.indices import compute_aei

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1) from e

    aei = compute_aei(
        audio, sample_rate, max_freq=max_freq, freq_step=freq_step, db_threshold=db_threshold
    )
    click.echo(f"AEI: {aei:.3f}")


@indices.command("bio")
@click.argument("path", type=click.Path(exists=True))
@click.option("--min-freq", default=2000.0, type=float, help="Minimum frequency (Hz)")
@click.option("--max-freq", default=8000.0, type=float, help="Maximum frequency (Hz)")
def indices_bio(path, min_freq, max_freq):
    """Compute Bioacoustic Index (BIO) for an audio file."""
    import librosa

    from bioamla.core.analysis.indices import compute_bio

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1) from e

    bio = compute_bio(audio, sample_rate, min_freq=min_freq, max_freq=max_freq)
    click.echo(f"BIO: {bio:.2f}")


@indices.command("ndsi")
@click.argument("path", type=click.Path(exists=True))
@click.option("--anthro-min", default=1000.0, type=float, help="Anthrophony min frequency (Hz)")
@click.option("--anthro-max", default=2000.0, type=float, help="Anthrophony max frequency (Hz)")
@click.option("--bio-min", default=2000.0, type=float, help="Biophony min frequency (Hz)")
@click.option("--bio-max", default=8000.0, type=float, help="Biophony max frequency (Hz)")
def indices_ndsi(path, anthro_min, anthro_max, bio_min, bio_max):
    """Compute Normalized Difference Soundscape Index (NDSI) for an audio file."""
    import librosa

    from bioamla.core.analysis.indices import compute_ndsi

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1) from e

    ndsi, anthro, bio = compute_ndsi(
        audio,
        sample_rate,
        anthro_min=anthro_min,
        anthro_max=anthro_max,
        bio_min=bio_min,
        bio_max=bio_max,
    )

    click.echo(f"NDSI: {ndsi:.3f}")
    click.echo(f"  Anthrophony ({anthro_min:.0f}-{anthro_max:.0f} Hz): {anthro:.2f}")
    click.echo(f"  Biophony ({bio_min:.0f}-{bio_max:.0f} Hz): {bio:.2f}")


@indices.command("entropy")
@click.argument("path", type=click.Path(exists=True))
@click.option("--spectral", "-s", is_flag=True, help="Compute spectral entropy")
@click.option("--temporal", "-t", is_flag=True, help="Compute temporal entropy")
def indices_entropy(path, spectral, temporal):
    """Compute entropy-based acoustic indices for an audio file."""
    import librosa

    from bioamla.core.analysis.indices import spectral_entropy, temporal_entropy

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1) from e

    if not spectral and not temporal:
        spectral = temporal = True

    if spectral:
        se = spectral_entropy(audio, sample_rate)
        click.echo(f"Spectral Entropy: {se:.3f}")

    if temporal:
        te = temporal_entropy(audio, sample_rate)
        click.echo(f"Temporal Entropy: {te:.3f}")

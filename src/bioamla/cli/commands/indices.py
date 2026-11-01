"""Acoustic indices for soundscape ecology analysis."""

import click


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
@click.option(
    "--aci-low-freq", "aci_min_freq", default=0.0, type=float, help="ACI low-frequency bound (Hz)"
)
@click.option(
    "--aci-high-freq",
    "aci_max_freq",
    default=None,
    type=float,
    help="ACI high-frequency bound (Hz)",
)
@click.option(
    "--bio-low-freq",
    "bio_min_freq",
    default=2000.0,
    type=float,
    help="BIO low-frequency bound (Hz)",
)
@click.option(
    "--bio-high-freq",
    "bio_max_freq",
    default=8000.0,
    type=float,
    help="BIO high-frequency bound (Hz)",
)
@click.option(
    "--threshold-db", "db_threshold", default=-50.0, type=float, help="dB threshold for ADI/AEI"
)
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

    from bioamla.cli.console import print_header, print_kv, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    found = run_operation(
        "indices.compute",
        file=file,
        n_fft=n_fft,
        aci_min_freq=aci_min_freq,
        aci_max_freq=aci_max_freq,
        bio_min_freq=bio_min_freq,
        bio_max_freq=bio_max_freq,
        db_threshold=db_threshold,
        output=output,
    )
    result = found.indices
    indices_dict = result.to_dict()
    output_dict = {"filepath": file, **indices_dict}
    if emit_file_result(file, indices=indices_dict):
        return
    if output:
        print_success(f"Results saved to {output}")
    elif output_format == "json":
        click.echo(json_lib.dumps(output_dict, indent=2))
    elif output_format == "csv":
        import csv
        import sys

        writer = csv.DictWriter(sys.stdout, fieldnames=list(output_dict.keys()))
        writer.writeheader()
        writer.writerow(output_dict)
    else:
        print_header(f"\n{file}:")
        print_kv("  ACI", f"{result.aci:.2f}")
        print_kv("  ADI", f"{result.adi:.3f}")
        print_kv("  AEI", f"{result.aei:.3f}")
        print_kv("  BIO", f"{result.bio:.2f}")
        print_kv("  NDSI", f"{result.ndsi:.3f}")
        if result.h_spectral is not None:
            print_kv("  H (spectral)", f"{result.h_spectral:.3f}")
        if result.h_temporal is not None:
            print_kv("  H (temporal)", f"{result.h_temporal:.3f}")


@indices.command("temporal")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--window-seconds",
    "segment_duration",
    default=60.0,
    type=float,
    help="Analysis window in seconds",
)
@click.option("--output", "-o", type=click.Path(), help="Output JSON file for results")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_temporal(file: str, segment_duration: float, output: str, n_fft: int) -> None:
    """Compute temporal acoustic indices (indices over time segments)."""
    from bioamla.cli.console import DIM_STYLE, echo, print_header, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    found = run_operation(
        "indices.temporal",
        file=file,
        window_duration=segment_duration,
        n_fft=n_fft,
        output=output,
    )
    rows = found.segments
    if emit_file_result(file, segments=rows):
        return
    if output:
        print_success(f"Temporal indices saved to {output}")
    else:
        print_header(f"\n{file} - Temporal indices ({segment_duration}s segments):")
        for row in rows[:10]:
            echo(
                f"  Segment {row['segment']} ({row['start_time']:.1f}s): "
                f"ACI={row['aci']:.2f}, ADI={row['adi']:.3f}, "
                f"AEI={row['aei']:.3f}, BIO={row['bio']:.2f}, NDSI={row['ndsi']:.3f}"
            )
        if len(rows) > 10:
            echo(f"  ... and {len(rows) - 10} more segments", style=DIM_STYLE)
        print_header(f"\nTotal segments: {len(rows)}")


@indices.command("aci")
@click.argument("file", type=click.Path(exists=True))
@click.option("--low-freq", "min_freq", default=0.0, type=float, help="Low-frequency bound (Hz)")
@click.option("--high-freq", "max_freq", default=None, type=float, help="High-frequency bound (Hz)")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_aci(file: str, min_freq: float, max_freq: float, n_fft: int) -> None:
    """Compute Acoustic Complexity Index (ACI) only."""
    from bioamla.cli.console import print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    found = run_operation(
        "indices.aci", file=file, min_freq=min_freq, max_freq=max_freq, n_fft=n_fft
    )
    value = found.values["aci"]
    if emit_file_result(file, index="aci", value=value):
        return
    print_kv("ACI", f"{value:.2f}")


@indices.command("adi")
@click.argument("file", type=click.Path(exists=True))
@click.option("--threshold-db", "db_threshold", default=-50.0, type=float, help="dB threshold")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_adi(file: str, db_threshold: float, n_fft: int) -> None:
    """Compute Acoustic Diversity Index (ADI) only."""
    from bioamla.cli.console import print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    found = run_operation("indices.adi", file=file, db_threshold=db_threshold, n_fft=n_fft)
    value = found.values["adi"]
    if emit_file_result(file, index="adi", value=value):
        return
    print_kv("ADI", f"{value:.3f}")


@indices.command("aei")
@click.argument("file", type=click.Path(exists=True))
@click.option("--threshold-db", "db_threshold", default=-50.0, type=float, help="dB threshold")
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_aei(file: str, db_threshold: float, n_fft: int) -> None:
    """Compute Acoustic Evenness Index (AEI) only."""
    from bioamla.cli.console import print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    found = run_operation("indices.aei", file=file, db_threshold=db_threshold, n_fft=n_fft)
    value = found.values["aei"]
    if emit_file_result(file, index="aei", value=value):
        return
    print_kv("AEI", f"{value:.3f}")


@indices.command("bio")
@click.argument("file", type=click.Path(exists=True))
@click.option("--low-freq", "min_freq", default=2000.0, type=float, help="Low-frequency bound (Hz)")
@click.option(
    "--high-freq", "max_freq", default=8000.0, type=float, help="High-frequency bound (Hz)"
)
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_bio(file: str, min_freq: float, max_freq: float, n_fft: int) -> None:
    """Compute Bioacoustic Index (BIO) only."""
    from bioamla.cli.console import print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    found = run_operation(
        "indices.bio", file=file, min_freq=min_freq, max_freq=max_freq, n_fft=n_fft
    )
    value = found.values["bio"]
    if emit_file_result(file, index="bio", value=value):
        return
    print_kv("BIO", f"{value:.2f}")


@indices.command("ndsi")
@click.argument("file", type=click.Path(exists=True))
@click.option("--n-fft", default=1024, type=int, help="FFT window size")
def indices_ndsi(file: str, n_fft: int) -> None:
    """Compute Normalized Difference Soundscape Index (NDSI) only."""
    from bioamla.cli.console import print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    found = run_operation("indices.ndsi", file=file, n_fft=n_fft)
    value = found.values["ndsi"]
    if emit_file_result(file, index="ndsi", value=value):
        return
    print_kv("NDSI", f"{value:.3f}")


@indices.command("entropy")
@click.argument("file", type=click.Path(exists=True))
@click.option("--n-fft", default=512, type=int, help="FFT window size")
def indices_entropy(file: str, n_fft: int) -> None:
    """Compute spectral and temporal entropy indices only."""
    from bioamla.cli.console import print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    found = run_operation("indices.entropy", file=file, n_fft=n_fft)
    h_spectral = found.values["h_spectral"]
    h_temporal = found.values["h_temporal"]
    if emit_file_result(file, indices={"h_spectral": h_spectral, "h_temporal": h_temporal}):
        return
    print_kv("H (spectral)", f"{h_spectral:.3f}")
    print_kv("H (temporal)", f"{h_temporal:.3f}")

"""Advanced acoustic detection algorithms."""

import click


@click.group()
def detect() -> None:
    """Advanced acoustic detection algorithms."""
    pass


@detect.command("energy")
@click.argument("file", type=click.Path(exists=True))
@click.option("--low-freq", "-l", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", "-h", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option(
    "--threshold-db", "-t", "threshold", default=-20.0, type=float, help="Detection threshold (dB)"
)
@click.option("--min-duration", default=0.05, type=float, help="Minimum detection duration (s)")
@click.option("--output", "-o", type=click.Path(), help="Output file for detections")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
def detect_energy(
    file: str,
    low_freq: float,
    high_freq: float,
    threshold: float,
    min_duration: float,
    output: str,
    output_format: str,
) -> None:
    """Detect sounds using band-limited energy detection (single file)."""
    import json as json_lib

    from bioamla.cli.console import DIM_STYLE, echo, print_header, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    found = run_operation(
        "detect.energy",
        file=file,
        low_freq=low_freq,
        high_freq=high_freq,
        threshold_db=threshold,
        min_duration=min_duration,
        output=output,
    )
    detections = found.detections
    rows = [d.to_dict() for d in detections]
    if emit_file_result(file, count=found.count, detections=rows):
        return
    if output:
        print_success(f"Saved {found.count} detections to {output}")
        return
    if output_format == "json":
        click.echo(json_lib.dumps(rows, indent=2))
    elif output_format == "csv":
        import csv
        import sys

        if rows:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        else:
            echo("No detections found.", style=DIM_STYLE)
    else:
        print_header(f"Found {len(detections)} detections:\n")
        for i, d in enumerate(detections, 1):
            echo(f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s (confidence: {d.confidence:.2f})")
        echo(f"\nTotal: {len(detections)} detections")


@detect.command("ribbit")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--pulse-rate",
    "-p",
    default=10.0,
    type=float,
    help="Expected pulse rate in Hz (pulses per second)",
)
@click.option(
    "--tolerance", default=0.2, type=float, help="Tolerance around expected pulse rate (fraction)"
)
@click.option("--low-freq", "-l", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", "-h", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option(
    "--window-seconds", "-w", "window", default=2.0, type=float, help="Analysis window in seconds"
)
@click.option("--min-score", default=0.3, type=float, help="Minimum detection score")
@click.option("--output", "-o", type=click.Path(), help="Output file for detections")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
def detect_ribbit(
    file: str,
    pulse_rate: float,
    tolerance: float,
    low_freq: float,
    high_freq: float,
    window: float,
    min_score: float,
    output: str,
    output_format: str,
) -> None:
    """Detect periodic calls using RIBBIT algorithm (single file)."""
    import json as json_lib

    from bioamla.cli.console import DIM_STYLE, echo, print_header, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    found = run_operation(
        "detect.ribbit",
        file=file,
        pulse_rate=pulse_rate,
        tolerance=tolerance,
        low_freq=low_freq,
        high_freq=high_freq,
        window=window,
        min_score=min_score,
        output=output,
    )
    detections = found.detections
    rows = [d.to_dict() for d in detections]
    if emit_file_result(file, count=found.count, detections=rows):
        return
    if output:
        print_success(f"Saved {found.count} detections to {output}")
        return
    if output_format == "json":
        click.echo(json_lib.dumps(rows, indent=2))
    elif output_format == "csv":
        import csv
        import sys

        if rows:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        else:
            echo("No detections found.", style=DIM_STYLE)
    else:
        print_header(f"Found {len(detections)} periodic call detections:\n")
        for i, d in enumerate(detections, 1):
            echo(
                f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s "
                f"(score: {d.confidence:.2f}, "
                f"pulse_rate: {d.metadata.get('pulse_rate_hz', 'N/A')}Hz)"
            )
        echo(f"\nTotal: {len(detections)} detections")


@detect.command("peaks")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--snr-threshold", "snr", default=2.0, type=float, help="Signal-to-noise ratio threshold"
)
@click.option("--min-distance", default=0.01, type=float, help="Minimum peak distance (s)")
@click.option("--low-freq", "-l", default=None, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", "-h", default=None, type=float, help="High frequency bound (Hz)")
@click.option("--output", "-o", type=click.Path(), help="Output file for detections")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
def detect_peaks(
    file: str,
    snr: float,
    min_distance: float,
    low_freq: float,
    high_freq: float,
    output: str,
    output_format: str,
) -> None:
    """Detect peaks using Continuous Wavelet Transform (CWT) (single file)."""
    import json as json_lib

    from bioamla.cli.console import DIM_STYLE, echo, print_header, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result
    from bioamla.detect import PEAK_EVENT_COLUMNS

    found = run_operation(
        "detect.peaks",
        file=file,
        snr=snr,
        min_distance=min_distance,
        low_freq=low_freq,
        high_freq=high_freq,
        output=output,
    )
    peaks = found.peaks
    rows = [p.to_event() for p in peaks]
    if emit_file_result(file, count=found.count, detections=rows):
        return
    if output:
        print_success(f"Saved {found.count} peaks to {output}")
    elif output_format == "json":
        click.echo(json_lib.dumps([p.to_dict() for p in peaks], indent=2))
    elif output_format == "csv":
        import csv
        import sys

        if rows:
            writer = csv.DictWriter(sys.stdout, fieldnames=PEAK_EVENT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        else:
            echo("No peaks found.", style=DIM_STYLE)
    else:
        print_header(f"Found {len(peaks)} peaks:\n")
        for i, p in enumerate(peaks[:20], 1):
            echo(f"{i}. {p.time:.3f}s (amplitude: {p.amplitude:.2f}, width: {p.width:.3f}s)")
        if len(peaks) > 20:
            echo(f"... and {len(peaks) - 20} more peaks", style=DIM_STYLE)
        echo(f"\nTotal: {len(peaks)} peaks")


@detect.command("accelerating")
@click.argument("file", type=click.Path(exists=True))
@click.option("--min-pulses", default=5, type=int, help="Minimum pulses to detect pattern")
@click.option(
    "--acceleration",
    "-a",
    default=1.5,
    type=float,
    help="Acceleration threshold (final_rate/initial_rate)",
)
@click.option(
    "--deceleration", "-d", default=None, type=float, help="Deceleration threshold (optional)"
)
@click.option("--low-freq", "-l", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", "-h", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option(
    "--window-seconds", "-w", "window", default=3.0, type=float, help="Analysis window in seconds"
)
@click.option("--output", "-o", type=click.Path(), help="Output file for detections")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
def detect_accelerating(
    file: str,
    min_pulses: int,
    acceleration: float,
    deceleration: float,
    low_freq: float,
    high_freq: float,
    window: float,
    output: str,
    output_format: str,
) -> None:
    """Detect accelerating or decelerating call patterns (single file)."""
    import json as json_lib

    from bioamla.cli.console import DIM_STYLE, echo, print_header, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    found = run_operation(
        "detect.accelerating",
        file=file,
        min_pulses=min_pulses,
        acceleration=acceleration,
        deceleration=deceleration,
        low_freq=low_freq,
        high_freq=high_freq,
        window=window,
        output=output,
    )
    detections = found.detections
    rows = [d.to_dict() for d in detections]
    if emit_file_result(file, count=found.count, detections=rows):
        return
    if output:
        print_success(f"Saved {found.count} detections to {output}")
        return
    if output_format == "json":
        click.echo(json_lib.dumps(rows, indent=2))
    elif output_format == "csv":
        import csv
        import sys

        if rows:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        else:
            echo("No detections found.", style=DIM_STYLE)
    else:
        print_header(f"Found {len(detections)} pattern detections:\n")
        for i, d in enumerate(detections, 1):
            pattern = d.metadata.get("pattern_type", "unknown")
            ratio = d.metadata.get("acceleration_ratio", 1.0)
            init_rate = d.metadata.get("initial_rate", 0)
            final_rate = d.metadata.get("final_rate", 0)
            echo(f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s")
            echo(f"   Pattern: {pattern}, ratio: {ratio:.2f}x")
            echo(f"   Rate: {init_rate:.1f} -> {final_rate:.1f} Hz")
        echo(f"\nTotal: {len(detections)} detections")

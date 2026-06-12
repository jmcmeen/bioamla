"""Advanced acoustic detection algorithms."""

import click

from bioamla.exceptions import BioamlaError


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

    from bioamla.detect import BandLimitedEnergyDetector, export_detections

    try:
        detector = BandLimitedEnergyDetector(
            low_freq=low_freq,
            high_freq=high_freq,
            threshold_db=threshold,
            min_duration=min_duration,
        )
        detections = detector.detect_from_file(file)

        if output:
            fmt = "json" if output.endswith(".json") else "csv"
            export_detections(detections, output, format=fmt)
            click.echo(f"Saved {len(detections)} detections to {output}")
            return
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if output_format == "json":
        click.echo(json_lib.dumps([d.to_dict() for d in detections], indent=2))
    elif output_format == "csv":
        import csv
        import sys

        if detections:
            fieldnames = list(detections[0].to_dict().keys())
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for d in detections:
                writer.writerow(d.to_dict())
        else:
            click.echo("No detections found.")
    else:
        click.echo(f"Found {len(detections)} detections:\n")
        for i, d in enumerate(detections, 1):
            click.echo(
                f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s (confidence: {d.confidence:.2f})"
            )
        click.echo(f"\nTotal: {len(detections)} detections")


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

    from bioamla.detect import RibbitDetector, export_detections

    try:
        detector = RibbitDetector(
            pulse_rate_hz=pulse_rate,
            pulse_rate_tolerance=tolerance,
            low_freq=low_freq,
            high_freq=high_freq,
            window_duration=window,
            min_score=min_score,
        )
        detections = detector.detect_from_file(file)

        if output:
            fmt = "json" if output.endswith(".json") else "csv"
            export_detections(detections, output, format=fmt)
            click.echo(f"Saved {len(detections)} detections to {output}")
            return
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if output_format == "json":
        click.echo(json_lib.dumps([d.to_dict() for d in detections], indent=2))
    elif output_format == "csv":
        import csv
        import sys

        if detections:
            fieldnames = list(detections[0].to_dict().keys())
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for d in detections:
                writer.writerow(d.to_dict())
        else:
            click.echo("No detections found.")
    else:
        click.echo(f"Found {len(detections)} periodic call detections:\n")
        for i, d in enumerate(detections, 1):
            click.echo(
                f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s "
                f"(score: {d.confidence:.2f}, "
                f"pulse_rate: {d.metadata.get('pulse_rate_hz', 'N/A')}Hz)"
            )
        click.echo(f"\nTotal: {len(detections)} detections")


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

    from bioamla.detect import CWTPeakDetector

    try:
        detector = CWTPeakDetector(
            snr_threshold=snr,
            min_peak_distance=min_distance,
            low_freq=low_freq,
            high_freq=high_freq,
        )
        peaks = detector.detect_from_file(file)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    fieldnames = ["start_time", "end_time", "confidence", "amplitude", "width"]

    def _row(p):
        return {
            "start_time": p.time,
            "end_time": p.time + p.width,
            "confidence": p.prominence,
            "amplitude": p.amplitude,
            "width": p.width,
        }

    if output:
        import csv
        from pathlib import Path

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in peaks:
                writer.writerow(_row(p))
        click.echo(f"Saved {len(peaks)} peaks to {output}")
    elif output_format == "json":
        click.echo(json_lib.dumps([p.to_dict() for p in peaks], indent=2))
    elif output_format == "csv":
        import csv
        import sys

        if peaks:
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for p in peaks:
                writer.writerow(_row(p))
        else:
            click.echo("No peaks found.")
    else:
        click.echo(f"Found {len(peaks)} peaks:\n")
        for i, p in enumerate(peaks[:20], 1):
            click.echo(f"{i}. {p.time:.3f}s (amplitude: {p.amplitude:.2f}, width: {p.width:.3f}s)")
        if len(peaks) > 20:
            click.echo(f"... and {len(peaks) - 20} more peaks")
        click.echo(f"\nTotal: {len(peaks)} peaks")


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

    from bioamla.detect import AcceleratingPatternDetector, export_detections

    try:
        detector = AcceleratingPatternDetector(
            min_pulses=min_pulses,
            acceleration_threshold=acceleration,
            deceleration_threshold=deceleration,
            low_freq=low_freq,
            high_freq=high_freq,
            window_duration=window,
        )
        detections = detector.detect_from_file(file)

        if output:
            fmt = "json" if output.endswith(".json") else "csv"
            export_detections(detections, output, format=fmt)
            click.echo(f"Saved {len(detections)} detections to {output}")
            return
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if output_format == "json":
        click.echo(json_lib.dumps([d.to_dict() for d in detections], indent=2))
    elif output_format == "csv":
        import csv
        import sys

        if detections:
            fieldnames = list(detections[0].to_dict().keys())
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for d in detections:
                writer.writerow(d.to_dict())
        else:
            click.echo("No detections found.")
    else:
        click.echo(f"Found {len(detections)} pattern detections:\n")
        for i, d in enumerate(detections, 1):
            pattern = d.metadata.get("pattern_type", "unknown")
            ratio = d.metadata.get("acceleration_ratio", 1.0)
            init_rate = d.metadata.get("initial_rate", 0)
            final_rate = d.metadata.get("final_rate", 0)
            click.echo(f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s")
            click.echo(f"   Pattern: {pattern}, ratio: {ratio:.2f}x")
            click.echo(f"   Rate: {init_rate:.1f} -> {final_rate:.1f} Hz")
        click.echo(f"\nTotal: {len(detections)} detections")

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
@click.option("--threshold", "-t", default=-20.0, type=float, help="Detection threshold (dB)")
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

    from bioamla.cli.service_helpers import handle_result, services

    result = services.detection.detect_energy(
        file,
        low_freq=low_freq,
        high_freq=high_freq,
        threshold_db=threshold,
        min_duration=min_duration,
    )
    detection_result = handle_result(result)
    detections = detection_result.detections

    if output:
        fmt = "json" if output.endswith(".json") else "csv"
        services.detection.export_detections(detections, output, format=fmt)
        click.echo(f"Saved {len(detections)} detections to {output}")
    elif output_format == "json":
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
                f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s "
                f"(confidence: {d.confidence:.2f})"
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
@click.option("--window", "-w", default=2.0, type=float, help="Analysis window duration (s)")
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

    from bioamla.cli.service_helpers import handle_result, services

    result = services.detection.detect_ribbit(
        file,
        pulse_rate_hz=pulse_rate,
        pulse_rate_tolerance=tolerance,
        low_freq=low_freq,
        high_freq=high_freq,
        window_duration=window,
        min_score=min_score,
    )
    detection_result = handle_result(result)
    detections = detection_result.detections

    if output:
        fmt = "json" if output.endswith(".json") else "csv"
        services.detection.export_detections(detections, output, format=fmt)
        click.echo(f"Saved {len(detections)} detections to {output}")
    elif output_format == "json":
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
                f"(score: {d.confidence:.2f}, pulse_rate: {d.metadata.get('pulse_rate_hz', 'N/A')}Hz)"
            )
        click.echo(f"\nTotal: {len(detections)} detections")


@detect.command("peaks")
@click.argument("file", type=click.Path(exists=True))
@click.option("--snr", default=2.0, type=float, help="Signal-to-noise ratio threshold")
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

    from bioamla.cli.service_helpers import handle_result, services

    result = services.detection.detect_peaks(
        file,
        snr_threshold=snr,
        min_peak_distance=min_distance,
        low_freq=low_freq,
        high_freq=high_freq,
    )
    detection_result = handle_result(result)
    detections = detection_result.detections

    if output:
        fieldnames = ["start_time", "end_time", "confidence", "amplitude", "width"]
        rows = []
        for d in detections:
            row = {
                "start_time": d.start_time,
                "end_time": d.end_time,
                "confidence": d.confidence,
                "amplitude": d.metadata.get("amplitude", ""),
                "width": d.metadata.get("width", ""),
            }
            rows.append(row)
        services.file.write_csv_dicts(output, rows, fieldnames=fieldnames)
        click.echo(f"Saved {len(detections)} peaks to {output}")
    elif output_format == "json":
        click.echo(json_lib.dumps([d.to_dict() for d in detections], indent=2))
    elif output_format == "csv":
        import csv
        import sys

        if detections:
            fieldnames = ["start_time", "end_time", "confidence", "amplitude", "width"]
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for d in detections:
                row = {
                    "start_time": d.start_time,
                    "end_time": d.end_time,
                    "confidence": d.confidence,
                    "amplitude": d.metadata.get("amplitude", ""),
                    "width": d.metadata.get("width", ""),
                }
                writer.writerow(row)
        else:
            click.echo("No peaks found.")
    else:
        click.echo(f"Found {len(detections)} peaks:\n")
        for i, d in enumerate(detections[:20], 1):
            click.echo(
                f"{i}. {d.start_time:.3f}s (amplitude: {d.metadata.get('amplitude', 0):.2f}, "
                f"width: {d.metadata.get('width', 0):.3f}s)"
            )
        if len(detections) > 20:
            click.echo(f"... and {len(detections) - 20} more peaks")
        click.echo(f"\nTotal: {len(detections)} peaks")


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
@click.option("--window", "-w", default=3.0, type=float, help="Analysis window duration (s)")
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

    from bioamla.cli.service_helpers import handle_result, services

    result = services.detection.detect_accelerating(
        file,
        min_pulses=min_pulses,
        acceleration_threshold=acceleration,
        deceleration_threshold=deceleration,
        low_freq=low_freq,
        high_freq=high_freq,
        window_duration=window,
    )
    detection_result = handle_result(result)
    detections = detection_result.detections

    if output:
        fmt = "json" if output.endswith(".json") else "csv"
        services.detection.export_detections(detections, output, format=fmt)
        click.echo(f"Saved {len(detections)} detections to {output}")
    elif output_format == "json":
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

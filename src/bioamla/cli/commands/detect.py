"""Advanced acoustic detection algorithms."""

import click

from bioamla.services.detection import DetectionService
from bioamla.services.file import FileService


@click.group()
def detect() -> None:
    """Advanced acoustic detection algorithms."""
    pass


@detect.command("energy")
@click.argument("path", type=click.Path(exists=True))
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
def detect_energy(path: str, low_freq: float, high_freq: float, threshold: float, min_duration: float, output: str, output_format: str) -> None:
    """Detect sounds using band-limited energy detection."""
    import json as json_lib
    from pathlib import Path as PathLib

    service = DetectionService()
    path_obj = PathLib(path)
    all_detections = []

    if path_obj.is_dir():
        audio_files = service._get_audio_files(str(path_obj), recursive=True)
        if not audio_files:
            click.echo(f"No audio files found in {path}")
            return

        from bioamla.cli.progress import ProgressBar, print_success

        with ProgressBar(
            total=len(audio_files),
            description="Detecting energy patterns",
        ) as progress:
            for audio_file in audio_files:
                result = service.detect_energy(
                    audio_file,
                    low_freq=low_freq,
                    high_freq=high_freq,
                    threshold_db=threshold,
                    min_duration=min_duration,
                )
                if result.success:
                    for d in result.data.detections:
                        d.metadata["source_file"] = audio_file
                    all_detections.extend(result.data.detections)
                progress.advance()

        print_success(f"Processed {len(audio_files)} files")
    else:
        result = service.detect_energy(
            str(path_obj),
            low_freq=low_freq,
            high_freq=high_freq,
            threshold_db=threshold,
            min_duration=min_duration,
        )
        if not result.success:
            click.echo(f"Error: {result.error}")
            raise SystemExit(1)
        for d in result.data.detections:
            d.metadata["source_file"] = str(path_obj)
        all_detections = result.data.detections

    if output:
        fmt = "json" if output.endswith(".json") else "csv"
        service.export_detections(all_detections, output, format=fmt)
        click.echo(f"Saved {len(all_detections)} detections to {output}")
    elif output_format == "json":
        click.echo(json_lib.dumps([d.to_dict() for d in all_detections], indent=2))
    elif output_format == "csv":
        import csv
        import sys

        if all_detections:
            fieldnames = list(all_detections[0].to_dict().keys())
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for d in all_detections:
                writer.writerow(d.to_dict())
        else:
            click.echo("No detections found.")
    else:
        click.echo(f"Found {len(all_detections)} detections:\n")
        for i, d in enumerate(all_detections, 1):
            source = d.metadata.get("source_file", "")
            if source:
                source = f" [{PathLib(source).name}]"
            click.echo(
                f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s "
                f"(confidence: {d.confidence:.2f}){source}"
            )

    if not output and output_format == "table":
        click.echo(f"\nTotal: {len(all_detections)} detections")


@detect.command("ribbit")
@click.argument("path", type=click.Path(exists=True))
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
    path: str, pulse_rate: float, tolerance: float, low_freq: float, high_freq: float, window: float, min_score: float, output: str, output_format: str
) -> None:
    """Detect periodic calls using RIBBIT algorithm."""
    import json as json_lib
    from pathlib import Path as PathLib

    service = DetectionService()
    path_obj = PathLib(path)
    all_detections = []

    if path_obj.is_dir():
        audio_files = service._get_audio_files(str(path_obj), recursive=True)
        if not audio_files:
            click.echo(f"No audio files found in {path}")
            return

        from bioamla.cli.progress import ProgressBar, print_success

        with ProgressBar(
            total=len(audio_files),
            description="Detecting RIBBIT patterns",
        ) as progress:
            for audio_file in audio_files:
                result = service.detect_ribbit(
                    audio_file,
                    pulse_rate_hz=pulse_rate,
                    pulse_rate_tolerance=tolerance,
                    low_freq=low_freq,
                    high_freq=high_freq,
                    window_duration=window,
                    min_score=min_score,
                )
                if result.success:
                    for d in result.data.detections:
                        d.metadata["source_file"] = audio_file
                    all_detections.extend(result.data.detections)
                progress.advance()

        print_success(f"Processed {len(audio_files)} files")
    else:
        result = service.detect_ribbit(
            str(path_obj),
            pulse_rate_hz=pulse_rate,
            pulse_rate_tolerance=tolerance,
            low_freq=low_freq,
            high_freq=high_freq,
            window_duration=window,
            min_score=min_score,
        )
        if not result.success:
            click.echo(f"Error: {result.error}")
            raise SystemExit(1)
        for d in result.data.detections:
            d.metadata["source_file"] = str(path_obj)
        all_detections = result.data.detections

    if output:
        fmt = "json" if output.endswith(".json") else "csv"
        service.export_detections(all_detections, output, format=fmt)
        click.echo(f"Saved {len(all_detections)} detections to {output}")
    elif output_format == "json":
        click.echo(json_lib.dumps([d.to_dict() for d in all_detections], indent=2))
    elif output_format == "csv":
        import csv
        import sys

        if all_detections:
            fieldnames = list(all_detections[0].to_dict().keys())
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for d in all_detections:
                writer.writerow(d.to_dict())
        else:
            click.echo("No detections found.")
    else:
        click.echo(f"Found {len(all_detections)} periodic call detections:\n")
        for i, d in enumerate(all_detections, 1):
            source = d.metadata.get("source_file", "")
            if source:
                source = f" [{PathLib(source).name}]"
            click.echo(
                f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s "
                f"(score: {d.confidence:.2f}, pulse_rate: {d.metadata.get('pulse_rate_hz', 'N/A')}Hz){source}"
            )

    if not output and output_format == "table":
        click.echo(f"\nTotal: {len(all_detections)} detections")


@detect.command("peaks")
@click.argument("path", type=click.Path(exists=True))
@click.option("--snr", default=2.0, type=float, help="Signal-to-noise ratio threshold")
@click.option("--min-distance", default=0.01, type=float, help="Minimum peak distance (s)")
@click.option("--low-freq", "-l", default=None, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", "-h", default=None, type=float, help="High frequency bound (Hz)")
@click.option("--sequences", is_flag=True, help="Detect peak sequences instead of individual peaks")
@click.option("--min-peaks", default=3, type=int, help="Minimum peaks for sequence detection")
@click.option("--output", "-o", type=click.Path(), help="Output file for detections")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
def detect_peaks(
    path: str, snr: float, min_distance: float, low_freq: float, high_freq: float, sequences: bool, min_peaks: int, output: str, output_format: str
) -> None:
    """Detect peaks using Continuous Wavelet Transform (CWT)."""
    import json as json_lib
    from pathlib import Path as PathLib

    service = DetectionService()
    path_obj = PathLib(path)

    if path_obj.is_dir():
        audio_files = service._get_audio_files(str(path_obj), recursive=True)
        if not audio_files:
            click.echo(f"No audio files found in {path}")
            return
    else:
        audio_files = [str(path_obj)]

    all_detections = []

    if sequences:
        # For sequences, we still need to use core detection for now
        # as the service doesn't have sequence detection
        import librosa

        from bioamla.core.detection import CWTPeakDetector, export_detections

        detector = CWTPeakDetector(
            snr_threshold=snr,
            min_peak_distance=min_distance,
            low_freq=low_freq,
            high_freq=high_freq,
        )

        if len(audio_files) > 1:
            from bioamla.cli.progress import ProgressBar, print_success

            with ProgressBar(
                total=len(audio_files),
                description="Detecting peak sequences",
            ) as progress:
                for audio_file in audio_files:
                    audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)
                    file_detections = detector.detect_sequences(
                        audio, sample_rate, min_peaks=min_peaks
                    )
                    for d in file_detections:
                        d.metadata["source_file"] = audio_file
                    all_detections.extend(file_detections)
                    progress.advance()

            print_success(f"Processed {len(audio_files)} files")
        else:
            for audio_file in audio_files:
                audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)
                file_detections = detector.detect_sequences(audio, sample_rate, min_peaks=min_peaks)
                for d in file_detections:
                    d.metadata["source_file"] = audio_file
                all_detections.extend(file_detections)

        if output:
            fmt = "json" if output.endswith(".json") else "csv"
            export_detections(all_detections, output, format=fmt)
            click.echo(f"Saved {len(all_detections)} sequence detections to {output}")
        elif output_format == "json":
            click.echo(json_lib.dumps([d.to_dict() for d in all_detections], indent=2))
        elif output_format == "csv":
            import csv
            import sys

            if all_detections:
                fieldnames = list(all_detections[0].to_dict().keys())
                writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
                writer.writeheader()
                for d in all_detections:
                    writer.writerow(d.to_dict())
            else:
                click.echo("No detections found.")
        else:
            click.echo(f"Found {len(all_detections)} peak sequences:\n")
            for i, d in enumerate(all_detections, 1):
                n_peaks = d.metadata.get("n_peaks", 0)
                interval = d.metadata.get("mean_interval", 0)
                source = d.metadata.get("source_file", "")
                if source:
                    source = f" [{PathLib(source).name}]"
                click.echo(
                    f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s "
                    f"({n_peaks} peaks, mean interval: {interval:.3f}s){source}"
                )

        if not output and output_format == "table":
            click.echo(f"\nTotal: {len(all_detections)} sequences")
    else:
        # Use service for peak detection
        if len(audio_files) > 1:
            from bioamla.cli.progress import ProgressBar, print_success

            with ProgressBar(
                total=len(audio_files),
                description="Detecting peaks",
            ) as progress:
                for audio_file in audio_files:
                    result = service.detect_peaks(
                        audio_file,
                        snr_threshold=snr,
                        min_peak_distance=min_distance,
                        low_freq=low_freq,
                        high_freq=high_freq,
                    )
                    if result.success:
                        for d in result.data.detections:
                            d.metadata["source_file"] = audio_file
                        all_detections.extend(result.data.detections)
                    progress.advance()

            print_success(f"Processed {len(audio_files)} files")
        else:
            for audio_file in audio_files:
                result = service.detect_peaks(
                    audio_file,
                    snr_threshold=snr,
                    min_peak_distance=min_distance,
                    low_freq=low_freq,
                    high_freq=high_freq,
                )
                if result.success:
                    for d in result.data.detections:
                        d.metadata["source_file"] = audio_file
                    all_detections.extend(result.data.detections)

        if output:
            file_svc = FileService()
            fieldnames = ["start_time", "end_time", "confidence", "amplitude", "width"]
            if len(audio_files) > 1:
                fieldnames.append("source_file")
            rows = []
            for d in all_detections:
                row = {
                    "start_time": d.start_time,
                    "end_time": d.end_time,
                    "confidence": d.confidence,
                    "amplitude": d.metadata.get("amplitude", ""),
                    "width": d.metadata.get("width", ""),
                }
                if len(audio_files) > 1:
                    row["source_file"] = d.metadata.get("source_file", "")
                rows.append(row)
            file_svc.write_csv_dicts(output, rows, fieldnames=fieldnames)
            click.echo(f"Saved {len(all_detections)} peaks to {output}")
        elif output_format == "json":
            click.echo(json_lib.dumps([d.to_dict() for d in all_detections], indent=2))
        elif output_format == "csv":
            import csv
            import sys

            if all_detections:
                fieldnames = ["start_time", "end_time", "confidence", "amplitude", "width"]
                if len(audio_files) > 1:
                    fieldnames.append("source_file")
                writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
                writer.writeheader()
                for d in all_detections:
                    row = {
                        "start_time": d.start_time,
                        "end_time": d.end_time,
                        "confidence": d.confidence,
                        "amplitude": d.metadata.get("amplitude", ""),
                        "width": d.metadata.get("width", ""),
                    }
                    if len(audio_files) > 1:
                        row["source_file"] = d.metadata.get("source_file", "")
                    writer.writerow(row)
            else:
                click.echo("No peaks found.")
        else:
            click.echo(f"Found {len(all_detections)} peaks:\n")
            for i, d in enumerate(all_detections[:20], 1):
                source = d.metadata.get("source_file", "")
                if source and len(audio_files) > 1:
                    source = f" [{PathLib(source).name}]"
                else:
                    source = ""
                click.echo(
                    f"{i}. {d.start_time:.3f}s (amplitude: {d.metadata.get('amplitude', 0):.2f}, "
                    f"width: {d.metadata.get('width', 0):.3f}s){source}"
                )
            if len(all_detections) > 20:
                click.echo(f"... and {len(all_detections) - 20} more peaks")

        if not output and output_format == "table":
            click.echo(f"\nTotal: {len(all_detections)} peaks")


@detect.command("accelerating")
@click.argument("path", type=click.Path(exists=True))
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
    path: str, min_pulses: int, acceleration: float, deceleration: float, low_freq: float, high_freq: float, window: float, output: str, output_format: str
) -> None:
    """Detect accelerating or decelerating call patterns."""
    import json as json_lib
    from pathlib import Path as PathLib

    service = DetectionService()
    path_obj = PathLib(path)
    all_detections = []

    if path_obj.is_dir():
        audio_files = service._get_audio_files(str(path_obj), recursive=True)
        if not audio_files:
            click.echo(f"No audio files found in {path}")
            return

        from bioamla.cli.progress import ProgressBar, print_success

        with ProgressBar(
            total=len(audio_files),
            description="Detecting accelerating patterns",
        ) as progress:
            for audio_file in audio_files:
                result = service.detect_accelerating(
                    audio_file,
                    min_pulses=min_pulses,
                    acceleration_threshold=acceleration,
                    deceleration_threshold=deceleration,
                    low_freq=low_freq,
                    high_freq=high_freq,
                    window_duration=window,
                )
                if result.success:
                    for d in result.data.detections:
                        d.metadata["source_file"] = audio_file
                    all_detections.extend(result.data.detections)
                progress.advance()

        print_success(f"Processed {len(audio_files)} files")
    else:
        result = service.detect_accelerating(
            str(path_obj),
            min_pulses=min_pulses,
            acceleration_threshold=acceleration,
            deceleration_threshold=deceleration,
            low_freq=low_freq,
            high_freq=high_freq,
            window_duration=window,
        )
        if not result.success:
            click.echo(f"Error: {result.error}")
            raise SystemExit(1)
        for d in result.data.detections:
            d.metadata["source_file"] = str(path_obj)
        all_detections = result.data.detections

    if output:
        fmt = "json" if output.endswith(".json") else "csv"
        service.export_detections(all_detections, output, format=fmt)
        click.echo(f"Saved {len(all_detections)} detections to {output}")
    elif output_format == "json":
        click.echo(json_lib.dumps([d.to_dict() for d in all_detections], indent=2))
    elif output_format == "csv":
        import csv
        import sys

        if all_detections:
            fieldnames = list(all_detections[0].to_dict().keys())
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for d in all_detections:
                writer.writerow(d.to_dict())
        else:
            click.echo("No detections found.")
    else:
        click.echo(f"Found {len(all_detections)} pattern detections:\n")
        for i, d in enumerate(all_detections, 1):
            pattern = d.metadata.get("pattern_type", "unknown")
            ratio = d.metadata.get("acceleration_ratio", 1.0)
            init_rate = d.metadata.get("initial_rate", 0)
            final_rate = d.metadata.get("final_rate", 0)
            source = d.metadata.get("source_file", "")
            if source:
                source = f" [{PathLib(source).name}]"
            click.echo(f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s{source}")
            click.echo(f"   Pattern: {pattern}, ratio: {ratio:.2f}x")
            click.echo(f"   Rate: {init_rate:.1f} -> {final_rate:.1f} Hz")

    if not output and output_format == "table":
        click.echo(f"\nTotal: {len(all_detections)} detections")


@detect.command("batch")
@click.argument("directory", type=click.Path(exists=True))
@click.option(
    "--detector",
    "-d",
    type=click.Choice(["energy", "ribbit", "peaks", "accelerating"]),
    default="energy",
    help="Detector type to use",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory for detection files",
)
@click.option("--low-freq", "-l", default=500.0, type=float, help="Low frequency bound (Hz)")
@click.option("--high-freq", "-h", default=5000.0, type=float, help="High frequency bound (Hz)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def detect_batch(directory: str, detector: str, output_dir: str, low_freq: float, high_freq: float, quiet: bool) -> None:
    """Run detection on all audio files in a directory."""
    service = DetectionService()

    result = service.batch_detect(
        directory=directory,
        detector_type=detector,
        output_dir=output_dir,
        low_freq=low_freq,
        high_freq=high_freq,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    batch_result = result.data
    click.echo("\nBatch detection complete:")
    click.echo(f"  Files processed: {batch_result.total_files}")
    click.echo(f"  Total detections: {batch_result.total_detections}")
    click.echo(f"  Output directory: {output_dir}")

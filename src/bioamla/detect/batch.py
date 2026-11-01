"""
Batch detection
===============

Run a single detector over every audio file in a directory, writing an
aggregated JSON of per-file detections. Folds the former
``services/batch_detection.py`` into plain functions built on
:func:`bioamla.batch.run_batch` / :func:`bioamla.batch.discover_files` with
direct ``pathlib`` I/O (no repository DI, no ServiceResult).
"""

import json
from pathlib import Path
from typing import Any

from bioamla.batch import (
    BatchConfig,
    BatchResult,
    config_csv_context,
    merge_analysis_results,
    require_input_dir,
    run_batch,
    write_csv,
)
from bioamla.batch import discover_files as _discover_files
from bioamla.common.constants import SUPPORTED_AUDIO_EXTENSIONS
from bioamla.detect.core import (
    AcceleratingPatternDetector,
    BandLimitedEnergyDetector,
    CWTPeakDetector,
    Detection,
    RibbitDetector,
)
from bioamla.exceptions import InvalidDetectionParams
from bioamla.protocols import OnProgress

__all__ = ["batch_detect_dir", "batch_detect_files", "batch_detection"]

_AUDIO_EXTS = {ext.lower() for ext in SUPPORTED_AUDIO_EXTENSIONS}


def _is_audio(path: Path) -> bool:
    return path.suffix.lower() in _AUDIO_EXTS


def _build_detector(method: str, params: dict[str, Any]):
    """Construct a detector instance for the given method name."""
    if method == "energy":
        return BandLimitedEnergyDetector(
            low_freq=params.get("low_freq", 500.0),
            high_freq=params.get("high_freq", 5000.0),
            threshold_db=params.get("threshold_db", -20.0),
            min_duration=params.get("min_duration", 0.05),
        )
    if method == "ribbit":
        return RibbitDetector(
            pulse_rate_hz=params.get("pulse_rate_hz", 10.0),
            pulse_rate_tolerance=params.get("pulse_rate_tolerance", 0.2),
            low_freq=params.get("low_freq", 500.0),
            high_freq=params.get("high_freq", 5000.0),
            window_duration=params.get("window_duration", 2.0),
            min_score=params.get("min_score", 0.3),
        )
    if method == "peaks":
        return CWTPeakDetector(
            snr_threshold=params.get("snr_threshold", 2.0),
            min_peak_distance=params.get("min_peak_distance", 0.01),
            low_freq=params.get("low_freq"),
            high_freq=params.get("high_freq"),
        )
    if method == "accelerating":
        return AcceleratingPatternDetector(
            min_pulses=params.get("min_pulses", 5),
            acceleration_threshold=params.get("acceleration_threshold", 1.5),
            deceleration_threshold=params.get("deceleration_threshold"),
            low_freq=params.get("low_freq", 500.0),
            high_freq=params.get("high_freq", 5000.0),
            window_duration=params.get("window_duration", 3.0),
        )
    raise InvalidDetectionParams(f"Unknown detection method: {method}")


def batch_detect_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    method: str = "energy",
    *,
    recursive: bool = True,
    continue_on_error: bool = True,
    max_workers: int = 1,
    on_progress: OnProgress | None = None,
    **params: Any,
) -> BatchResult:
    """Run a detector over every audio file in ``input_dir``.

    Discovers audio files, runs the selected detector on each (recording
    per-file success/failure via :func:`bioamla.batch.run_batch`), and writes an
    aggregated ``detections_<method>.json`` under ``output_dir`` mapping each
    file to its list of detection dicts.

    Args:
        input_dir: Directory containing audio files.
        output_dir: Directory where the aggregated JSON is written.
        method: Detector to run ("energy", "ribbit", "peaks", "accelerating").
        recursive: Whether to recurse into subdirectories.
        continue_on_error: Keep going when an individual file fails.
        max_workers: Number of worker threads (1 = sequential).
        on_progress: Optional ``(completed, total)`` progress callback.
        **params: Detector-specific parameters (e.g. low_freq, high_freq).

    Returns:
        A :class:`bioamla.batch.BatchResult`; its ``metadata`` holds the output
        file path and total detection count on success.

    Raises:
        InvalidDetectionParams: If ``method`` is unknown.
    """
    _build_detector(method, params)  # unknown method -> InvalidDetectionParams, before discovery
    files = _discover_files(input_dir, recursive=recursive, file_filter=_is_audio)
    return batch_detect_files(
        files,
        output_dir,
        method,
        continue_on_error=continue_on_error,
        max_workers=max_workers,
        on_progress=on_progress,
        **params,
    )


def batch_detect_files(
    files: list[str | Path],
    output_dir: str | Path,
    method: str = "energy",
    *,
    continue_on_error: bool = True,
    max_workers: int = 1,
    on_progress: OnProgress | None = None,
    **params: Any,
) -> BatchResult:
    """Run a detector over an explicit list of audio files.

    Writes ``<output_dir>/detections_<method>.json`` with one entry per file that
    was processed (in input order), each with its detections. Nothing is written
    when no file succeeds.

    Args:
        files: Audio files to process; a file listed more than once runs once.
        output_dir: Directory where the aggregated JSON is written.
        method: Detector to run ("energy", "ribbit", "peaks", "accelerating").
        continue_on_error: Keep going when an individual file fails.
        max_workers: Number of worker threads (1 = sequential).
        on_progress: Optional ``(completed, total)`` progress callback.
        **params: Detector-specific parameters (e.g. low_freq, high_freq).

    Returns:
        A :class:`bioamla.batch.BatchResult`. Its ``metadata`` holds ``method``,
        ``total_detections``, ``output_file`` (``None`` when nothing was written)
        and ``file_counts`` (file path -> number of detections, for files that
        succeeded).

    Raises:
        InvalidDetectionParams: If ``method`` is unknown.
    """
    detector = _build_detector(method, params)
    files = list(dict.fromkeys(Path(f) for f in files))
    by_path: dict[Path, dict[str, Any]] = {}

    def _process(path: Path) -> str:
        by_path[path] = _detection_record(detector, method, path)
        return str(path)

    result = run_batch(
        files,
        _process,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation="detect",
    )
    # Input order, whichever order the workers finished in.
    aggregated = [by_path[f] for f in files if f in by_path]
    output_path = _write_detections_json(aggregated, output_dir, method)
    result.metadata = {
        "method": method,
        "total_detections": sum(item["num_detections"] for item in aggregated),
        "output_file": str(output_path) if output_path else None,
        "file_counts": {item["filepath"]: item["num_detections"] for item in aggregated},
    }
    return result


def _detection_record(detector: Any, method: str, path: Path) -> dict[str, Any]:
    """Run ``detector`` on one file; return its entry for ``detections_<method>.json``.

    Peaks become detections spanning each peak's width, with its prominence as the
    confidence, so every method's entries have the same shape.
    """
    if method == "peaks":
        detections: list[Detection] = [
            Detection(
                start_time=p.time,
                end_time=p.time + p.width,
                confidence=p.prominence,
                metadata={"amplitude": p.amplitude, "width": p.width},
            )
            for p in detector.detect_from_file(path)
        ]
    else:
        detections = detector.detect_from_file(path)
    return {
        "filepath": str(path),
        "detector_type": method,
        "num_detections": len(detections),
        "detections": [d.to_dict() for d in detections],
    }


def _write_detections_json(
    records: list[dict[str, Any]], output_dir: str | Path, method: str
) -> Path | None:
    """Write ``<output_dir>/detections_<method>.json``; None (nothing written) if empty."""
    if not records:
        return None
    output_path = Path(output_dir) / f"detections_{method}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output_path


def batch_detection(
    config: BatchConfig,
    method: str = "energy",
    *,
    on_progress: OnProgress | None = None,
    **params: Any,
) -> BatchResult:
    """Run one detector over a batch, in directory or CSV mode.

    Detections go to ``detections_<method>.json`` (``metadata["output_file"]``) in
    ``config.output_dir``, or, without one, beside the input: the input folder, or the
    metadata CSV's folder. CSV mode also adds each file's ``num_detections`` to its
    row (blank for a file that failed) and writes the CSV.

    Args:
        config: The batch input and output.
        method: ``energy``, ``ribbit``, ``peaks`` or ``accelerating``.
        on_progress: ``(done, total)`` callback.
        **params: Detector parameters (see :func:`batch_detect_dir`).
    """
    if config.input_file:
        context = config_csv_context(config)
        out_dir = Path(context.output_dir) if context.output_dir else context.csv_path.parent
        result = batch_detect_files(
            [row.file_path for row in context.rows],
            out_dir,
            method,
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            on_progress=on_progress,
            **params,
        )
        counts = result.metadata["file_counts"]
        for row in context.rows:
            # Blank for a file that failed, so no earlier run's count lingers.
            merge_analysis_results(row, {"num_detections": counts.get(str(row.file_path), "")})
        result.metadata["metadata_csv"] = str(write_csv(context))
        return result

    in_dir = require_input_dir(config)
    return batch_detect_dir(
        in_dir,
        str(config.output_dir) if config.output_dir else in_dir,
        method=method,
        recursive=config.recursive,
        continue_on_error=config.continue_on_error,
        max_workers=config.max_workers,
        on_progress=on_progress,
        **params,
    )

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
from typing import Any, Dict, List, Optional, Union

from bioamla.batch import BatchResult, run_batch
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

__all__ = ["batch_detect_dir"]

_AUDIO_EXTS = {ext.lower() for ext in SUPPORTED_AUDIO_EXTENSIONS}


def _is_audio(path: Path) -> bool:
    return path.suffix.lower() in _AUDIO_EXTS


def _build_detector(method: str, params: Dict[str, Any]):
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
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    method: str = "energy",
    *,
    recursive: bool = True,
    continue_on_error: bool = True,
    max_workers: int = 1,
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
        max_workers: Number of worker processes (1 = sequential).
        **params: Detector-specific parameters (e.g. low_freq, high_freq).

    Returns:
        A :class:`bioamla.batch.BatchResult`; its ``metadata`` holds the output
        file path and total detection count on success.

    Raises:
        InvalidDetectionParams: If ``method`` is unknown.
    """
    # Validate method up-front (raises InvalidDetectionParams on unknown).
    detector = _build_detector(method, params)

    files = _discover_files(input_dir, recursive=recursive, file_filter=_is_audio)

    aggregated: List[Dict[str, Any]] = []

    def _process(path: Path) -> str:
        if method == "peaks":
            peaks = detector.detect_from_file(path)
            detections: List[Detection] = [
                Detection(
                    start_time=p.time,
                    end_time=p.time + p.width,
                    confidence=p.prominence,
                    metadata={"amplitude": p.amplitude, "width": p.width},
                )
                for p in peaks
            ]
        else:
            detections = detector.detect_from_file(path)

        aggregated.append(
            {
                "filepath": str(path),
                "detector_type": method,
                "num_detections": len(detections),
                "detections": [d.to_dict() for d in detections],
            }
        )
        return str(path)

    result = run_batch(
        files,
        _process,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
    )

    output_path: Optional[Path] = None
    if aggregated:
        output_path = Path(output_dir) / f"detections_{method}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")

    total_detections = sum(item["num_detections"] for item in aggregated)
    result.metadata = {
        "method": method,
        "total_detections": total_detections,
        "output_file": str(output_path) if output_path else None,
    }

    return result

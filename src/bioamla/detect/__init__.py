"""Advanced acoustic detection algorithms.

Energy, RIBBIT (periodic call), CWT peak-sequence, and accelerating-pattern
detectors. Detectors run on the slim core (librosa + scipy) — no optional
extras required. Functions and detector methods return plain data and raise
:class:`bioamla.exceptions.BioamlaError` subclasses on failure.

Example:
    >>> from bioamla.detect import BandLimitedEnergyDetector
    >>> detector = BandLimitedEnergyDetector(low_freq=500, high_freq=3000)
    >>> detections = detector.detect_from_file("recording.wav")

:func:`detect_file` runs a detector on one file and saves the detections:

    >>> found = detect_file(detector, "recording.wav", output="recording.csv")
"""

from bioamla._extras import require_extra

# Fail early, with the install command, if the [detect] extra is missing.
require_extra("detect", "bioamla.detect")

from bioamla.detect.batch import batch_detect_dir, batch_detect_files, batch_detection
from bioamla.detect.core import (
    PEAK_EVENT_COLUMNS,
    AcceleratingPatternDetector,
    BandLimitedEnergyDetector,
    CWTPeakDetector,
    Detection,
    FileDetections,
    FilePeaks,
    PeakDetection,
    RibbitDetector,
    batch_detect,
    detect_all,
    detect_file,
    detect_peaks_file,
    export_detections,
    export_peaks,
)
from bioamla.exceptions import (
    AudioLoadError,
    DependencyError,
    DetectionError,
    InvalidDetectionParams,
)

__all__ = [
    # Data classes
    "Detection",
    "FileDetections",
    "FilePeaks",
    "PeakDetection",
    "PEAK_EVENT_COLUMNS",
    # Detector classes
    "BandLimitedEnergyDetector",
    "RibbitDetector",
    "CWTPeakDetector",
    "AcceleratingPatternDetector",
    # Convenience functions
    "detect_all",
    "detect_file",
    "detect_peaks_file",
    "export_detections",
    "export_peaks",
    "batch_detect",
    "batch_detect_dir",
    "batch_detect_files",
    "batch_detection",
    # Exceptions
    "AudioLoadError",
    "DependencyError",
    "DetectionError",
    "InvalidDetectionParams",
]

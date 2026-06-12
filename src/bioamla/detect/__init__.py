"""Advanced acoustic detection algorithms.

Energy, RIBBIT (periodic call), CWT peak-sequence, and accelerating-pattern
detectors. Detectors run on the slim core (librosa + scipy) — no optional
extras required. Functions and detector methods return plain data and raise
:class:`bioamla.exceptions.BioamlaError` subclasses on failure.

Example:
    >>> from bioamla.detect import BandLimitedEnergyDetector
    >>> detector = BandLimitedEnergyDetector(low_freq=500, high_freq=3000)
    >>> detections = detector.detect_from_file("recording.wav")
"""

from bioamla.detect.batch import batch_detect_dir
from bioamla.detect.core import (
    AcceleratingPatternDetector,
    BandLimitedEnergyDetector,
    CWTPeakDetector,
    Detection,
    PeakDetection,
    RibbitDetector,
    batch_detect,
    detect_all,
    export_detections,
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
    "PeakDetection",
    # Detector classes
    "BandLimitedEnergyDetector",
    "RibbitDetector",
    "CWTPeakDetector",
    "AcceleratingPatternDetector",
    # Convenience functions
    "detect_all",
    "export_detections",
    "batch_detect",
    "batch_detect_dir",
    # Exceptions
    "AudioLoadError",
    "DependencyError",
    "DetectionError",
    "InvalidDetectionParams",
]

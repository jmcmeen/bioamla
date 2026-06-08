"""Advanced acoustic detection algorithms.

Energy, RIBBIT (periodic call), CWT peak-sequence, and accelerating-pattern
detectors, plus an OpenSoundscape-backed RIBBIT path. Functions and detector
methods return plain data and raise :class:`bioamla.exceptions.BioamlaError`
subclasses on failure.

The OpenSoundscape RIBBIT functions import ``opensoundscape`` lazily; on a slim
install they raise :class:`bioamla.exceptions.DependencyError` pointing the user
at ``bioamla[detect]``.

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
from bioamla.detect.opensoundscape import (
    RIBBIT_PRESETS,
    RibbitDetection,
    create_ribbit_profile,
    get_ribbit_preset,
    list_ribbit_presets,
    ribbit_detect,
    ribbit_detect_preset,
    ribbit_detect_samples,
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
    "RibbitDetection",
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
    # OpenSoundscape RIBBIT
    "ribbit_detect",
    "ribbit_detect_samples",
    "ribbit_detect_preset",
    "list_ribbit_presets",
    "get_ribbit_preset",
    "create_ribbit_profile",
    "RIBBIT_PRESETS",
    # Exceptions
    "AudioLoadError",
    "DependencyError",
    "DetectionError",
    "InvalidDetectionParams",
]

"""
Detection Package
=================

Detection domain for bioacoustic analysis including:
- Various detector implementations (BandLimitedEnergy, RIBBIT, CWT, etc.)
- AST-based detection
- Detection data structures and utilities
"""

from bioamla.core.analysis.detectors import (
    AcceleratingPatternDetector,
    BandLimitedEnergyDetector,
    CWTPeakDetector,
    Detection,
    PeakDetection,
    RibbitDetector,
    batch_detect,
    export_detections,
)

__all__ = [
    # detectors
    "Detection",
    "PeakDetection",
    "BandLimitedEnergyDetector",
    "RibbitDetector",
    "CWTPeakDetector",
    "AcceleratingPatternDetector",
    "batch_detect",
    "export_detections",
]

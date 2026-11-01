"""Operations for the detect domain: thin adapters over :mod:`bioamla.detect`.

See :mod:`bioamla.operations`. Each operation maps its request onto the public
library calls a script would make, and returns their result. Detection on one file
is a ``job``: its cost grows with the recording's length.
"""

from __future__ import annotations

from bioamla.common.config import config_dataclass
from bioamla.detect.core import (
    AcceleratingPatternDetector,
    BandLimitedEnergyDetector,
    CWTPeakDetector,
    FileDetections,
    FilePeaks,
    RibbitDetector,
    detect_file,
    detect_peaks_file,
)
from bioamla.operations import OperationContext, PathStr, operation


@config_dataclass
class EnergyDetectRequest:
    """Band-limited energy detection on one file.

    Attributes:
        file: The audio file.
        low_freq: Low frequency bound (Hz).
        high_freq: High frequency bound (Hz).
        threshold_db: Detection threshold (dB relative to the loudest frame).
        min_duration: Minimum detection duration (s).
        output: Where to save the detections (``.json`` for JSON, otherwise CSV).
    """

    file: PathStr
    low_freq: float = 500.0
    high_freq: float = 5000.0
    threshold_db: float = -20.0
    min_duration: float = 0.05
    output: PathStr | None = None


@operation("detect.energy", request=EnergyDetectRequest, kind="job")
def detect_energy(request: EnergyDetectRequest, context: OperationContext) -> FileDetections:
    """Detect sounds in one file with band-limited energy detection."""
    detector = BandLimitedEnergyDetector(
        low_freq=request.low_freq,
        high_freq=request.high_freq,
        threshold_db=request.threshold_db,
        min_duration=request.min_duration,
    )
    return detect_file(detector, request.file, output=request.output)


@config_dataclass
class RibbitDetectRequest:
    """RIBBIT periodic-call detection on one file.

    Attributes:
        file: The audio file.
        pulse_rate: Expected pulse rate (pulses per second).
        tolerance: Tolerance around the pulse rate, as a fraction.
        low_freq: Low frequency bound (Hz).
        high_freq: High frequency bound (Hz).
        window: Analysis window (s).
        min_score: Minimum detection score.
        output: Where to save the detections (``.json`` for JSON, otherwise CSV).
    """

    file: PathStr
    pulse_rate: float = 10.0
    tolerance: float = 0.2
    low_freq: float = 500.0
    high_freq: float = 5000.0
    window: float = 2.0
    min_score: float = 0.3
    output: PathStr | None = None


@operation("detect.ribbit", request=RibbitDetectRequest, kind="job")
def detect_ribbit(request: RibbitDetectRequest, context: OperationContext) -> FileDetections:
    """Detect periodic calls in one file with RIBBIT."""
    detector = RibbitDetector(
        pulse_rate_hz=request.pulse_rate,
        pulse_rate_tolerance=request.tolerance,
        low_freq=request.low_freq,
        high_freq=request.high_freq,
        window_duration=request.window,
        min_score=request.min_score,
    )
    return detect_file(detector, request.file, output=request.output)


@config_dataclass
class PeaksDetectRequest:
    """CWT peak detection on one file.

    Attributes:
        file: The audio file.
        snr: Signal-to-noise ratio threshold.
        min_distance: Minimum distance between peaks (s).
        low_freq: Low frequency bound (Hz), or ``None`` for no bound.
        high_freq: High frequency bound (Hz), or ``None`` for no bound.
        output: Where to save the peaks (``.json`` for JSON, otherwise CSV).
    """

    file: PathStr
    snr: float = 2.0
    min_distance: float = 0.01
    low_freq: float | None = None
    high_freq: float | None = None
    output: PathStr | None = None


@operation("detect.peaks", request=PeaksDetectRequest, kind="job")
def detect_peaks(request: PeaksDetectRequest, context: OperationContext) -> FilePeaks:
    """Find peaks in one file with the continuous wavelet transform."""
    detector = CWTPeakDetector(
        snr_threshold=request.snr,
        min_peak_distance=request.min_distance,
        low_freq=request.low_freq,
        high_freq=request.high_freq,
    )
    return detect_peaks_file(detector, request.file, output=request.output)


@config_dataclass
class AcceleratingDetectRequest:
    """Accelerating or decelerating call-pattern detection on one file.

    Attributes:
        file: The audio file.
        min_pulses: Minimum pulses in a pattern.
        acceleration: Acceleration threshold (final rate / initial rate).
        deceleration: Deceleration threshold, or ``None`` to find only accelerating
            patterns.
        low_freq: Low frequency bound (Hz).
        high_freq: High frequency bound (Hz).
        window: Analysis window (s).
        output: Where to save the detections (``.json`` for JSON, otherwise CSV).
    """

    file: PathStr
    min_pulses: int = 5
    acceleration: float = 1.5
    deceleration: float | None = None
    low_freq: float = 500.0
    high_freq: float = 5000.0
    window: float = 3.0
    output: PathStr | None = None


@operation("detect.accelerating", request=AcceleratingDetectRequest, kind="job")
def detect_accelerating(
    request: AcceleratingDetectRequest, context: OperationContext
) -> FileDetections:
    """Detect accelerating or decelerating call patterns in one file."""
    detector = AcceleratingPatternDetector(
        min_pulses=request.min_pulses,
        acceleration_threshold=request.acceleration,
        deceleration_threshold=request.deceleration,
        low_freq=request.low_freq,
        high_freq=request.high_freq,
        window_duration=request.window,
    )
    return detect_file(detector, request.file, output=request.output)


__all__ = [
    "AcceleratingDetectRequest",
    "EnergyDetectRequest",
    "PeaksDetectRequest",
    "RibbitDetectRequest",
    "detect_accelerating",
    "detect_energy",
    "detect_peaks",
    "detect_ribbit",
]

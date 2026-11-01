"""Operations for ``batch detect``: thin adapters over :func:`bioamla.detect.batch_detection`.

See :mod:`bioamla.operations` and :class:`~bioamla.batch.BatchRequest`. Each operation
runs one detector over a batch (a folder or a metadata CSV) and returns its
:class:`~bioamla.batch.BatchResult`; detections go to ``detections_<method>.json``.
"""

from __future__ import annotations

from bioamla.batch import BatchRequest, BatchResult
from bioamla.common.config import config_dataclass
from bioamla.detect.batch import batch_detection
from bioamla.operations import OperationContext, operation


@config_dataclass
class BatchEnergyRequest(BatchRequest):
    """Band-limited energy detection over a batch.

    Attributes:
        low_freq: Low frequency bound (Hz).
        high_freq: High frequency bound (Hz).
        threshold_db: Detection threshold (dB).
        min_duration: Minimum detection duration (s).
    """

    low_freq: float = 500.0
    high_freq: float = 5000.0
    threshold_db: float = -20.0
    min_duration: float = 0.05


@operation("batch.detect.energy", request=BatchEnergyRequest, kind="job")
def batch_detect_energy(request: BatchEnergyRequest, context: OperationContext) -> BatchResult:
    """Detect sounds in every file with band-limited energy detection."""
    return batch_detection(
        request.to_config(),
        "energy",
        on_progress=context.progress,
        low_freq=request.low_freq,
        high_freq=request.high_freq,
        threshold_db=request.threshold_db,
        min_duration=request.min_duration,
    )


@config_dataclass
class BatchRibbitRequest(BatchRequest):
    """RIBBIT periodic-call detection over a batch.

    Attributes:
        pulse_rate: Expected pulse rate (Hz).
        pulse_tolerance: Tolerance around the pulse rate, as a fraction.
        low_freq: Low frequency bound (Hz).
        high_freq: High frequency bound (Hz).
        window_duration: Analysis window (s).
        min_score: Minimum detection score.
    """

    pulse_rate: float = 10.0
    pulse_tolerance: float = 0.2
    low_freq: float = 500.0
    high_freq: float = 5000.0
    window_duration: float = 2.0
    min_score: float = 0.3


@operation("batch.detect.ribbit", request=BatchRibbitRequest, kind="job")
def batch_detect_ribbit(request: BatchRibbitRequest, context: OperationContext) -> BatchResult:
    """Detect periodic calls in every file with RIBBIT."""
    return batch_detection(
        request.to_config(),
        "ribbit",
        on_progress=context.progress,
        pulse_rate_hz=request.pulse_rate,
        pulse_rate_tolerance=request.pulse_tolerance,
        low_freq=request.low_freq,
        high_freq=request.high_freq,
        window_duration=request.window_duration,
        min_score=request.min_score,
    )


@config_dataclass
class BatchPeaksRequest(BatchRequest):
    """CWT peak detection over a batch.

    Attributes:
        snr_threshold: Signal-to-noise ratio threshold.
        min_peak_distance: Minimum distance between peaks (s).
        low_freq: Low frequency bound (Hz), or ``None``.
        high_freq: High frequency bound (Hz), or ``None``.
    """

    snr_threshold: float = 2.0
    min_peak_distance: float = 0.01
    low_freq: float | None = None
    high_freq: float | None = None


@operation("batch.detect.peaks", request=BatchPeaksRequest, kind="job")
def batch_detect_peaks(request: BatchPeaksRequest, context: OperationContext) -> BatchResult:
    """Find peaks in every file with the continuous wavelet transform."""
    return batch_detection(
        request.to_config(),
        "peaks",
        on_progress=context.progress,
        snr_threshold=request.snr_threshold,
        min_peak_distance=request.min_peak_distance,
        low_freq=request.low_freq,
        high_freq=request.high_freq,
    )


@config_dataclass
class BatchAcceleratingRequest(BatchRequest):
    """Accelerating or decelerating call-pattern detection over a batch.

    Attributes:
        min_pulses: Minimum pulses in a pattern.
        accel_threshold: Acceleration threshold.
        decel_threshold: Deceleration threshold, or ``None``.
        low_freq: Low frequency bound (Hz).
        high_freq: High frequency bound (Hz).
        window_duration: Analysis window (s).
    """

    min_pulses: int = 5
    accel_threshold: float = 1.5
    decel_threshold: float | None = None
    low_freq: float = 500.0
    high_freq: float = 5000.0
    window_duration: float = 3.0


@operation("batch.detect.accelerating", request=BatchAcceleratingRequest, kind="job")
def batch_detect_accelerating(
    request: BatchAcceleratingRequest, context: OperationContext
) -> BatchResult:
    """Detect accelerating or decelerating call patterns in every file."""
    return batch_detection(
        request.to_config(),
        "accelerating",
        on_progress=context.progress,
        min_pulses=request.min_pulses,
        acceleration_threshold=request.accel_threshold,
        deceleration_threshold=request.decel_threshold,
        low_freq=request.low_freq,
        high_freq=request.high_freq,
        window_duration=request.window_duration,
    )

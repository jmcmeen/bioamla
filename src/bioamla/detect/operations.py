"""Operations for the detect domain: thin adapters over :mod:`bioamla.detect`.

See :mod:`bioamla.operations`. Each operation maps its request onto the public
library calls a script would make, and returns their result.
"""

from __future__ import annotations

from bioamla.common.config import config_dataclass
from bioamla.detect.core import BandLimitedEnergyDetector, FileDetections, detect_file
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


@operation("detect.energy", request=EnergyDetectRequest)
def detect_energy(request: EnergyDetectRequest, context: OperationContext) -> FileDetections:
    """Detect sounds in one file with band-limited energy detection."""
    detector = BandLimitedEnergyDetector(
        low_freq=request.low_freq,
        high_freq=request.high_freq,
        threshold_db=request.threshold_db,
        min_duration=request.min_duration,
    )
    return detect_file(detector, request.file, output=request.output)


__all__ = ["EnergyDetectRequest", "detect_energy"]

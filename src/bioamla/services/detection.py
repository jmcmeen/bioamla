# services/detection.py
"""
Service for acoustic detection operations.
"""

from pathlib import Path
from typing import List, Optional

from bioamla.models.detection import (
    DetectionInfo,
    DetectionResult,
)
from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult


class DetectionService(BaseService):
    """
    Service for acoustic detection operations.

    Provides ServiceResult-wrapped methods for various detection algorithms.
    All file I/O operations are delegated to the file repository.
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize the service.

        Args:
            file_repository: File repository for all file I/O operations (required).
        """
        super().__init__(file_repository)

    def detect_energy(
        self,
        filepath: str,
        low_freq: float = 500.0,
        high_freq: float = 5000.0,
        threshold_db: float = -20.0,
        min_duration: float = 0.05,
    ) -> ServiceResult[DetectionResult]:
        """
        Detect sounds using band-limited energy detection.

        Args:
            filepath: Path to audio file
            low_freq: Low frequency bound (Hz)
            high_freq: High frequency bound (Hz)
            threshold_db: Detection threshold (dB)
            min_duration: Minimum detection duration (s)

        Returns:
            ServiceResult containing DetectionResult
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.detectors import BandLimitedEnergyDetector

            detector = BandLimitedEnergyDetector(
                low_freq=low_freq,
                high_freq=high_freq,
                threshold_db=threshold_db,
                min_duration=min_duration,
            )

            core_detections = detector.detect_from_file(filepath)

            detections = [
                DetectionInfo(
                    start_time=d.start_time,
                    end_time=d.end_time,
                    confidence=d.confidence,
                    label=d.label,
                    metadata=d.metadata,
                )
                for d in core_detections
            ]

            result = DetectionResult(
                filepath=filepath,
                detector_type="energy",
                num_detections=len(detections),
                detections=detections,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Found {len(detections)} detections",
            )
        except Exception as e:
            return ServiceResult.fail(f"Energy detection failed: {e}")

    def detect_ribbit(
        self,
        filepath: str,
        pulse_rate_hz: float = 10.0,
        pulse_rate_tolerance: float = 0.2,
        low_freq: float = 500.0,
        high_freq: float = 5000.0,
        window_duration: float = 2.0,
        min_score: float = 0.3,
    ) -> ServiceResult[DetectionResult]:
        """
        Detect periodic calls using RIBBIT algorithm.

        Args:
            filepath: Path to audio file
            pulse_rate_hz: Expected pulse rate in Hz
            pulse_rate_tolerance: Tolerance around expected pulse rate
            low_freq: Low frequency bound (Hz)
            high_freq: High frequency bound (Hz)
            window_duration: Analysis window duration (s)
            min_score: Minimum detection score

        Returns:
            ServiceResult containing DetectionResult
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.detectors import RibbitDetector

            detector = RibbitDetector(
                pulse_rate_hz=pulse_rate_hz,
                pulse_rate_tolerance=pulse_rate_tolerance,
                low_freq=low_freq,
                high_freq=high_freq,
                window_duration=window_duration,
                min_score=min_score,
            )

            core_detections = detector.detect_from_file(filepath)

            detections = [
                DetectionInfo(
                    start_time=d.start_time,
                    end_time=d.end_time,
                    confidence=d.confidence,
                    label=d.label,
                    metadata=d.metadata,
                )
                for d in core_detections
            ]

            result = DetectionResult(
                filepath=filepath,
                detector_type="ribbit",
                num_detections=len(detections),
                detections=detections,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Found {len(detections)} periodic call detections",
            )
        except Exception as e:
            return ServiceResult.fail(f"RIBBIT detection failed: {e}")

    def detect_peaks(
        self,
        filepath: str,
        snr_threshold: float = 2.0,
        min_peak_distance: float = 0.01,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
    ) -> ServiceResult[DetectionResult]:
        """
        Detect peaks using Continuous Wavelet Transform (CWT).

        Args:
            filepath: Path to audio file
            snr_threshold: Signal-to-noise ratio threshold
            min_peak_distance: Minimum peak distance (s)
            low_freq: Low frequency bound (Hz)
            high_freq: High frequency bound (Hz)

        Returns:
            ServiceResult containing DetectionResult
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.adapters.pydub import load_audio
            from bioamla.core.detectors import CWTPeakDetector

            detector = CWTPeakDetector(
                snr_threshold=snr_threshold,
                min_peak_distance=min_peak_distance,
                low_freq=low_freq,
                high_freq=high_freq,
            )

            audio, sample_rate = load_audio(filepath)
            peaks = detector.detect(audio, sample_rate)

            # Convert peaks to DetectionInfo format
            detections = [
                DetectionInfo(
                    start_time=p.time,
                    end_time=p.time + p.width,
                    confidence=p.prominence,
                    metadata={
                        "amplitude": p.amplitude,
                        "width": p.width,
                    },
                )
                for p in peaks
            ]

            result = DetectionResult(
                filepath=filepath,
                detector_type="peaks",
                num_detections=len(detections),
                detections=detections,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Found {len(detections)} peaks",
            )
        except Exception as e:
            return ServiceResult.fail(f"Peak detection failed: {e}")

    def detect_accelerating(
        self,
        filepath: str,
        min_pulses: int = 5,
        acceleration_threshold: float = 1.5,
        deceleration_threshold: Optional[float] = None,
        low_freq: float = 500.0,
        high_freq: float = 5000.0,
        window_duration: float = 3.0,
    ) -> ServiceResult[DetectionResult]:
        """
        Detect accelerating or decelerating call patterns.

        Args:
            filepath: Path to audio file
            min_pulses: Minimum pulses to detect pattern
            acceleration_threshold: Acceleration threshold (final_rate/initial_rate)
            deceleration_threshold: Deceleration threshold (optional)
            low_freq: Low frequency bound (Hz)
            high_freq: High frequency bound (Hz)
            window_duration: Analysis window duration (s)

        Returns:
            ServiceResult containing DetectionResult
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.detectors import AcceleratingPatternDetector

            detector = AcceleratingPatternDetector(
                min_pulses=min_pulses,
                acceleration_threshold=acceleration_threshold,
                deceleration_threshold=deceleration_threshold,
                low_freq=low_freq,
                high_freq=high_freq,
                window_duration=window_duration,
            )

            core_detections = detector.detect_from_file(filepath)

            detections = [
                DetectionInfo(
                    start_time=d.start_time,
                    end_time=d.end_time,
                    confidence=d.confidence,
                    label=d.label,
                    metadata=d.metadata,
                )
                for d in core_detections
            ]

            result = DetectionResult(
                filepath=filepath,
                detector_type="accelerating",
                num_detections=len(detections),
                detections=detections,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Found {len(detections)} pattern detections",
            )
        except Exception as e:
            return ServiceResult.fail(f"Accelerating pattern detection failed: {e}")

    def export_detections(
        self,
        detections: List[DetectionInfo],
        output_path: str,
        format: str = "csv",
    ) -> ServiceResult[str]:
        """
        Export detections to a file.

        Args:
            detections: List of DetectionInfo objects
            output_path: Output file path
            format: Output format (csv, json)

        Returns:
            ServiceResult containing output path
        """
        try:
            path = Path(output_path)
            self.file_repository.mkdir(path.parent, parents=True)

            if format == "json":
                import json

                content = json.dumps([d.to_dict() for d in detections], indent=2)
                self.file_repository.write_text(path, content)
            else:
                import csv
                from io import StringIO

                if not detections:
                    return ServiceResult.fail("No detections to export")

                # Write CSV to in-memory buffer
                buffer = StringIO()
                fieldnames = ["start_time", "end_time", "confidence", "label"]
                writer = csv.DictWriter(buffer, fieldnames=fieldnames)
                writer.writeheader()
                for d in detections:
                    writer.writerow({
                        "start_time": d.start_time,
                        "end_time": d.end_time,
                        "confidence": d.confidence,
                        "label": d.label,
                    })

                # Write buffer contents to file via repository
                self.file_repository.write_text(path, buffer.getvalue())

            return ServiceResult.ok(
                data=str(path),
                message=f"Exported {len(detections)} detections to {path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to export detections: {e}")

"""Tests for DetectionService - core paths for pre-migration verification."""

import pytest

from bioamla.repository.local import LocalFileRepository
from bioamla.services.detection import DetectionService


class TestDetectionServiceEnergy:
    """Tests for energy detection."""

    def test_detect_energy_basic(self, test_audio_path: str) -> None:
        """Test energy detection runs without error."""
        repository = LocalFileRepository()
        service = DetectionService(repository)

        result = service.detect_energy(
            filepath=test_audio_path,
            low_freq=200.0,
            high_freq=2000.0,
            threshold_db=-30.0,
        )

        assert result.success, f"Energy detection failed: {result.error}"
        assert result.data is not None
        assert result.data.detector_type == "energy"
        assert result.data.filepath == test_audio_path

    def test_detect_energy_returns_detections_list(self, test_audio_path: str) -> None:
        """Test energy detection returns a list of detections."""
        repository = LocalFileRepository()
        service = DetectionService(repository)

        result = service.detect_energy(
            filepath=test_audio_path,
            threshold_db=-50.0,  # Low threshold to ensure detections
        )

        assert result.success
        assert hasattr(result.data, "detections")
        assert isinstance(result.data.detections, list)

    def test_detect_energy_invalid_file(self, mock_repository) -> None:
        """Test energy detection with invalid file path."""
        service = DetectionService(mock_repository)

        result = service.detect_energy(
            filepath="/nonexistent/file.wav",
        )

        assert not result.success
        assert result.error is not None


class TestDetectionServicePeaks:
    """Tests for peak detection."""

    def test_detect_peaks_basic(self, test_audio_path: str) -> None:
        """Test peak detection runs without error."""
        repository = LocalFileRepository()
        service = DetectionService(repository)

        result = service.detect_peaks(
            filepath=test_audio_path,
            snr_threshold=1.0,
        )

        assert result.success, f"Peak detection failed: {result.error}"
        assert result.data is not None
        assert result.data.detector_type == "peaks"

    def test_detect_peaks_with_frequency_band(self, test_audio_path: str) -> None:
        """Test peak detection with frequency band filtering."""
        repository = LocalFileRepository()
        service = DetectionService(repository)

        result = service.detect_peaks(
            filepath=test_audio_path,
            snr_threshold=1.0,
            low_freq=200.0,
            high_freq=2000.0,
        )

        assert result.success, f"Peak detection failed: {result.error}"


class TestDetectionServiceRibbit:
    """Tests for RIBBIT detection."""

    def test_detect_ribbit_basic(self, test_audio_path: str) -> None:
        """Test RIBBIT detection runs without error."""
        repository = LocalFileRepository()
        service = DetectionService(repository)

        result = service.detect_ribbit(
            filepath=test_audio_path,
            pulse_rate_hz=10.0,
            low_freq=500.0,
            high_freq=2000.0,
        )

        assert result.success, f"RIBBIT detection failed: {result.error}"
        assert result.data.detector_type == "ribbit"


class TestDetectionServiceExport:
    """Tests for detection export functionality."""

    def test_export_detections_csv(
        self, test_audio_path: str, tmp_path
    ) -> None:
        """Test exporting detections to CSV."""
        repository = LocalFileRepository()
        service = DetectionService(repository)

        # First get some detections
        detect_result = service.detect_energy(
            filepath=test_audio_path,
            threshold_db=-50.0,
        )

        if detect_result.success and detect_result.data.detections:
            output_path = str(tmp_path / "detections.csv")
            export_result = service.export_detections(
                detections=detect_result.data.detections,
                output_path=output_path,
                format="csv",
            )

            assert export_result.success, f"Export failed: {export_result.error}"

# tests/controllers/test_ribbit.py
"""
Tests for RibbitController.
"""

from unittest.mock import MagicMock, Mock

import pytest

from bioamla.controllers.ribbit import (
    BatchDetectionSummary,
    DetectionSummary,
    RibbitController,
)


class TestRibbitController:
    """Tests for RibbitController."""

    @pytest.fixture
    def controller(self):
        return RibbitController()

    @pytest.fixture
    def mock_detector(self, mocker):
        """Mock the RibbitDetector."""
        mock_detector_class = mocker.patch("bioamla.core.detection.ribbit.RibbitDetector")

        # Create a mock result object
        mock_result = MagicMock()
        mock_result.filepath = "/path/to/audio.wav"
        mock_result.profile_name = "spring_peeper"
        mock_result.num_detections = 3
        mock_result.total_detection_time = 1.5
        mock_result.detection_percentage = 15.0
        mock_result.duration = 10.0
        mock_result.processing_time = 0.5
        mock_result.error = None

        # Create mock detections
        mock_detection = MagicMock()
        mock_detection.start_time = 0.0
        mock_detection.end_time = 0.5
        mock_detection.duration = 0.5
        mock_detection.score = 0.8
        mock_detection.pulse_rate = 25.0
        mock_detection.to_dict.return_value = {
            "start_time": 0.0,
            "end_time": 0.5,
            "score": 0.8,
        }
        mock_result.detections = [mock_detection, mock_detection, mock_detection]

        instance = Mock()
        instance.detect.return_value = mock_result
        mock_detector_class.return_value = instance
        mock_detector_class.from_preset.return_value = instance

        return instance

    def test_detect_valid_file_success(self, controller, tmp_audio_file, mock_detector):
        """Test that detection on a valid audio file succeeds."""
        result = controller.detect(tmp_audio_file, preset="spring_peeper")

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, DetectionSummary)
        assert result.data.num_detections == 3
        assert result.data.profile_name == "spring_peeper"

    def test_detect_nonexistent_file_fails(self, controller):
        """Test that detection on a nonexistent file fails with error."""
        result = controller.detect("/nonexistent/path/audio.wav")

        assert result.success is False
        assert result.error is not None
        assert "does not exist" in result.error

    def test_detect_batch_success(
        self, controller, tmp_dir_with_audio_files, mock_detector, mocker
    ):
        """Test that batch detection on multiple files succeeds."""
        # Mock storage for run tracking
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_complete_run")
        mocker.patch("bioamla.core.files.TextFile")

        result = controller.detect_batch(tmp_dir_with_audio_files, preset="spring_peeper")

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, BatchDetectionSummary)
        assert result.data.total_files == 3


class TestPresetManagement:
    """Tests for preset profile management."""

    @pytest.fixture
    def controller(self):
        return RibbitController()

    def test_list_presets_success(self, controller, mocker):
        """Test that listing presets returns preset list."""
        mock_get_presets = mocker.patch("bioamla.core.detection.ribbit.get_preset_profiles")
        mock_profile = MagicMock()
        mock_profile.species = "Spring Peeper"
        mock_profile.description = "Spring peeper detection"
        mock_profile.signal_band = (2000, 4000)
        mock_profile.pulse_rate_range = (20, 40)

        mock_get_presets.return_value = {"spring_peeper": mock_profile}

        result = controller.list_presets()

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 1
        assert result.data[0]["name"] == "spring_peeper"

    def test_get_preset_not_found_fails(self, controller, mocker):
        """Test that getting nonexistent preset fails."""
        mock_get_presets = mocker.patch("bioamla.core.detection.ribbit.get_preset_profiles")
        mock_get_presets.return_value = {"spring_peeper": MagicMock()}

        result = controller.get_preset("nonexistent_preset")

        assert result.success is False
        assert "Unknown preset" in result.error

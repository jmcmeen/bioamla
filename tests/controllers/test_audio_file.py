# tests/controllers/test_audio_file.py
"""
Tests for AudioFileController.
"""

import numpy as np
import pytest

from bioamla.controllers.audio_file import AudioData, AudioFileController


class TestAudioFileController:
    """Tests for AudioFileController."""

    @pytest.fixture
    def controller(self):
        return AudioFileController()

    def test_open_valid_file_success(self, controller, tmp_audio_file):
        """Test that opening a valid audio file succeeds."""
        result = controller.open(tmp_audio_file)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, AudioData)
        assert result.data.sample_rate == 16000

    def test_open_nonexistent_file_fails(self, controller):
        """Test that opening a nonexistent file fails with error."""
        result = controller.open("/nonexistent/path/audio.wav")

        assert result.success is False
        assert result.error is not None
        assert "does not exist" in result.error

    def test_save_creates_file(self, controller, sample_audio_data, tmp_path):
        """Test that save creates a new audio file."""
        output_path = str(tmp_path / "output.wav")
        result = controller.save(sample_audio_data, output_path)

        assert result.success is True
        assert (tmp_path / "output.wav").exists()


class TestAudioData:
    """Tests for AudioData dataclass."""

    def test_duration_calculated_correctly(self):
        """Test that duration is calculated from samples and sample rate."""
        samples = np.zeros(16000, dtype=np.float32)  # 1 second at 16kHz
        audio = AudioData(samples=samples, sample_rate=16000)

        assert audio.duration == 1.0

    def test_channels_is_one_for_mono(self):
        """Test that channels is 1 for mono audio."""
        samples = np.zeros(16000, dtype=np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, channels=1)

        assert audio.channels == 1

    def test_channels_is_two_for_stereo(self):
        """Test that channels is 2 for stereo audio."""
        samples = np.zeros((16000, 2), dtype=np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, channels=2)

        assert audio.channels == 2

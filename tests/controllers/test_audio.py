# tests/controllers/test_audio.py
"""
Tests for AudioController (legacy batch operations).
"""


import numpy as np
import pytest

from bioamla.controllers.audio import (
    AudioController,
    AudioMetadata,
    ProcessedAudio,
)


class TestAudioController:
    """Tests for AudioController."""

    @pytest.fixture
    def controller(self):
        return AudioController()

    def test_list_files_success(self, controller, tmp_dir_with_audio_files):
        """Test that listing audio files succeeds."""
        result = controller.list_files(tmp_dir_with_audio_files)

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 3

    def test_list_files_nonexistent_dir_fails(self, controller):
        """Test that listing files in nonexistent directory fails."""
        result = controller.list_files("/nonexistent/directory")

        assert result.success is False
        assert "does not exist" in result.error

    def test_get_metadata_success(self, controller, tmp_audio_file, mocker):
        """Test that getting metadata succeeds."""
        mock_get_metadata = mocker.patch("bioamla.core.utils.get_wav_metadata")
        mock_get_metadata.return_value = {
            "duration": 1.0,
            "sample_rate": 16000,
            "channels": 1,
            "bit_depth": 16,
            "format": "WAV",
        }

        result = controller.get_metadata(tmp_audio_file)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, AudioMetadata)
        assert result.data.sample_rate == 16000


class TestSignalProcessing:
    """Tests for signal processing methods."""

    @pytest.fixture
    def controller(self):
        return AudioController()

    def test_resample_success(self, controller, tmp_audio_file, tmp_path, mocker):
        """Test that resampling audio succeeds."""
        mock_load = mocker.patch("bioamla.core.audio.signal.load_audio")
        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        mock_resample = mocker.patch("bioamla.core.audio.signal.resample_audio")
        mock_resample.return_value = np.zeros(8000, dtype=np.float32)

        mocker.patch("bioamla.core.audio.signal.save_audio")

        output_path = str(tmp_path / "output.wav")
        result = controller.resample(tmp_audio_file, output_path, target_rate=8000)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, ProcessedAudio)
        assert result.data.sample_rate == 8000

    def test_resample_nonexistent_file_fails(self, controller, tmp_path):
        """Test that resampling nonexistent file fails."""
        result = controller.resample(
            "/nonexistent/audio.wav",
            str(tmp_path / "output.wav"),
            target_rate=8000,
        )

        assert result.success is False
        assert "does not exist" in result.error

    def test_filter_requires_parameters(self, controller, tmp_audio_file, tmp_path):
        """Test that filter requires at least one filter parameter."""
        output_path = str(tmp_path / "output.wav")
        result = controller.filter_audio(tmp_audio_file, output_path)

        assert result.success is False
        assert "Must specify" in result.error


class TestAnalysis:
    """Tests for audio analysis methods."""

    @pytest.fixture
    def controller(self):
        return AudioController()

    def test_analyze_success(self, controller, tmp_audio_file, mocker):
        """Test that analyzing audio succeeds."""
        mock_load = mocker.patch("bioamla.core.audio.signal.load_audio")
        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        mock_analyze = mocker.patch("bioamla.core.audio.analyze_audio")
        mock_analyze.return_value = {
            "duration": 1.0,
            "channels": 1,
            "rms_db": -20.0,
            "peak_db": -6.0,
            "silence_ratio": 0.1,
            "frequency_stats": {},
        }

        result = controller.analyze(tmp_audio_file)

        assert result.success is True
        assert result.data is not None
        assert result.data.rms_db == -20.0

    def test_analyze_nonexistent_file_fails(self, controller):
        """Test that analyzing nonexistent file fails."""
        result = controller.analyze("/nonexistent/audio.wav")

        assert result.success is False
        assert "does not exist" in result.error


class TestBatchOperations:
    """Tests for batch operations."""

    @pytest.fixture
    def controller(self):
        return AudioController()

    def test_resample_batch_empty_dir_fails(self, controller, tmp_path):
        """Test that batch resample on empty directory fails."""
        result = controller.resample_batch(
            str(tmp_path),
            str(tmp_path / "output"),
            target_rate=8000,
        )

        assert result.success is False
        assert "No audio files found" in result.error

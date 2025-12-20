# tests/controllers/test_audio_transform.py
"""
Tests for AudioTransformController.
"""

import numpy as np
import pytest

from bioamla.controllers.audio_file import AudioData
from bioamla.controllers.audio_transform import AudioTransformController


class TestAudioTransformController:
    """Tests for AudioTransformController."""

    @pytest.fixture
    def controller(self):
        return AudioTransformController()

    @pytest.fixture
    def audio_data(self):
        """Create test audio data with a signal."""
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        samples = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return AudioData(samples=samples, sample_rate=sr)

    def test_normalize_peak_success(self, controller, audio_data):
        """Test that normalize_peak returns normalized audio."""
        result = controller.normalize_peak(audio_data, target_peak=0.9)

        assert result.success is True
        assert result.data is not None
        assert np.max(np.abs(result.data.samples)) == pytest.approx(0.9, rel=0.01)

    def test_resample_changes_sample_rate(self, controller, audio_data):
        """Test that resample changes the sample rate."""
        result = controller.resample(audio_data, target_sample_rate=8000)

        assert result.success is True
        assert result.data.sample_rate == 8000
        # Duration should remain the same
        assert result.data.duration == pytest.approx(1.0, rel=0.01)

    def test_to_mono_converts_stereo(self, controller):
        """Test that to_mono converts stereo to mono."""
        # Create stereo audio
        sr = 16000
        samples = np.random.randn(sr, 2).astype(np.float32) * 0.1
        audio = AudioData(samples=samples, sample_rate=sr, channels=2)

        result = controller.to_mono(audio)

        assert result.success is True
        assert result.data.channels == 1
        assert result.data.samples.ndim == 1

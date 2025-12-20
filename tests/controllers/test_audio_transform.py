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


class TestFilterOperations:
    """Tests for filtering operations."""

    @pytest.fixture
    def controller(self):
        return AudioTransformController()

    @pytest.fixture
    def audio_data(self):
        """Create test audio data."""
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        samples = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return AudioData(samples=samples, sample_rate=sr)

    def test_apply_lowpass_success(self, controller, audio_data, mocker):
        """Test that lowpass filter succeeds."""
        mock_filter = mocker.patch("bioamla.core.audio.signal.lowpass_filter")
        mock_filter.return_value = audio_data.samples * 0.5

        result = controller.apply_lowpass(audio_data, cutoff_hz=2000)

        assert result.success is True
        assert result.data.is_modified is True
        assert "lowpass" in result.data.metadata.get("last_operation", "")

    def test_apply_highpass_success(self, controller, audio_data, mocker):
        """Test that highpass filter succeeds."""
        mock_filter = mocker.patch("bioamla.core.audio.signal.highpass_filter")
        mock_filter.return_value = audio_data.samples * 0.5

        result = controller.apply_highpass(audio_data, cutoff_hz=500)

        assert result.success is True
        assert result.data.is_modified is True
        assert "highpass" in result.data.metadata.get("last_operation", "")

    def test_apply_bandpass_success(self, controller, audio_data, mocker):
        """Test that bandpass filter succeeds."""
        mock_filter = mocker.patch("bioamla.core.audio.signal.bandpass_filter")
        mock_filter.return_value = audio_data.samples * 0.5

        result = controller.apply_bandpass(audio_data, low_hz=500, high_hz=4000)

        assert result.success is True
        assert result.data.is_modified is True
        assert "bandpass" in result.data.metadata.get("last_operation", "")


class TestTrimOperations:
    """Tests for trim operations."""

    @pytest.fixture
    def controller(self):
        return AudioTransformController()

    @pytest.fixture
    def audio_data(self):
        """Create test audio data."""
        sr = 16000
        samples = np.random.randn(sr * 2).astype(np.float32) * 0.1  # 2 seconds
        return AudioData(samples=samples, sample_rate=sr)

    def test_trim_success(self, controller, audio_data, mocker):
        """Test that trim succeeds."""
        mock_trim = mocker.patch("bioamla.core.audio.signal.trim_audio")
        mock_trim.return_value = audio_data.samples[:8000]

        result = controller.trim(audio_data, start_time=0.0, end_time=0.5)

        assert result.success is True
        assert result.data.is_modified is True
        mock_trim.assert_called_once()

    def test_trim_silence_success(self, controller, audio_data, mocker):
        """Test that trim_silence succeeds."""
        mock_trim = mocker.patch("bioamla.core.audio.signal.trim_silence")
        mock_trim.return_value = audio_data.samples[1000:-1000]

        result = controller.trim_silence(audio_data, threshold_db=-40)

        assert result.success is True
        assert "removed_duration" in result.metadata


class TestNoiseReduction:
    """Tests for noise reduction operations."""

    @pytest.fixture
    def controller(self):
        return AudioTransformController()

    @pytest.fixture
    def audio_data(self):
        """Create test audio data with noise."""
        sr = 16000
        samples = np.random.randn(sr).astype(np.float32) * 0.5
        return AudioData(samples=samples, sample_rate=sr)

    def test_denoise_success(self, controller, audio_data, mocker):
        """Test that denoise succeeds."""
        mock_denoise = mocker.patch("bioamla.core.audio.signal.spectral_denoise")
        mock_denoise.return_value = audio_data.samples * 0.5

        result = controller.denoise(audio_data, strength=1.0)

        assert result.success is True
        assert result.data.is_modified is True
        assert "denoise" in result.data.metadata.get("last_operation", "")


class TestGainOperations:
    """Tests for gain operations."""

    @pytest.fixture
    def controller(self):
        return AudioTransformController()

    @pytest.fixture
    def audio_data(self):
        """Create test audio data."""
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        samples = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return AudioData(samples=samples, sample_rate=sr)

    def test_apply_gain_positive(self, controller, audio_data):
        """Test that positive gain increases volume."""
        result = controller.apply_gain(audio_data, gain_db=6.0)

        assert result.success is True
        assert np.max(np.abs(result.data.samples)) > np.max(np.abs(audio_data.samples))

    def test_apply_gain_negative(self, controller, audio_data):
        """Test that negative gain decreases volume."""
        result = controller.apply_gain(audio_data, gain_db=-6.0)

        assert result.success is True
        assert np.max(np.abs(result.data.samples)) < np.max(np.abs(audio_data.samples))

    def test_apply_gain_clips_at_one(self, controller, audio_data):
        """Test that gain is clipped to prevent exceeding +-1."""
        # Apply excessive gain
        result = controller.apply_gain(audio_data, gain_db=30.0)

        assert result.success is True
        assert np.max(np.abs(result.data.samples)) <= 1.0


class TestNormalization:
    """Tests for normalization operations."""

    @pytest.fixture
    def controller(self):
        return AudioTransformController()

    @pytest.fixture
    def audio_data(self):
        """Create test audio data."""
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        samples = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return AudioData(samples=samples, sample_rate=sr)

    def test_normalize_loudness_success(self, controller, audio_data, mocker):
        """Test that loudness normalization succeeds."""
        mock_normalize = mocker.patch("bioamla.core.audio.signal.normalize_loudness")
        mock_normalize.return_value = audio_data.samples * 2

        result = controller.normalize_loudness(audio_data, target_db=-20)

        assert result.success is True
        assert result.data.is_modified is True


class TestChainOperations:
    """Tests for chain operations."""

    @pytest.fixture
    def controller(self):
        return AudioTransformController()

    @pytest.fixture
    def audio_data(self):
        """Create test audio data."""
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        samples = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return AudioData(samples=samples, sample_rate=sr)

    def test_chain_multiple_operations(self, controller, audio_data, mocker):
        """Test that chain applies multiple operations."""
        mock_bandpass = mocker.patch("bioamla.core.audio.signal.bandpass_filter")
        mock_bandpass.return_value = audio_data.samples
        mock_normalize = mocker.patch("bioamla.core.audio.signal.normalize_loudness")
        mock_normalize.return_value = audio_data.samples
        mock_resample = mocker.patch("bioamla.core.audio.signal.resample_audio")
        mock_resample.return_value = np.zeros(8000, dtype=np.float32)

        result = controller.chain(
            audio_data,
            [
                ("apply_bandpass", {"low_hz": 500, "high_hz": 4000}),
                ("normalize_loudness", {"target_db": -20}),
                ("resample", {"target_sample_rate": 8000}),
            ],
        )

        assert result.success is True
        assert "chain" in result.message.lower()

    def test_chain_unknown_operation_fails(self, controller, audio_data):
        """Test that chain fails with unknown operation."""
        result = controller.chain(
            audio_data,
            [("unknown_operation", {})],
        )

        assert result.success is False
        assert "Unknown operation" in result.error

    def test_chain_partial_failure(self, controller, audio_data, mocker):
        """Test that chain fails with unknown operation."""
        mock_bandpass = mocker.patch("bioamla.core.audio.signal.bandpass_filter")
        mock_bandpass.return_value = audio_data.samples

        result = controller.chain(
            audio_data,
            [
                ("apply_bandpass", {"low_hz": 500, "high_hz": 4000}),
                ("unknown_operation", {}),
            ],
        )

        assert result.success is False
        assert "Unknown operation" in result.error


class TestAnalysisOperations:
    """Tests for analysis operations."""

    @pytest.fixture
    def controller(self):
        return AudioTransformController()

    @pytest.fixture
    def audio_data(self):
        """Create test audio data."""
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        samples = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return AudioData(samples=samples, sample_rate=sr)

    def test_get_amplitude_stats_success(self, controller, audio_data, mocker):
        """Test that amplitude stats are returned."""
        from unittest.mock import MagicMock

        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {"peak": 0.5, "rms": 0.35}
        mock_fn = mocker.patch("bioamla.core.audio.audio.get_amplitude_stats")
        mock_fn.return_value = mock_stats

        result = controller.get_amplitude_stats(audio_data)

        assert result.success is True
        assert "peak" in result.data

    def test_get_frequency_stats_success(self, controller, audio_data, mocker):
        """Test that frequency stats are returned."""
        from unittest.mock import MagicMock

        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {"centroid": 440.0, "bandwidth": 100.0}
        mock_fn = mocker.patch("bioamla.core.audio.audio.get_frequency_stats")
        mock_fn.return_value = mock_stats

        result = controller.get_frequency_stats(audio_data)

        assert result.success is True
        assert "centroid" in result.data

    def test_detect_silence_success(self, controller, audio_data, mocker):
        """Test that silence detection works."""
        from unittest.mock import MagicMock

        mock_info = MagicMock()
        mock_info.to_dict.return_value = {"silence_ratio": 0.1, "regions": []}
        mock_fn = mocker.patch("bioamla.core.audio.audio.detect_silence")
        mock_fn.return_value = mock_info

        result = controller.detect_silence(audio_data, threshold_db=-40)

        assert result.success is True


class TestPlaybackPreparation:
    """Tests for playback preparation."""

    @pytest.fixture
    def controller(self):
        return AudioTransformController()

    def test_prepare_for_playback_success(self, controller, mocker):
        """Test that prepare_for_playback prepares audio correctly."""
        sr = 16000
        samples = np.random.randn(sr, 2).astype(np.float32) * 0.3  # Stereo
        audio = AudioData(samples=samples, sample_rate=sr, channels=2)

        mock_resample = mocker.patch("bioamla.core.audio.signal.resample_audio")
        mock_resample.return_value = np.zeros(44100, dtype=np.float32)

        mock_normalize = mocker.patch("bioamla.core.audio.signal.peak_normalize")
        mock_normalize.return_value = np.zeros(44100, dtype=np.float32)

        result = controller.prepare_for_playback(
            audio, target_sample_rate=44100, normalize=True
        )

        assert result.success is True
        assert result.metadata["sample_rate"] == 44100

    def test_to_mono_already_mono(self, controller):
        """Test that to_mono returns same audio if already mono."""
        sr = 16000
        samples = np.random.randn(sr).astype(np.float32) * 0.1
        audio = AudioData(samples=samples, sample_rate=sr, channels=1)

        result = controller.to_mono(audio)

        assert result.success is True
        assert result.message == "Already mono"

"""Tests for OpenSoundscape adapters."""

import numpy as np
import pytest
import scipy.io.wavfile as wav

from bioamla.adapters.opensoundscape import AudioAdapter, SpectrogramAdapter


class TestAudioAdapter:
    """Tests for AudioAdapter class."""

    @pytest.fixture
    def sample_audio_path(self, tmp_path) -> str:
        """Create a temporary test audio file."""
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

        audio_path = tmp_path / "test_audio.wav"
        wav.write(str(audio_path), sample_rate, samples)

        return str(audio_path)

    @pytest.fixture
    def sample_samples(self) -> tuple[np.ndarray, int]:
        """Create sample audio data as numpy array."""
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = 0.5 * np.sin(2 * np.pi * frequency * t)

        return samples, sample_rate

    def test_from_file(self, sample_audio_path: str) -> None:
        """Test loading audio from file."""
        adapter = AudioAdapter.from_file(sample_audio_path)

        assert adapter is not None
        assert adapter.sample_rate == 16000
        assert adapter.duration > 0
        assert len(adapter.samples) > 0

    def test_from_file_with_resample(self, sample_audio_path: str) -> None:
        """Test loading audio with target sample rate."""
        adapter = AudioAdapter.from_file(sample_audio_path, sample_rate=8000)

        assert adapter.sample_rate == 8000
        # Duration should be preserved
        assert abs(adapter.duration - 1.0) < 0.1

    def test_from_samples(self, sample_samples: tuple[np.ndarray, int]) -> None:
        """Test creating adapter from numpy array."""
        samples, sample_rate = sample_samples
        adapter = AudioAdapter.from_samples(samples, sample_rate)

        assert adapter.sample_rate == sample_rate
        assert len(adapter.samples) == len(samples)
        np.testing.assert_array_almost_equal(adapter.samples, samples, decimal=5)

    def test_to_samples(self, sample_samples: tuple[np.ndarray, int]) -> None:
        """Test converting adapter back to numpy array."""
        samples, sample_rate = sample_samples
        adapter = AudioAdapter.from_samples(samples, sample_rate)

        result = adapter.to_samples()

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, samples, decimal=5)

    def test_resample(self, sample_samples: tuple[np.ndarray, int]) -> None:
        """Test resampling audio."""
        samples, sample_rate = sample_samples
        adapter = AudioAdapter.from_samples(samples, sample_rate)

        resampled = adapter.resample(8000)

        assert resampled.sample_rate == 8000
        # Duration should be preserved
        assert abs(resampled.duration - adapter.duration) < 0.1
        # Sample count should be halved (approximately)
        assert len(resampled.samples) < len(samples)

    def test_bandpass(self, sample_samples: tuple[np.ndarray, int]) -> None:
        """Test bandpass filter."""
        samples, sample_rate = sample_samples
        adapter = AudioAdapter.from_samples(samples, sample_rate)

        filtered = adapter.bandpass(300, 600)

        assert filtered.sample_rate == sample_rate
        assert len(filtered.samples) == len(samples)
        # Filtered audio should be different from original
        assert not np.allclose(filtered.samples, samples)

    def test_lowpass(self, sample_samples: tuple[np.ndarray, int]) -> None:
        """Test lowpass filter."""
        samples, sample_rate = sample_samples
        adapter = AudioAdapter.from_samples(samples, sample_rate)

        filtered = adapter.lowpass(1000)

        assert filtered.sample_rate == sample_rate
        assert len(filtered.samples) == len(samples)

    def test_highpass(self, sample_samples: tuple[np.ndarray, int]) -> None:
        """Test highpass filter."""
        samples, sample_rate = sample_samples
        adapter = AudioAdapter.from_samples(samples, sample_rate)

        filtered = adapter.highpass(200)

        assert filtered.sample_rate == sample_rate
        assert len(filtered.samples) == len(samples)

    def test_trim(self, sample_samples: tuple[np.ndarray, int]) -> None:
        """Test trimming audio."""
        samples, sample_rate = sample_samples
        adapter = AudioAdapter.from_samples(samples, sample_rate)

        trimmed = adapter.trim(0.2, 0.8)

        assert trimmed.sample_rate == sample_rate
        # Duration should be approximately 0.6 seconds
        assert abs(trimmed.duration - 0.6) < 0.1
        assert len(trimmed.samples) < len(samples)

    def test_normalize(self, sample_samples: tuple[np.ndarray, int]) -> None:
        """Test audio normalization."""
        samples, sample_rate = sample_samples
        # Scale down to simulate quiet audio
        quiet_samples = samples * 0.1
        adapter = AudioAdapter.from_samples(quiet_samples, sample_rate)

        normalized = adapter.normalize(peak_level=1.0)

        assert normalized.sample_rate == sample_rate
        # Peak should be close to 1.0
        assert abs(np.max(np.abs(normalized.samples)) - 1.0) < 0.01

    def test_method_chaining(self, sample_samples: tuple[np.ndarray, int]) -> None:
        """Test that methods can be chained."""
        samples, sample_rate = sample_samples
        adapter = AudioAdapter.from_samples(samples, sample_rate)

        result = adapter.bandpass(300, 600).resample(8000).normalize(0.8)

        assert result.sample_rate == 8000
        assert abs(np.max(np.abs(result.samples)) - 0.8) < 0.01

    def test_properties(self, sample_samples: tuple[np.ndarray, int]) -> None:
        """Test adapter properties."""
        samples, sample_rate = sample_samples
        adapter = AudioAdapter.from_samples(samples, sample_rate)

        assert adapter.sample_rate == sample_rate
        assert abs(adapter.duration - 1.0) < 0.01
        assert isinstance(adapter.samples, np.ndarray)


class TestSpectrogramAdapter:
    """Tests for SpectrogramAdapter class."""

    @pytest.fixture
    def audio_adapter(self) -> AudioAdapter:
        """Create an AudioAdapter for testing."""
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = 0.5 * np.sin(2 * np.pi * frequency * t)

        return AudioAdapter.from_samples(samples, sample_rate)

    @pytest.fixture
    def sample_audio_path(self, tmp_path) -> str:
        """Create a temporary test audio file."""
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

        audio_path = tmp_path / "test_audio.wav"
        wav.write(str(audio_path), sample_rate, samples)

        return str(audio_path)

    def test_to_mel_spectrogram(self, audio_adapter: AudioAdapter) -> None:
        """Test mel spectrogram generation."""
        spec_adapter = SpectrogramAdapter()

        mel_spec = spec_adapter.to_mel_spectrogram(audio_adapter)

        assert isinstance(mel_spec, np.ndarray)
        assert mel_spec.ndim == 2
        # Default n_mels is 128
        assert mel_spec.shape[0] == 128

    def test_to_mel_spectrogram_custom_params(self, audio_adapter: AudioAdapter) -> None:
        """Test mel spectrogram with custom parameters."""
        spec_adapter = SpectrogramAdapter()

        mel_spec = spec_adapter.to_mel_spectrogram(
            audio_adapter,
            n_mels=64,
            window_samples=256,
            overlap_fraction=0.5,
        )

        assert mel_spec.shape[0] == 64

    def test_to_spectrogram(self, audio_adapter: AudioAdapter) -> None:
        """Test linear spectrogram generation."""
        spec_adapter = SpectrogramAdapter()

        spec = spec_adapter.to_spectrogram(audio_adapter)

        assert isinstance(spec, np.ndarray)
        assert spec.ndim == 2

    def test_mel_from_file(self, sample_audio_path: str) -> None:
        """Test generating mel spectrogram directly from file."""
        mel_spec = SpectrogramAdapter.mel_from_file(sample_audio_path)

        assert isinstance(mel_spec, np.ndarray)
        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] == 128

    def test_mel_from_file_with_params(self, sample_audio_path: str) -> None:
        """Test mel spectrogram from file with custom parameters."""
        mel_spec = SpectrogramAdapter.mel_from_file(
            sample_audio_path,
            sample_rate=8000,
            n_mels=64,
        )

        assert mel_spec.shape[0] == 64


class TestAdapterIntegration:
    """Integration tests for adapter workflow."""

    def test_audio_to_spectrogram_pipeline(self, tmp_path) -> None:
        """Test complete pipeline from file to spectrogram."""
        # Create test file
        sample_rate = 16000
        duration = 2.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

        audio_path = tmp_path / "pipeline_test.wav"
        wav.write(str(audio_path), sample_rate, samples)

        # Pipeline: load -> filter -> resample -> spectrogram
        audio = AudioAdapter.from_file(str(audio_path))
        filtered = audio.bandpass(300, 600)
        resampled = filtered.resample(8000)

        spec_adapter = SpectrogramAdapter()
        mel_spec = spec_adapter.to_mel_spectrogram(resampled, n_mels=64)

        assert isinstance(mel_spec, np.ndarray)
        assert mel_spec.shape[0] == 64

    def test_roundtrip_samples(self) -> None:
        """Test that samples survive roundtrip through adapter."""
        sample_rate = 16000
        original = np.random.randn(sample_rate).astype(np.float32)

        adapter = AudioAdapter.from_samples(original, sample_rate)
        recovered = adapter.to_samples()

        np.testing.assert_array_almost_equal(original, recovered, decimal=5)

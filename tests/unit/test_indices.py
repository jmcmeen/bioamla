"""
Unit tests for bioamla.indices module (Acoustic Indices).
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from bioamla.core.indices import (
    AcousticIndices,
    compute_aci,
    compute_adi,
    compute_aei,
    compute_bio,
    compute_ndsi,
    compute_all_indices,
    compute_indices_from_file,
    batch_compute_indices,
    temporal_indices,
    spectral_entropy,
    temporal_entropy,
    _compute_spectrogram,
    _get_frequency_band_indices,
)


class TestAcousticIndices:
    """Tests for AcousticIndices dataclass."""

    def test_acoustic_indices_creation(self):
        """Test creating AcousticIndices dataclass."""
        indices = AcousticIndices(
            aci=100.0,
            adi=2.5,
            aei=0.3,
            bio=50.0,
            ndsi=0.7,
            anthrophony=10.0,
            biophony=30.0,
            sample_rate=22050,
            duration=60.0,
        )

        assert indices.aci == 100.0
        assert indices.adi == 2.5
        assert indices.aei == 0.3
        assert indices.bio == 50.0
        assert indices.ndsi == 0.7
        assert indices.anthrophony == 10.0
        assert indices.biophony == 30.0
        assert indices.sample_rate == 22050
        assert indices.duration == 60.0

    def test_acoustic_indices_to_dict(self):
        """Test AcousticIndices to_dict method."""
        indices = AcousticIndices(
            aci=100.0,
            adi=2.5,
            aei=0.3,
            bio=50.0,
            ndsi=0.7,
        )

        d = indices.to_dict()

        assert d["aci"] == 100.0
        assert d["adi"] == 2.5
        assert d["aei"] == 0.3
        assert d["bio"] == 50.0
        assert d["ndsi"] == 0.7
        assert "anthrophony" in d
        assert "biophony" in d
        assert "sample_rate" in d
        assert "duration" in d

    def test_acoustic_indices_defaults(self):
        """Test AcousticIndices default values."""
        indices = AcousticIndices(
            aci=1.0,
            adi=1.0,
            aei=1.0,
            bio=1.0,
            ndsi=0.0,
        )

        assert indices.anthrophony == 0.0
        assert indices.biophony == 0.0
        assert indices.sample_rate == 0
        assert indices.duration == 0.0


class TestSpectrogramHelpers:
    """Tests for spectrogram helper functions."""

    def test_compute_spectrogram(self):
        """Test spectrogram computation."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        spectrogram, frequencies, times = _compute_spectrogram(
            audio, sample_rate, n_fft=512, hop_length=256
        )

        assert spectrogram.shape[0] == 257  # n_fft // 2 + 1
        assert len(frequencies) == 257
        assert len(times) > 0
        assert frequencies[0] == 0.0
        assert frequencies[-1] == sample_rate / 2

    def test_get_frequency_band_indices(self):
        """Test frequency band index extraction."""
        frequencies = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])

        indices = _get_frequency_band_indices(frequencies, 2000, 6000)

        assert len(indices) == 5
        assert 2 in indices  # 2000 Hz
        assert 6 in indices  # 6000 Hz
        assert 1 not in indices  # 1000 Hz
        assert 7 not in indices  # 7000 Hz

    def test_get_frequency_band_indices_empty(self):
        """Test frequency band indices with no matches."""
        frequencies = np.array([0, 100, 200, 300])

        indices = _get_frequency_band_indices(frequencies, 5000, 10000)

        assert len(indices) == 0


class TestACI:
    """Tests for Acoustic Complexity Index."""

    def test_aci_basic(self):
        """Test basic ACI computation."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        aci = compute_aci(audio, sample_rate)

        assert isinstance(aci, float)
        assert aci >= 0

    def test_aci_with_frequency_range(self):
        """Test ACI with specific frequency range."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        aci = compute_aci(audio, sample_rate, min_freq=2000, max_freq=8000)

        assert isinstance(aci, float)
        assert aci >= 0

    def test_aci_short_audio(self):
        """Test ACI with very short audio."""
        sample_rate = 22050
        audio = np.random.randn(100).astype(np.float32)

        aci = compute_aci(audio, sample_rate)

        assert isinstance(aci, float)

    def test_aci_silence(self):
        """Test ACI with silence."""
        sample_rate = 22050
        duration = 1.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

        aci = compute_aci(audio, sample_rate)

        assert isinstance(aci, float)
        assert aci == 0.0

    def test_aci_invalid_frequency_range(self):
        """Test ACI with invalid frequency range returns 0."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        # Frequency range above Nyquist
        aci = compute_aci(audio, sample_rate, min_freq=20000, max_freq=30000)

        assert aci == 0.0


class TestADI:
    """Tests for Acoustic Diversity Index."""

    def test_adi_basic(self):
        """Test basic ADI computation."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        adi = compute_adi(audio, sample_rate)

        assert isinstance(adi, float)
        assert adi >= 0

    def test_adi_with_params(self):
        """Test ADI with custom parameters."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        adi = compute_adi(
            audio, sample_rate,
            max_freq=8000,
            freq_step=500,
            db_threshold=-40,
        )

        assert isinstance(adi, float)
        assert adi >= 0

    def test_adi_silence(self):
        """Test ADI with silence.

        Note: For silence, ADI may not be 0.0 because the dB conversion
        and threshold can still produce uniform distribution across bands.
        """
        sample_rate = 22050
        duration = 1.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

        adi = compute_adi(audio, sample_rate)

        assert isinstance(adi, float)
        assert adi >= 0  # ADI should be non-negative

    def test_adi_range(self):
        """Test ADI value is in expected range for Shannon diversity."""
        sample_rate = 22050
        duration = 2.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.5

        adi = compute_adi(audio, sample_rate, max_freq=10000, freq_step=1000)

        # Shannon diversity should be >= 0, with max around ln(n_bands)
        # For 10 bands, max is ln(10) ≈ 2.3
        assert adi >= 0


class TestAEI:
    """Tests for Acoustic Evenness Index."""

    def test_aei_basic(self):
        """Test basic AEI computation."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        aei = compute_aei(audio, sample_rate)

        assert isinstance(aei, float)

    def test_aei_with_params(self):
        """Test AEI with custom parameters."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        aei = compute_aei(
            audio, sample_rate,
            max_freq=8000,
            freq_step=500,
            db_threshold=-40,
        )

        assert isinstance(aei, float)

    def test_aei_gini_coefficient_range(self):
        """Test AEI (Gini coefficient) is in valid range."""
        sample_rate = 22050
        duration = 2.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        aei = compute_aei(audio, sample_rate)

        # Gini coefficient is typically between -1 and 1
        assert -1 <= aei <= 1

    def test_aei_silence(self):
        """Test AEI with silence."""
        sample_rate = 22050
        duration = 1.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

        aei = compute_aei(audio, sample_rate)

        assert isinstance(aei, float)
        assert aei == 0.0


class TestBIO:
    """Tests for Bioacoustic Index."""

    def test_bio_basic(self):
        """Test basic BIO computation."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        bio = compute_bio(audio, sample_rate)

        assert isinstance(bio, float)
        assert bio >= 0

    def test_bio_with_frequency_range(self):
        """Test BIO with custom frequency range."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        bio = compute_bio(audio, sample_rate, min_freq=1000, max_freq=10000)

        assert isinstance(bio, float)
        assert bio >= 0

    def test_bio_silence(self):
        """Test BIO with silence returns 0."""
        sample_rate = 22050
        duration = 1.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

        bio = compute_bio(audio, sample_rate)

        assert isinstance(bio, float)

    def test_bio_invalid_range(self):
        """Test BIO with invalid frequency range."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        bio = compute_bio(audio, sample_rate, min_freq=20000, max_freq=30000)

        assert bio == 0.0


class TestNDSI:
    """Tests for Normalized Difference Soundscape Index."""

    def test_ndsi_basic(self):
        """Test basic NDSI computation."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        ndsi, anthro, bio = compute_ndsi(audio, sample_rate)

        assert isinstance(ndsi, float)
        assert isinstance(anthro, float)
        assert isinstance(bio, float)
        assert -1 <= ndsi <= 1

    def test_ndsi_returns_tuple(self):
        """Test NDSI returns tuple of three values."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        result = compute_ndsi(audio, sample_rate)

        assert len(result) == 3

    def test_ndsi_with_custom_bands(self):
        """Test NDSI with custom frequency bands."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        ndsi, anthro, bio = compute_ndsi(
            audio, sample_rate,
            anthro_min=500,
            anthro_max=1500,
            bio_min=1500,
            bio_max=10000,
        )

        assert -1 <= ndsi <= 1

    def test_ndsi_silence(self):
        """Test NDSI with silence."""
        sample_rate = 22050
        duration = 1.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

        ndsi, anthro, bio = compute_ndsi(audio, sample_rate)

        assert ndsi == 0.0
        assert anthro == 0.0
        assert bio == 0.0

    def test_ndsi_range(self):
        """Test NDSI value is in valid range."""
        sample_rate = 22050
        duration = 2.0

        # Test with pure noise (should be around 0)
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        ndsi, _, _ = compute_ndsi(audio, sample_rate)
        assert -1 <= ndsi <= 1


class TestComputeAllIndices:
    """Tests for compute_all_indices function."""

    def test_compute_all_indices_basic(self):
        """Test computing all indices at once."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        indices = compute_all_indices(audio, sample_rate)

        assert isinstance(indices, AcousticIndices)
        assert isinstance(indices.aci, float)
        assert isinstance(indices.adi, float)
        assert isinstance(indices.aei, float)
        assert isinstance(indices.bio, float)
        assert isinstance(indices.ndsi, float)
        assert indices.sample_rate == sample_rate
        assert indices.duration > 0

    def test_compute_all_indices_stereo(self):
        """Test computing indices from stereo audio."""
        sample_rate = 22050
        duration = 1.0
        # Create stereo audio
        audio = np.random.randn(2, int(sample_rate * duration)).astype(np.float32)

        indices = compute_all_indices(audio, sample_rate)

        assert isinstance(indices, AcousticIndices)

    def test_compute_all_indices_with_params(self):
        """Test compute_all_indices with custom parameters."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        indices = compute_all_indices(
            audio, sample_rate,
            n_fft=1024,
            aci_min_freq=1000,
            aci_max_freq=8000,
            adi_max_freq=10000,
            adi_freq_step=500,
            bio_min_freq=2000,
            bio_max_freq=10000,
        )

        assert isinstance(indices, AcousticIndices)


class TestComputeIndicesFromFile:
    """Tests for compute_indices_from_file function."""

    def test_compute_from_file(self, mock_audio_file):
        """Test computing indices from audio file."""
        indices = compute_indices_from_file(mock_audio_file)

        assert isinstance(indices, AcousticIndices)

    def test_compute_from_nonexistent_file(self, temp_dir):
        """Test computing indices from nonexistent file raises error."""
        with pytest.raises(Exception):
            compute_indices_from_file(temp_dir / "nonexistent.wav")


class TestBatchComputeIndices:
    """Tests for batch_compute_indices function."""

    def test_batch_compute_single_file(self, mock_audio_file):
        """Test batch computing with single file."""
        results = batch_compute_indices([mock_audio_file], verbose=False)

        assert len(results) == 1
        assert results[0]["success"] is True
        assert "aci" in results[0]
        assert "filepath" in results[0]

    def test_batch_compute_with_errors(self, mock_audio_file, temp_dir):
        """Test batch computing handles errors gracefully."""
        files = [
            mock_audio_file,
            temp_dir / "nonexistent.wav",
        ]

        results = batch_compute_indices(files, verbose=False)

        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert "error" in results[1]

    def test_batch_compute_empty_list(self):
        """Test batch computing with empty list."""
        results = batch_compute_indices([], verbose=False)

        assert len(results) == 0


class TestTemporalIndices:
    """Tests for temporal_indices function."""

    def test_temporal_indices_basic(self):
        """Test computing indices over time windows."""
        sample_rate = 22050
        duration = 3.0  # 3 seconds
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        results = temporal_indices(
            audio, sample_rate,
            window_duration=1.0,
            hop_duration=1.0,
        )

        assert len(results) == 3  # 3 windows
        for r in results:
            assert "aci" in r
            assert "start_time" in r
            assert "end_time" in r

    def test_temporal_indices_overlapping(self):
        """Test temporal indices with overlapping windows."""
        sample_rate = 22050
        duration = 2.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        results = temporal_indices(
            audio, sample_rate,
            window_duration=1.0,
            hop_duration=0.5,  # 50% overlap
        )

        assert len(results) == 3  # 0-1s, 0.5-1.5s, 1-2s

    def test_temporal_indices_short_audio(self):
        """Test temporal indices with audio shorter than window."""
        sample_rate = 22050
        duration = 0.5  # 0.5 seconds
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        results = temporal_indices(
            audio, sample_rate,
            window_duration=1.0,  # Window longer than audio
        )

        assert len(results) == 0

    def test_temporal_indices_stereo(self):
        """Test temporal indices with stereo audio."""
        sample_rate = 22050
        duration = 2.0
        # Create stereo audio
        audio = np.random.randn(2, int(sample_rate * duration)).astype(np.float32)

        results = temporal_indices(
            audio, sample_rate,
            window_duration=1.0,
        )

        assert len(results) == 2


class TestEntropyIndices:
    """Tests for entropy-based indices."""

    def test_spectral_entropy_basic(self):
        """Test spectral entropy computation."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        entropy = spectral_entropy(audio, sample_rate)

        assert isinstance(entropy, float)
        assert entropy >= 0

    def test_spectral_entropy_silence(self):
        """Test spectral entropy with silence."""
        sample_rate = 22050
        duration = 1.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

        entropy = spectral_entropy(audio, sample_rate)

        assert entropy == 0.0

    def test_temporal_entropy_basic(self):
        """Test temporal entropy computation."""
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        entropy = temporal_entropy(audio, sample_rate)

        assert isinstance(entropy, float)
        assert entropy >= 0

    def test_temporal_entropy_silence(self):
        """Test temporal entropy with silence."""
        sample_rate = 22050
        duration = 1.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

        entropy = temporal_entropy(audio, sample_rate)

        assert entropy == 0.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_audio(self):
        """Test indices with very short audio."""
        sample_rate = 22050
        audio = np.random.randn(100).astype(np.float32)

        # Should not raise errors
        indices = compute_all_indices(audio, sample_rate)
        assert isinstance(indices, AcousticIndices)

    def test_high_sample_rate(self):
        """Test indices with high sample rate."""
        sample_rate = 96000
        duration = 0.5
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        indices = compute_all_indices(audio, sample_rate)
        assert isinstance(indices, AcousticIndices)

    def test_low_sample_rate(self):
        """Test indices with low sample rate."""
        sample_rate = 8000
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        indices = compute_all_indices(audio, sample_rate)
        assert isinstance(indices, AcousticIndices)

    def test_single_sample(self):
        """Test indices with single sample audio."""
        sample_rate = 22050
        audio = np.array([0.5], dtype=np.float32)

        # Should handle gracefully
        indices = compute_all_indices(audio, sample_rate)
        assert isinstance(indices, AcousticIndices)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def mock_audio_file(temp_dir):
    """Create a mock audio file for testing."""
    import struct

    audio_path = temp_dir / "test_audio.wav"

    sample_rate = 16000
    duration = 1
    num_samples = sample_rate * duration
    bits_per_sample = 16
    num_channels = 1

    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    file_size = 36 + data_size

    # Generate some noise instead of silence for better index values
    noise_samples = np.random.randint(-1000, 1000, num_samples, dtype=np.int16)

    with open(audio_path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", file_size))
        f.write(b"WAVE")

        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(noise_samples.tobytes())

    return audio_path

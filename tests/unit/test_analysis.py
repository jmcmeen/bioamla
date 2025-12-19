"""
Unit tests for bioamla.analysis module.
"""

import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bioamla.services.helpers.analysis import (
    AmplitudeStats,
    AudioAnalysis,
    AudioInfo,
    FrequencyStats,
    SilenceInfo,
    analyze_audio,
    calculate_dbfs,
    calculate_peak,
    calculate_rms,
    detect_silence,
    get_amplitude_stats,
    get_audio_info,
    get_channels,
    get_duration,
    get_frequency_stats,
    get_peak_frequency,
    get_sample_rate,
    is_silent,
    summarize_analysis,
)


class TestAudioInfo:
    """Tests for basic audio information extraction."""

    def test_get_audio_info(self, mock_audio_file):
        """Test getting audio info from file."""
        info = get_audio_info(str(mock_audio_file))

        assert isinstance(info, AudioInfo)
        assert info.sample_rate == 16000
        assert info.channels == 1
        assert info.duration > 0
        assert info.samples > 0

    def test_get_audio_info_stereo(self, temp_dir):
        """Test getting info from stereo file."""
        filepath = temp_dir / "stereo.wav"
        audio = np.random.randn(2, 16000).astype(np.float32)
        sf.write(str(filepath), audio.T, 16000)

        info = get_audio_info(str(filepath))
        assert info.channels == 2

    def test_get_audio_info_missing_file(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            get_audio_info("/nonexistent/file.wav")

    def test_get_duration(self, mock_audio_file):
        """Test get_duration convenience function."""
        duration = get_duration(str(mock_audio_file))
        assert duration > 0
        assert isinstance(duration, float)

    def test_get_sample_rate(self, mock_audio_file):
        """Test get_sample_rate convenience function."""
        sr = get_sample_rate(str(mock_audio_file))
        assert sr == 16000

    def test_get_channels(self, mock_audio_file):
        """Test get_channels convenience function."""
        channels = get_channels(str(mock_audio_file))
        assert channels == 1

    def test_audio_info_to_dict(self, mock_audio_file):
        """Test AudioInfo to_dict method."""
        info = get_audio_info(str(mock_audio_file))
        d = info.to_dict()

        assert "duration" in d
        assert "sample_rate" in d
        assert "channels" in d
        assert d["sample_rate"] == 16000


class TestAmplitudeAnalysis:
    """Tests for amplitude analysis functions."""

    def test_calculate_rms_sine(self):
        """Test RMS calculation for sine wave."""
        # Sine wave RMS should be peak / sqrt(2)
        t = np.linspace(0, 1, 16000)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        rms = calculate_rms(audio)
        expected_rms = 1.0 / np.sqrt(2)  # ~0.707

        assert np.isclose(rms, expected_rms, atol=0.01)

    def test_calculate_rms_silence(self):
        """Test RMS of silent audio."""
        audio = np.zeros(16000, dtype=np.float32)
        rms = calculate_rms(audio)
        assert rms == 0.0

    def test_calculate_rms_stereo(self):
        """Test RMS calculation handles stereo."""
        audio = np.random.randn(2, 16000).astype(np.float32) * 0.5
        rms = calculate_rms(audio)
        assert rms > 0

    def test_calculate_dbfs(self):
        """Test dBFS conversion."""
        # Full scale (1.0) should be 0 dBFS
        assert calculate_dbfs(1.0) == 0.0

        # Half amplitude should be about -6 dBFS
        dbfs = calculate_dbfs(0.5)
        assert np.isclose(dbfs, -6.02, atol=0.1)

        # Quarter amplitude should be about -12 dBFS
        dbfs = calculate_dbfs(0.25)
        assert np.isclose(dbfs, -12.04, atol=0.1)

    def test_calculate_dbfs_zero(self):
        """Test dBFS of zero amplitude."""
        dbfs = calculate_dbfs(0.0)
        assert dbfs == -np.inf

    def test_calculate_peak(self):
        """Test peak calculation."""
        audio = np.array([0.1, -0.5, 0.3, 0.8, -0.2], dtype=np.float32)
        peak = calculate_peak(audio)
        assert np.isclose(peak, 0.8, atol=1e-6)

    def test_calculate_peak_stereo(self):
        """Test peak calculation with stereo."""
        audio = np.array([[0.5, -0.3], [0.2, 0.9]], dtype=np.float32)
        peak = calculate_peak(audio)
        assert np.isclose(peak, 0.9, atol=1e-6)

    def test_get_amplitude_stats(self):
        """Test comprehensive amplitude stats."""
        t = np.linspace(0, 1, 16000)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        stats = get_amplitude_stats(audio)

        assert isinstance(stats, AmplitudeStats)
        assert 0 < stats.rms < 1
        assert stats.peak == 0.5
        assert stats.rms_db < 0
        assert stats.peak_db < 0
        assert stats.crest_factor >= 0

    def test_amplitude_stats_to_dict(self):
        """Test AmplitudeStats to_dict method."""
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        stats = get_amplitude_stats(audio)
        d = stats.to_dict()

        assert "rms" in d
        assert "rms_db" in d
        assert "peak" in d
        assert "peak_db" in d


class TestFrequencyAnalysis:
    """Tests for frequency analysis functions."""

    def test_get_peak_frequency_sine(self):
        """Test peak frequency detection for pure sine."""
        sr = 16000
        freq = 440
        t = np.linspace(0, 1, sr)
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

        peak_freq, peak_mag = get_peak_frequency(audio, sr)

        # Should be close to 440 Hz
        assert np.isclose(peak_freq, freq, atol=20)
        assert peak_mag > 0

    def test_get_peak_frequency_with_range(self):
        """Test peak frequency with frequency range filter."""
        sr = 16000
        t = np.linspace(0, 1, sr)
        # Mix of 200Hz and 2000Hz
        audio = (np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)

        # Should find 200Hz when filtering to low frequencies
        peak_freq, _ = get_peak_frequency(audio, sr, min_freq=100, max_freq=500)
        assert np.isclose(peak_freq, 200, atol=20)

    def test_get_frequency_stats(self):
        """Test comprehensive frequency stats."""
        sr = 16000
        t = np.linspace(0, 1, sr)
        audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        stats = get_frequency_stats(audio, sr)

        assert isinstance(stats, FrequencyStats)
        assert stats.peak_frequency > 0
        assert stats.spectral_centroid > 0
        assert stats.bandwidth >= 0

    def test_get_frequency_stats_silent(self):
        """Test frequency stats for silent audio."""
        audio = np.zeros(16000, dtype=np.float32)
        stats = get_frequency_stats(audio, 16000)

        assert stats.peak_frequency == 0.0
        assert stats.spectral_centroid == 0.0

    def test_frequency_stats_to_dict(self):
        """Test FrequencyStats to_dict method."""
        audio = np.random.randn(16000).astype(np.float32)
        stats = get_frequency_stats(audio, 16000)
        d = stats.to_dict()

        assert "peak_frequency" in d
        assert "spectral_centroid" in d
        assert "bandwidth" in d


class TestSilenceDetection:
    """Tests for silence detection functions."""

    def test_detect_silence_silent_audio(self):
        """Test detection of completely silent audio."""
        audio = np.zeros(16000, dtype=np.float32)

        result = detect_silence(audio, 16000)

        assert isinstance(result, SilenceInfo)
        assert result.is_silent is True
        assert result.silence_ratio > 0.9

    def test_detect_silence_loud_audio(self):
        """Test detection on loud audio."""
        t = np.linspace(0, 1, 16000)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        result = detect_silence(audio, 16000)

        assert result.is_silent is False
        assert result.sound_ratio > 0.5

    def test_detect_silence_mixed(self, temp_dir):
        """Test detection on audio with silence and sound."""
        sr = 16000
        # 0.5s silence, 1s tone, 0.5s silence
        silence1 = np.zeros(int(0.5 * sr), dtype=np.float32)
        tone = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))).astype(np.float32)
        silence2 = np.zeros(int(0.5 * sr), dtype=np.float32)
        audio = np.concatenate([silence1, tone, silence2])

        result = detect_silence(audio, sr, threshold_db=-40)

        assert result.is_silent is False
        assert len(result.sound_segments) >= 1
        assert result.silence_ratio > 0
        assert result.sound_ratio > 0

    def test_is_silent_function(self):
        """Test is_silent convenience function."""
        silent_audio = np.zeros(16000, dtype=np.float32)
        assert is_silent(silent_audio) is True

        loud_audio = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))).astype(np.float32)
        assert is_silent(loud_audio) is False

    def test_silence_info_to_dict(self):
        """Test SilenceInfo to_dict method."""
        audio = np.zeros(16000, dtype=np.float32)
        result = detect_silence(audio, 16000)
        d = result.to_dict()

        assert "is_silent" in d
        assert "silence_ratio" in d
        assert "sound_segments" in d


class TestCompleteAnalysis:
    """Tests for complete audio analysis."""

    def test_analyze_audio(self, mock_audio_file):
        """Test complete audio analysis."""
        analysis = analyze_audio(str(mock_audio_file))

        assert isinstance(analysis, AudioAnalysis)
        assert isinstance(analysis.info, AudioInfo)
        assert isinstance(analysis.amplitude, AmplitudeStats)
        assert isinstance(analysis.frequency, FrequencyStats)
        assert isinstance(analysis.silence, SilenceInfo)
        assert analysis.file_path == str(mock_audio_file)

    def test_analyze_audio_missing_file(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            analyze_audio("/nonexistent/file.wav")

    def test_analysis_to_dict(self, mock_audio_file):
        """Test AudioAnalysis to_dict method."""
        analysis = analyze_audio(str(mock_audio_file))
        d = analysis.to_dict()

        assert "file_path" in d
        assert "info" in d
        assert "amplitude" in d
        assert "frequency" in d
        assert "silence" in d

    def test_summarize_analysis(self, temp_dir):
        """Test summarizing multiple analyses."""
        # Create test files
        analyses = []
        for i in range(3):
            filepath = temp_dir / f"test_{i}.wav"
            audio = np.random.randn(16000 * (i + 1)).astype(np.float32) * 0.3
            sf.write(str(filepath), audio, 16000)
            analyses.append(analyze_audio(str(filepath)))

        summary = summarize_analysis(analyses)

        assert summary["total_files"] == 3
        assert summary["total_duration"] > 0
        assert summary["avg_duration"] > 0
        assert "avg_rms_db" in summary
        assert "silent_file_count" in summary

    def test_summarize_empty_list(self):
        """Test summarizing empty list."""
        summary = summarize_analysis([])
        assert summary["total_files"] == 0
        assert summary["total_duration"] == 0.0


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
    filepath = temp_dir / "test.wav"

    # Create a simple sine wave
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    sf.write(str(filepath), audio, sr)
    return filepath

"""Coverage tests for bioamla.audio.analysis."""

import numpy as np
import pytest

from bioamla.audio import AudioData
from bioamla.audio.analysis import (
    calculate_dbfs,
    calculate_peak,
    calculate_rms,
    detect_silence,
    get_amplitude_stats,
    get_frequency_stats,
    get_peak_frequency,
    is_silent,
)


class TestCalculations:
    def test_calculate_rms_2d_flattens(self) -> None:
        audio = np.ones((100, 2), dtype=np.float32) * 0.5
        assert calculate_rms(audio) == pytest.approx(0.5, abs=1e-3)

    def test_calculate_dbfs_zero_is_neg_inf(self) -> None:
        assert calculate_dbfs(0.0) == -np.inf

    def test_calculate_dbfs_full_scale(self) -> None:
        assert calculate_dbfs(1.0) == pytest.approx(0.0)

    def test_calculate_peak_2d_flattens(self) -> None:
        audio = np.array([[0.1, -0.9], [0.2, 0.3]], dtype=np.float32)
        assert calculate_peak(audio) == pytest.approx(0.9)


class TestAmplitudeStats:
    def test_silent_audio_zero_crest(self) -> None:
        stats = get_amplitude_stats(np.zeros(1000, dtype=np.float32))
        assert stats.crest_factor == 0.0
        assert stats.rms == 0.0

    def test_signal(self, sample_audio_data: AudioData) -> None:
        stats = get_amplitude_stats(sample_audio_data.samples)
        assert stats.crest_factor > 0


class TestPeakFrequency:
    def test_peak_near_440(self, sample_audio_data: AudioData) -> None:
        freq, mag = get_peak_frequency(sample_audio_data.samples, sample_audio_data.sample_rate)
        assert 300 < freq < 600
        assert mag > 0

    def test_stereo_input(self, sample_audio_stereo: AudioData) -> None:
        freq, _ = get_peak_frequency(sample_audio_stereo.samples, sample_audio_stereo.sample_rate)
        assert freq > 0

    def test_freq_range_filter(self, sample_audio_data: AudioData) -> None:
        freq, _ = get_peak_frequency(
            sample_audio_data.samples,
            sample_audio_data.sample_rate,
            min_freq=400,
            max_freq=500,
        )
        assert 400 <= freq <= 500

    def test_empty_mask_returns_zero(self, sample_audio_data: AudioData) -> None:
        freq, mag = get_peak_frequency(
            sample_audio_data.samples,
            sample_audio_data.sample_rate,
            min_freq=100000,
            max_freq=200000,
        )
        assert freq == 0.0 and mag == 0.0


class TestFrequencyStats:
    def test_signal(self, sample_audio_data: AudioData) -> None:
        stats = get_frequency_stats(sample_audio_data.samples, sample_audio_data.sample_rate)
        assert stats.peak_frequency > 0
        assert stats.spectral_centroid > 0
        assert stats.bandwidth >= 0

    def test_silent_returns_zeros(self) -> None:
        stats = get_frequency_stats(np.zeros(4096, dtype=np.float32), 16000)
        assert stats.peak_frequency == 0.0
        assert stats.spectral_centroid == 0.0

    def test_stereo_input(self, sample_audio_stereo: AudioData) -> None:
        stats = get_frequency_stats(sample_audio_stereo.samples, sample_audio_stereo.sample_rate)
        assert stats.peak_frequency > 0


class TestDetectSilence:
    def test_with_silence(self, sample_audio_with_silence: AudioData) -> None:
        info = detect_silence(
            sample_audio_with_silence.samples,
            sample_audio_with_silence.sample_rate,
        )
        assert info.silence_ratio > 0
        assert info.threshold_used == -40

    def test_all_silent(self) -> None:
        info = detect_silence(np.zeros(16000, dtype=np.float32), 16000)
        assert info.is_silent is True

    def test_short_audio_branch_silent(self) -> None:
        info = detect_silence(np.zeros(100, dtype=np.float32), 16000)
        assert info.is_silent is True
        assert info.silence_ratio == 1.0

    def test_short_audio_branch_loud(self) -> None:
        loud = np.ones(100, dtype=np.float32) * 0.9
        info = detect_silence(loud, 16000)
        assert info.is_silent is False
        assert len(info.sound_segments) == 1

    def test_stereo_input(self, sample_audio_stereo: AudioData) -> None:
        info = detect_silence(sample_audio_stereo.samples, sample_audio_stereo.sample_rate)
        assert info is not None


class TestIsSilent:
    def test_true_for_zeros(self) -> None:
        assert is_silent(np.zeros(1000, dtype=np.float32)) is True

    def test_false_for_signal(self, sample_audio_data: AudioData) -> None:
        assert is_silent(sample_audio_data.samples) is False

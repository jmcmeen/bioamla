"""Tests for the audio domain (flattened, exception-based API)."""

import numpy as np
import pytest

from bioamla.audio import (
    AudioAnalysis,
    AudioData,
    AudioInfo,
    AudioPlayer,
    analyze_audio,
    bandpass_filter,
    detect_silence,
    get_amplitude_stats,
    get_audio_info,
    get_frequency_stats,
    highpass_filter,
    is_silent,
    list_audio_files,
    load_audio,
    load_audio_data,
    lowpass_filter,
    normalize_loudness,
    peak_normalize,
    resample_audio,
    save_audio,
    save_audio_data,
    segment_on_silence,
    spectral_denoise,
    split_audio_on_silence,
    trim_audio,
    trim_silence,
)
from bioamla.exceptions import (
    AudioLoadError,
    DependencyError,
    NotFoundError,
)


class TestAmplitudeAnalysis:
    def test_amplitude_stats(self, sample_audio_data: AudioData) -> None:
        stats = get_amplitude_stats(sample_audio_data.samples)
        assert stats.rms > 0
        assert stats.peak > 0
        assert stats.peak >= stats.rms

    def test_is_silent_false_for_signal(self, sample_audio_data: AudioData) -> None:
        assert is_silent(sample_audio_data.samples) is False

    def test_is_silent_true_for_zeros(self) -> None:
        silent = np.zeros(16000, dtype=np.float32)
        assert is_silent(silent) is True


class TestFrequencyAnalysis:
    def test_frequency_stats_peak(self, sample_audio_data: AudioData) -> None:
        stats = get_frequency_stats(
            sample_audio_data.samples, sample_audio_data.sample_rate
        )
        # 440 Hz sine -> peak frequency should be near 440 Hz
        assert 300 < stats.peak_frequency < 600

    def test_detect_silence_signal(self, sample_audio_with_silence: AudioData) -> None:
        info = detect_silence(
            sample_audio_with_silence.samples, sample_audio_with_silence.sample_rate
        )
        assert info.silence_ratio > 0
        assert len(info.sound_segments) >= 1


class TestFilters:
    def test_lowpass(self, sample_audio_data: AudioData) -> None:
        out = lowpass_filter(sample_audio_data.samples, sample_audio_data.sample_rate, 1000)
        assert out.shape == sample_audio_data.samples.shape
        assert out.dtype == np.float32

    def test_highpass(self, sample_audio_data: AudioData) -> None:
        out = highpass_filter(sample_audio_data.samples, sample_audio_data.sample_rate, 1000)
        assert out.shape == sample_audio_data.samples.shape

    def test_bandpass(self, sample_audio_data: AudioData) -> None:
        out = bandpass_filter(
            sample_audio_data.samples, sample_audio_data.sample_rate, 200, 2000
        )
        assert out.shape == sample_audio_data.samples.shape

    def test_bandpass_invalid_range_raises(self, sample_audio_data: AudioData) -> None:
        with pytest.raises(ValueError):
            bandpass_filter(
                sample_audio_data.samples, sample_audio_data.sample_rate, 2000, 200
            )


class TestNormalize:
    def test_peak_normalize(self, sample_audio_data: AudioData) -> None:
        out = peak_normalize(sample_audio_data.samples, target_peak=0.9)
        assert np.max(np.abs(out)) == pytest.approx(0.9, abs=1e-4)

    def test_normalize_loudness(self, sample_audio_data: AudioData) -> None:
        out = normalize_loudness(
            sample_audio_data.samples, sample_audio_data.sample_rate, target_db=-10
        )
        assert out.shape == sample_audio_data.samples.shape
        assert np.max(np.abs(out)) <= 1.0


class TestResample:
    def test_resample_changes_length(self, sample_audio_data: AudioData) -> None:
        out = resample_audio(sample_audio_data.samples, 16000, 8000)
        assert len(out) == pytest.approx(len(sample_audio_data.samples) // 2, rel=0.05)

    def test_resample_noop(self, sample_audio_data: AudioData) -> None:
        out = resample_audio(sample_audio_data.samples, 16000, 16000)
        assert out is sample_audio_data.samples


class TestTrim:
    def test_trim_audio_range(self, sample_audio_3s: AudioData) -> None:
        out = trim_audio(
            sample_audio_3s.samples, sample_audio_3s.sample_rate, start_time=0.5, end_time=1.5
        )
        assert len(out) == pytest.approx(sample_audio_3s.sample_rate, rel=0.01)

    def test_trim_audio_invalid_raises(self, sample_audio_3s: AudioData) -> None:
        with pytest.raises(ValueError):
            trim_audio(
                sample_audio_3s.samples,
                sample_audio_3s.sample_rate,
                start_time=2.0,
                end_time=1.0,
            )

    def test_trim_silence(self, sample_audio_with_silence: AudioData) -> None:
        out = trim_silence(
            sample_audio_with_silence.samples, sample_audio_with_silence.sample_rate
        )
        assert len(out) < len(sample_audio_with_silence.samples)


class TestDenoise:
    def test_denoise_runs(self, sample_audio_with_noise: AudioData) -> None:
        out = spectral_denoise(
            sample_audio_with_noise.samples, sample_audio_with_noise.sample_rate
        )
        assert out.dtype == np.float32
        assert len(out) > 0


class TestSegment:
    def test_segment_on_silence(self, sample_audio_3s: AudioData) -> None:
        segments = segment_on_silence(
            sample_audio_3s.samples,
            sample_audio_3s.sample_rate,
            min_segment_duration=0.1,
        )
        assert isinstance(segments, list)

    def test_split_audio_on_silence(self, sample_audio_3s: AudioData) -> None:
        chunks = split_audio_on_silence(
            sample_audio_3s.samples,
            sample_audio_3s.sample_rate,
            min_segment_duration=0.1,
        )
        for chunk, start, end in chunks:
            assert isinstance(chunk, np.ndarray)
            assert end >= start


class TestFileIO:
    def test_load_audio_tuple(self, test_audio_path: str) -> None:
        audio, sr = load_audio(test_audio_path)
        assert isinstance(audio, np.ndarray)
        assert sr == 16000
        assert audio.ndim == 1

    def test_load_audio_data(self, test_audio_path: str) -> None:
        data = load_audio_data(test_audio_path)
        assert isinstance(data, AudioData)
        assert data.sample_rate == 16000
        assert data.duration == pytest.approx(1.0, rel=0.05)

    def test_load_missing_raises_notfound(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            load_audio_data(str(tmp_path / "does_not_exist.wav"))

    def test_save_audio_roundtrip(self, sample_audio_data: AudioData, tmp_path) -> None:
        out = tmp_path / "out.wav"
        save_audio(str(out), sample_audio_data.samples, sample_audio_data.sample_rate)
        assert out.exists()
        audio, sr = load_audio(str(out))
        assert sr == sample_audio_data.sample_rate

    def test_save_audio_data_roundtrip(
        self, sample_audio_data: AudioData, tmp_path
    ) -> None:
        out = tmp_path / "out.wav"
        save_audio_data(sample_audio_data, str(out))
        assert out.exists()


class TestInfo:
    def test_get_audio_info(self, test_audio_path: str) -> None:
        info = get_audio_info(test_audio_path)
        assert isinstance(info, AudioInfo)
        assert info.sample_rate == 16000
        assert info.duration == pytest.approx(1.0, rel=0.1)

    def test_get_audio_info_missing_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            get_audio_info(str(tmp_path / "missing.wav"))

    def test_analyze_audio(self, test_audio_path: str) -> None:
        analysis = analyze_audio(test_audio_path)
        assert isinstance(analysis, AudioAnalysis)
        d = analysis.to_dict()
        for key in ("file_path", "info", "amplitude", "frequency", "silence"):
            assert key in d

    def test_analyze_audio_missing_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            analyze_audio(str(tmp_path / "missing.wav"))


class TestDiscovery:
    def test_list_audio_files(self, test_audio_dir: str) -> None:
        files = list_audio_files(test_audio_dir)
        assert len(files) == 3


class TestPlayback:
    def test_player_load_no_device(self, sample_audio_data: AudioData) -> None:
        player = AudioPlayer()
        player.load(sample_audio_data.samples, sample_audio_data.sample_rate)
        assert player.duration == pytest.approx(1.0, rel=0.05)
        assert player.is_stopped

    def test_play_without_load_raises(self) -> None:
        player = AudioPlayer()
        with pytest.raises((RuntimeError, DependencyError)):
            player.play()


class TestLoadWaveformTensor:
    def test_load_waveform_tensor(self, test_audio_path: str) -> None:
        try:
            import torchaudio  # noqa: F401
        except ImportError:
            with pytest.raises(DependencyError):
                from bioamla.audio import load_waveform_tensor

                load_waveform_tensor(test_audio_path)
            return

        from bioamla.audio import load_waveform_tensor

        waveform, sr = load_waveform_tensor(test_audio_path)
        assert sr == 16000

    def test_bad_audio_raises_load_error(self, tmp_path) -> None:
        # A non-audio file should raise AudioLoadError when decoded.
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not audio data")
        with pytest.raises(AudioLoadError):
            load_audio(str(bad))

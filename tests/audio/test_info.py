"""Coverage tests for bioamla.audio.info."""

from unittest.mock import patch

import numpy as np
import pytest

from bioamla.audio.info import (
    AudioAnalysis,
    AudioInfo,
    analyze_audio,
    analyze_audio_batch,
    get_audio_info,
    get_channels,
    get_duration,
    get_sample_rate,
    summarize_analysis,
)
from bioamla.exceptions import AudioLoadError, NotFoundError


class TestGetAudioInfo:
    def test_returns_info(self, test_audio_path: str) -> None:
        info = get_audio_info(test_audio_path)
        assert isinstance(info, AudioInfo)
        assert info.sample_rate == 16000
        assert info.channels == 1
        assert info.duration > 0

    def test_to_dict(self, test_audio_path: str) -> None:
        d = get_audio_info(test_audio_path).to_dict()
        assert d["sample_rate"] == 16000
        assert "subtype" in d

    def test_missing_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            get_audio_info(str(tmp_path / "missing.wav"))

    def test_decode_failure_raises(self, test_audio_path: str) -> None:
        with patch("bioamla.audio._pydub.get_audio_info", side_effect=RuntimeError("boom")):
            with pytest.raises(AudioLoadError):
                get_audio_info(test_audio_path)


class TestConvenienceGetters:
    def test_get_duration(self, test_audio_path: str) -> None:
        assert get_duration(test_audio_path) > 0

    def test_get_sample_rate(self, test_audio_path: str) -> None:
        assert get_sample_rate(test_audio_path) == 16000

    def test_get_channels(self, test_audio_path: str) -> None:
        assert get_channels(test_audio_path) == 1


class TestAnalyzeAudio:
    def test_full_analysis(self, test_audio_path: str) -> None:
        analysis = analyze_audio(test_audio_path)
        assert isinstance(analysis, AudioAnalysis)
        assert analysis.amplitude.rms > 0
        d = analysis.to_dict()
        assert "amplitude" in d and "frequency" in d and "silence" in d

    def test_missing_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            analyze_audio(str(tmp_path / "missing.wav"))

    def test_load_failure_raises(self, test_audio_path: str) -> None:
        with patch("bioamla.audio._pydub.load_audio", side_effect=RuntimeError("boom")):
            with pytest.raises(AudioLoadError):
                analyze_audio(test_audio_path)


class TestAnalyzeBatch:
    def test_batch_ok(self, test_audio_path: str, test_audio_path_3s: str) -> None:
        results = analyze_audio_batch([test_audio_path, test_audio_path_3s], verbose=True)
        assert len(results) == 2

    def test_batch_skips_failures(self, test_audio_path: str, tmp_path) -> None:
        results = analyze_audio_batch(
            [test_audio_path, str(tmp_path / "missing.wav")], verbose=True
        )
        assert len(results) == 1


class TestSummarizeAnalysis:
    def test_empty(self) -> None:
        summary = summarize_analysis([])
        assert summary["total_files"] == 0
        assert summary["total_duration"] == 0.0

    def test_with_results(self, test_audio_path: str, test_audio_path_3s: str) -> None:
        results = analyze_audio_batch([test_audio_path, test_audio_path_3s], verbose=False)
        summary = summarize_analysis(results)
        assert summary["total_files"] == 2
        assert summary["total_duration"] > 0
        assert "avg_peak_frequency" in summary
        assert "silent_file_ratio" in summary

    def test_infinite_db_handled(self, test_audio_path: str) -> None:
        results = analyze_audio_batch([test_audio_path], verbose=False)
        # force rms_db to -inf to exercise the filter branch
        results[0].amplitude.rms_db = -np.inf
        results[0].amplitude.peak_db = -np.inf
        summary = summarize_analysis(results)
        assert summary["avg_rms_db"] == -np.inf

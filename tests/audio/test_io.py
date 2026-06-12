"""Coverage tests for bioamla.audio.io."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from bioamla.audio import AudioData
from bioamla.audio.io import (
    batch_process,
    create_temp_audio_file,
    load_audio,
    load_audio_data,
    load_waveform_tensor,
    process_file,
    save_audio,
    save_audio_data,
    save_audio_data_as,
)
from bioamla.exceptions import AudioLoadError, AudioSaveError, NotFoundError


class TestLoadAudioData:
    def test_round_trip(self, test_audio_path: str) -> None:
        data = load_audio_data(test_audio_path)
        assert isinstance(data, AudioData)
        assert data.sample_rate == 16000
        assert data.channels == 1
        assert data.num_samples > 0
        assert data.is_modified is False
        assert "original_duration" in data.metadata

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            load_audio_data(str(tmp_path / "missing.wav"))

    def test_with_resample(self, test_audio_path: str) -> None:
        data = load_audio_data(test_audio_path, sample_rate=8000)
        assert data.sample_rate == 8000
        assert data.is_modified is True

    def test_load_failure_raises_audio_load_error(self, test_audio_path: str) -> None:
        with patch("bioamla.audio._pydub.load_audio", side_effect=RuntimeError("boom")):
            with pytest.raises(AudioLoadError):
                load_audio_data(test_audio_path)

    def test_resample_failure_raises(self, test_audio_path: str) -> None:
        with patch("bioamla.audio.processing.resample_audio", side_effect=RuntimeError("bad")):
            with pytest.raises(AudioLoadError):
                load_audio_data(test_audio_path, sample_rate=8000)

    def test_stereo_kept(self, test_audio_path: str) -> None:
        fake = np.zeros((100, 2), dtype=np.float32)
        with patch("bioamla.audio._pydub.load_audio", return_value=(fake, 16000)):
            data = load_audio_data(test_audio_path)
        assert data.channels == 2


class TestSaveAudioData:
    def test_round_trip(self, sample_audio_data: AudioData, tmp_path) -> None:
        out = tmp_path / "out.wav"
        result = save_audio_data(sample_audio_data, str(out))
        assert isinstance(result, Path)
        assert out.exists()

    def test_failure_raises(self, sample_audio_data: AudioData, tmp_path) -> None:
        with patch("bioamla.audio._pydub.save_audio", side_effect=RuntimeError("fail")):
            with pytest.raises(AudioSaveError):
                save_audio_data(sample_audio_data, str(tmp_path / "x.wav"))


class TestSaveAudioDataAs:
    def test_no_resample(self, sample_audio_data: AudioData, tmp_path) -> None:
        out = tmp_path / "out.wav"
        save_audio_data_as(sample_audio_data, str(out))
        assert out.exists()

    def test_with_resample(self, sample_audio_data: AudioData, tmp_path) -> None:
        out = tmp_path / "out.wav"
        save_audio_data_as(sample_audio_data, str(out), target_sample_rate=8000)
        assert out.exists()

    def test_resample_failure_raises(self, sample_audio_data: AudioData, tmp_path) -> None:
        with patch("bioamla.audio.processing.resample_audio", side_effect=RuntimeError("bad")):
            with pytest.raises(AudioSaveError):
                save_audio_data_as(
                    sample_audio_data, str(tmp_path / "o.wav"), target_sample_rate=8000
                )


class TestCreateTempAudioFile:
    def test_creates_file(self, sample_audio_data: AudioData) -> None:
        path = create_temp_audio_file(sample_audio_data)
        try:
            assert path.exists()
        finally:
            path.unlink(missing_ok=True)

    def test_failure_cleans_up_and_raises(self, sample_audio_data: AudioData) -> None:
        with patch("bioamla.audio._pydub.save_audio", side_effect=RuntimeError("fail")):
            with pytest.raises(AudioSaveError):
                create_temp_audio_file(sample_audio_data)


class TestArrayLevelIO:
    def test_load_save_round_trip(self, test_audio_path: str, tmp_path) -> None:
        audio, sr = load_audio(test_audio_path)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        out = save_audio(str(tmp_path / "rt.wav"), audio, sr)
        assert Path(out).exists()

    def test_load_failure_raises(self, test_audio_path: str) -> None:
        with patch("bioamla.audio._pydub.load_audio", side_effect=RuntimeError("boom")):
            with pytest.raises(AudioLoadError):
                load_audio(test_audio_path)

    def test_load_stereo_downmixes(self, test_audio_path: str) -> None:
        stereo = np.zeros((2, 100), dtype=np.float32)
        with patch("bioamla.audio._pydub.load_audio", return_value=(stereo, 16000)):
            audio, sr = load_audio(test_audio_path)
        assert audio.ndim == 1

    def test_save_failure_raises(self, tmp_path) -> None:
        with patch("bioamla.audio._pydub.save_audio", side_effect=RuntimeError("fail")):
            with pytest.raises(AudioSaveError):
                save_audio(str(tmp_path / "x.wav"), np.zeros(10, dtype=np.float32), 16000)


@pytest.mark.usefixtures("requires_torchcodec")
class TestLoadWaveformTensor:
    def test_delegates(self, test_audio_path: str) -> None:
        pytest.importorskip("torch")
        wave, sr = load_waveform_tensor(test_audio_path)
        assert sr == 16000


class TestProcessFile:
    def test_identity(self, test_audio_path: str, tmp_path) -> None:
        out = tmp_path / "proc.wav"
        result = process_file(test_audio_path, str(out), lambda a, sr: a)
        assert Path(result).exists()

    def test_with_resample(self, test_audio_path: str, tmp_path) -> None:
        out = tmp_path / "proc.wav"
        process_file(test_audio_path, str(out), lambda a, sr: a, sample_rate=8000)
        assert out.exists()


class TestBatchProcess:
    def test_processes_directory(self, test_audio_dir: str, tmp_path) -> None:
        out_dir = tmp_path / "out"
        stats = batch_process(test_audio_dir, str(out_dir), lambda a, sr: a, verbose=False)
        assert stats["files_processed"] == 3
        assert stats["files_failed"] == 0

    def test_missing_input_dir_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            batch_process(str(tmp_path / "nope"), str(tmp_path / "out"), lambda a, sr: a)

    def test_empty_dir_returns_zero(self, tmp_path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        stats = batch_process(str(empty), str(tmp_path / "out"), lambda a, sr: a, verbose=True)
        assert stats["files_processed"] == 0

    def test_per_file_error_counted(self, test_audio_dir: str, tmp_path) -> None:
        def boom(_a, _sr):
            raise RuntimeError("nope")

        stats = batch_process(test_audio_dir, str(tmp_path / "out"), boom, verbose=True)
        assert stats["files_failed"] == 3
        assert stats["files_processed"] == 0

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
        with patch("bioamla.audio._codec.load_audio", side_effect=RuntimeError("boom")):
            with pytest.raises(AudioLoadError):
                load_audio_data(test_audio_path)

    def test_resample_failure_raises(self, test_audio_path: str) -> None:
        with patch("bioamla.audio.processing.resample_audio", side_effect=RuntimeError("bad")):
            with pytest.raises(AudioLoadError):
                load_audio_data(test_audio_path, sample_rate=8000)

    def test_stereo_kept(self, test_audio_path: str) -> None:
        fake = np.zeros((100, 2), dtype=np.float32)
        with patch("bioamla.audio._codec.load_audio", return_value=(fake, 16000)):
            data = load_audio_data(test_audio_path)
        assert data.channels == 2


class TestSaveAudioData:
    def test_round_trip(self, sample_audio_data: AudioData, tmp_path) -> None:
        out = tmp_path / "out.wav"
        result = save_audio_data(sample_audio_data, str(out))
        assert isinstance(result, Path)
        assert out.exists()

    def test_failure_raises(self, sample_audio_data: AudioData, tmp_path) -> None:
        with patch("bioamla.audio._codec.save_audio", side_effect=RuntimeError("fail")):
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
        with patch("bioamla.audio._codec.save_audio", side_effect=RuntimeError("fail")):
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
        with patch("bioamla.audio._codec.load_audio", side_effect=RuntimeError("boom")):
            with pytest.raises(AudioLoadError):
                load_audio(test_audio_path)

    def test_load_stereo_downmixes(self, test_audio_path: str) -> None:
        stereo = np.column_stack([np.ones(100), np.zeros(100)]).astype(np.float32)
        with patch("bioamla.audio._codec.load_audio_channels", return_value=(stereo, 16000)):
            audio, sr = load_audio(test_audio_path)
        assert audio.shape == (100,) and np.allclose(audio, 0.5)  # channels averaged

    def test_save_failure_raises(self, tmp_path) -> None:
        with patch("bioamla.audio._codec.save_audio", side_effect=RuntimeError("fail")):
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

    def test_parallel_with_progress(self, test_audio_dir: str, tmp_path) -> None:
        calls: list[tuple[int, int]] = []
        out_dir = tmp_path / "out"
        stats = batch_process(
            test_audio_dir,
            str(out_dir),
            lambda a, sr: a,
            verbose=False,
            max_workers=3,
            on_progress=lambda done, total: calls.append((done, total)),
        )
        assert stats["files_processed"] == 3
        assert sorted(p.name for p in out_dir.glob("*.wav")) == [
            "audio_0.wav",
            "audio_1.wav",
            "audio_2.wav",
        ]
        assert calls[-1] == (3, 3)

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

    def test_reports_through_logging_not_stdout(
        self, test_audio_dir: str, tmp_path, caplog, capsys
    ) -> None:
        def boom(_a, _sr):
            raise RuntimeError("nope")

        with caplog.at_level("INFO", logger="bioamla"):
            batch_process(test_audio_dir, str(tmp_path / "out"), boom, verbose=True)

        assert capsys.readouterr().out == ""
        levels = {r.levelname for r in caplog.records}
        assert levels == {"INFO", "WARNING"}
        assert sum(r.levelname == "WARNING" for r in caplog.records) == 3


def test_wav_bytes_round_trip_exactly():
    """audio_to_wav_bytes / audio_from_wav_bytes carry float32 samples unchanged, mono or
    stereo, with the metadata the caller attaches; bytes that aren't WAV are an
    AudioLoadError."""
    import numpy as np
    import pytest

    from bioamla.audio import AudioData, audio_from_wav_bytes, audio_to_wav_bytes
    from bioamla.exceptions import AudioLoadError

    mono = AudioData(samples=np.linspace(-1.5, 1.5, 1000, dtype="float32"), sample_rate=16000)
    back = audio_from_wav_bytes(
        audio_to_wav_bytes(mono), source_path="a.wav", metadata={"start_time": 2.5}
    )
    assert back.sample_rate == 16000 and back.channels == 1 and back.source_path == "a.wav"
    assert back.metadata == {"start_time": 2.5}
    assert np.array_equal(back.samples, mono.samples)  # no clipping, no quantization

    stereo = AudioData(samples=np.zeros((500, 2), dtype="float32"), sample_rate=8000, channels=2)
    assert audio_from_wav_bytes(audio_to_wav_bytes(stereo)).channels == 2
    with pytest.raises(AudioLoadError, match="Not WAV audio"):
        audio_from_wav_bytes(b"<html>not audio</html>")


def test_save_audio_window_writes_whole_files_or_nothing(tmp_path):
    """save_audio_window keeps the float32 encoding, creates the folder, refuses to replace a
    file unless asked, replaces it in one step when asked, and leaves nothing else behind."""
    import numpy as np
    import pytest
    import soundfile as sf

    from bioamla.audio import AudioData, save_audio_window
    from bioamla.exceptions import AudioSaveError

    audio = AudioData(samples=np.linspace(-1, 1, 800, dtype="float32"), sample_rate=8000)
    out = save_audio_window(audio, tmp_path / "clips" / "a.wav")
    assert out == tmp_path / "clips" / "a.wav" and sf.info(out).subtype == "FLOAT"
    assert np.array_equal(sf.read(out, dtype="float32")[0], audio.samples)
    with pytest.raises(AudioSaveError, match="a.wav exists"):
        save_audio_window(audio, out)

    quieter = AudioData(samples=audio.samples * 0.5, sample_rate=8000)
    assert save_audio_window(quieter, out, overwrite=True) == out
    assert np.array_equal(sf.read(out, dtype="float32")[0], quieter.samples)
    assert [p.name for p in out.parent.iterdir()] == ["a.wav"]  # no temporary file left

    # A window that can't be encoded changes nothing: the old file stays, no new one appears.
    broken = AudioData(samples=audio.samples, sample_rate=0)
    with pytest.raises(AudioSaveError):
        save_audio_window(broken, out, overwrite=True)
    assert np.array_equal(sf.read(out, dtype="float32")[0], quieter.samples)
    with pytest.raises(AudioSaveError):
        save_audio_window(broken, tmp_path / "clips" / "b.wav")
    assert [p.name for p in out.parent.iterdir()] == ["a.wav"]

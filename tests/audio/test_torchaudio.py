"""Coverage tests for bioamla.audio.torchaudio (torchaudio mocked / real tensors)."""

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from bioamla.audio.torchaudio import (  # noqa: E402
    get_wav_files,
    get_wav_info,
    get_wavefile_sample_rate,
    get_wavefile_shape,
    load_audio_from_bytes,
    load_waveform_tensor,
    resample_waveform_tensor,
    split_waveform_tensor,
)
from bioamla.exceptions import AudioLoadError  # noqa: E402


class TestSplitWaveformTensor:
    def test_splits_into_segments(self) -> None:
        wave = torch.zeros((1, 16000 * 3))
        segments = split_waveform_tensor(wave, freq=16000, segment_duration=1, segment_overlap=0)
        assert len(segments) == 3
        seg, start, end = segments[0]
        assert seg.shape == (1, 16000)
        assert start == 0
        assert end == 16000

    def test_with_overlap(self) -> None:
        wave = torch.zeros((1, 16000 * 2))
        segments = split_waveform_tensor(wave, freq=16000, segment_duration=1, segment_overlap=0)
        assert len(segments) == 2

    def test_too_short_returns_empty(self) -> None:
        wave = torch.zeros((1, 100))
        assert split_waveform_tensor(wave, 16000, 1, 0) == []


class TestGetWavFiles:
    def test_lists_wav(self, test_audio_dir: str) -> None:
        files = get_wav_files(test_audio_dir)
        assert len(files) == 3


class TestMockedTorchaudio:
    def test_get_wav_info(self) -> None:
        fake_ta = MagicMock()
        fake_ta.info.return_value = "INFO"
        with patch("bioamla.audio.torchaudio._import_torchaudio", return_value=fake_ta):
            assert get_wav_info("/x.wav") == "INFO"

    def test_get_wavefile_shape(self) -> None:
        fake_ta = MagicMock()
        fake_ta.load.return_value = (torch.zeros((1, 1000)), 16000)
        with patch("bioamla.audio.torchaudio._import_torchaudio", return_value=fake_ta):
            assert get_wavefile_shape("/x.wav") == torch.Size([1, 1000])

    def test_get_wavefile_sample_rate(self) -> None:
        fake_ta = MagicMock()
        fake_ta.load.return_value = (torch.zeros((1, 1000)), 22050)
        with patch("bioamla.audio.torchaudio._import_torchaudio", return_value=fake_ta):
            assert get_wavefile_sample_rate("/x.wav") == 22050

    def test_load_waveform_tensor_ok(self) -> None:
        fake_ta = MagicMock()
        wave = torch.zeros((1, 500))
        fake_ta.load.return_value = (wave, 16000)
        with patch("bioamla.audio.torchaudio._import_torchaudio", return_value=fake_ta):
            out, sr = load_waveform_tensor("/x.wav")
        assert sr == 16000
        assert out.shape == (1, 500)

    def test_load_waveform_tensor_error(self) -> None:
        fake_ta = MagicMock()
        fake_ta.load.side_effect = RuntimeError("bad")
        with patch("bioamla.audio.torchaudio._import_torchaudio", return_value=fake_ta):
            with pytest.raises(AudioLoadError):
                load_waveform_tensor("/x.wav")

    def test_resample_waveform_tensor(self) -> None:
        wave = torch.zeros((1, 16000))
        out = resample_waveform_tensor(wave, 16000, 8000)
        assert out.shape[1] == pytest.approx(8000, abs=10)

    def test_load_audio_from_bytes_mono(self) -> None:
        fake_ta = MagicMock()
        fake_ta.load.return_value = (torch.zeros((1, 16000)), 16000)
        with patch("bioamla.audio.torchaudio._import_torchaudio", return_value=fake_ta):
            arr, sr = load_audio_from_bytes(b"data", target_sr=16000)
        assert sr == 16000
        assert arr.shape == (16000,)

    def test_load_audio_from_bytes_stereo_resample(self) -> None:
        fake_ta = MagicMock()
        fake_ta.load.return_value = (torch.zeros((2, 44100)), 44100)
        # use the real Resample transform by importing torchaudio for transforms
        import torchaudio

        fake_ta.transforms = torchaudio.transforms
        with patch("bioamla.audio.torchaudio._import_torchaudio", return_value=fake_ta):
            arr, sr = load_audio_from_bytes(b"data", target_sr=16000)
        assert sr == 16000
        assert arr.ndim == 1

    def test_load_audio_from_bytes_error(self) -> None:
        fake_ta = MagicMock()
        fake_ta.load.side_effect = RuntimeError("decode fail")
        with patch("bioamla.audio.torchaudio._import_torchaudio", return_value=fake_ta):
            with pytest.raises(ValueError):
                load_audio_from_bytes(b"bad")

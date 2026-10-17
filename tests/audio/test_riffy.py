"""Tests for the riffy WAV header backend in bioamla.audio._riffy."""

import wave

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from bioamla.audio._riffy import get_wav_info, is_wav_path
from bioamla.exceptions import AudioLoadError, InvalidInputError, NotFoundError


def _write_pcm_wav(path, sample_rate=22050, channels=2, bits=16, seconds=0.5) -> None:
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bits // 8)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00" * frames * channels * (bits // 8))


class TestIsWavPath:
    @pytest.mark.parametrize("name", ["a.wav", "A.WAV", "b.wave", "/x/y/z.Wav"])
    def test_wav_suffixes(self, name: str) -> None:
        assert is_wav_path(name)

    @pytest.mark.parametrize("name", ["a.flac", "a.mp3", "a", "a.wav.txt"])
    def test_other_suffixes(self, name: str) -> None:
        assert not is_wav_path(name)


class TestGetWavInfo:
    def test_pcm16_mono(self, test_audio_path: str) -> None:
        info = get_wav_info(test_audio_path)
        assert info["sample_rate"] == 16000
        assert info["channels"] == 1
        assert info["samples"] == 16000
        assert info["duration"] == pytest.approx(1.0)
        assert info["bit_depth"] == 16
        assert info["subtype"] == "PCM_16"
        assert info["format"] == "WAV"
        assert info["codec"] == "pcm"

    def test_pcm_stereo_24bit(self, tmp_path) -> None:
        path = tmp_path / "stereo.wav"
        _write_pcm_wav(path, sample_rate=48000, channels=2, bits=24, seconds=0.25)
        info = get_wav_info(path)
        assert info["sample_rate"] == 48000
        assert info["channels"] == 2
        assert info["bit_depth"] == 24
        assert info["subtype"] == "PCM_24"
        assert info["samples"] == 12000
        assert info["duration"] == pytest.approx(0.25)

    def test_pcm_u8(self, tmp_path) -> None:
        path = tmp_path / "u8.wav"
        _write_pcm_wav(path, sample_rate=8000, channels=1, bits=8, seconds=0.1)
        info = get_wav_info(path)
        assert info["bit_depth"] == 8
        assert info["subtype"] == "PCM_U8"

    def test_matches_pydub_shape(self, test_audio_path: str) -> None:
        info = get_wav_info(test_audio_path)
        assert set(info) == {
            "duration",
            "sample_rate",
            "channels",
            "samples",
            "format",
            "codec",
            "bit_depth",
            "subtype",
        }

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            get_wav_info(tmp_path / "missing.wav")

    def test_non_wav_suffix_raises(self, tmp_path) -> None:
        path = tmp_path / "clip.flac"
        path.write_bytes(b"fLaC")
        with pytest.raises(InvalidInputError):
            get_wav_info(path)

    def test_garbage_raises_audio_load_error(self, tmp_path) -> None:
        path = tmp_path / "junk.wav"
        path.write_bytes(b"not a riff file at all")
        with pytest.raises(AudioLoadError):
            get_wav_info(path)

    def test_truncated_header_raises_audio_load_error(self, tmp_path) -> None:
        path = tmp_path / "short.wav"
        path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        with pytest.raises(AudioLoadError):
            get_wav_info(path)

    def test_float_wav_raises_audio_load_error(self, tmp_path) -> None:
        # scipy writes float32 as WAVE_FORMAT_IEEE_FLOAT (3); riffy is PCM-only.
        path = tmp_path / "float.wav"
        wavfile.write(str(path), 16000, np.zeros(1600, dtype=np.float32))
        with pytest.raises(AudioLoadError):
            get_wav_info(path)

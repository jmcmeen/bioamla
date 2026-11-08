"""Coverage tests for bioamla.audio._pydub (pydub/ffmpeg mocked)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bioamla.audio import _pydub
from bioamla.exceptions import AudioLoadError, AudioSaveError, NotFoundError


class TestLoadAudio:
    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            _pydub.load_audio(str(tmp_path / "missing.wav"))

    def test_wav_via_soundfile(self, test_audio_path: str) -> None:
        audio, sr = _pydub.load_audio(test_audio_path)
        assert sr == 16000
        assert audio.ndim == 1

    def test_soundfile_failure_falls_back_to_ffmpeg(self, test_audio_path: str) -> None:
        decoded = (np.array([[0.0], [0.5]], dtype=np.float32), 16000)
        with (
            patch("soundfile.read", side_effect=RuntimeError("sf bad")),
            patch.object(_pydub, "_ffmpeg_decode", return_value=decoded) as decode,
        ):
            audio, sr = _pydub.load_audio(test_audio_path)
        assert sr == 16000 and audio.tolist() == [0.0, 0.5]
        assert decode.call_args.args[1:] == (0.0, None)  # the whole file

    def test_non_native_format_uses_ffmpeg(self, tmp_path) -> None:
        path = tmp_path / "x.mp3"
        path.write_bytes(b"fake")
        stereo = np.array([[0.2, 0.4], [0.0, 1.0]], dtype=np.float32)
        with patch.object(_pydub, "_ffmpeg_decode", return_value=(stereo, 16000)):
            audio, sr = _pydub.load_audio(str(path))
            kept, _ = _pydub.load_audio_channels(str(path))
        assert sr == 16000 and np.allclose(audio, [0.3, 0.5])  # downmixed
        assert kept.shape == (2, 2)  # load_audio_channels keeps them

    @pytest.mark.usefixtures("requires_ffmpeg_cli")
    def test_undecodable_file_is_an_audio_load_error(self, tmp_path) -> None:
        path = tmp_path / "x.mp3"
        path.write_bytes(b"fake")
        with pytest.raises(AudioLoadError, match="Error opening"):
            _pydub.load_audio(str(path))

    def test_soundfile_stereo_downmix(self, test_audio_path: str) -> None:
        stereo = np.zeros((100, 2), dtype=np.float32)
        with patch("soundfile.read", return_value=(stereo, 16000)):
            audio, _ = _pydub.load_audio(test_audio_path)
        assert audio.ndim == 1


class TestSaveAudio:
    def test_save_closes_the_output_file(self, tmp_path) -> None:
        import gc
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            _pydub.save_audio(str(tmp_path / "out.wav"), np.zeros(160, dtype=np.float32), 16000)
            gc.collect()
        assert not [w for w in caught if issubclass(w.category, ResourceWarning)]

    def test_save_wav_and_flac_through_soundfile(self, tmp_path) -> None:
        import soundfile as sf

        for name in ("out.wav", "out.flac"):
            _pydub.save_audio(str(tmp_path / name), np.zeros(10, dtype=np.float32), 16000)
            assert sf.info(tmp_path / name).subtype == "PCM_16"

    def test_save_m4a_uses_ipod(self, tmp_path) -> None:
        with patch.object(_pydub, "_ffmpeg_encode") as encode:
            _pydub.save_audio(str(tmp_path / "out.m4a"), np.zeros(10, dtype=np.float32), 16000)
        assert encode.call_args.args[3] == "ipod"

    def test_save_explicit_format(self, tmp_path) -> None:
        import soundfile as sf

        for fmt in ("flac", "FLAC", ".flac"):  # any case, with or without a dot
            path = tmp_path / f"out-{fmt.strip('.')}.dat"
            _pydub.save_audio(str(path), np.zeros(10, dtype=np.float32), 16000, format=fmt,
                              bit_depth=24)  # fmt: skip
            assert sf.info(path).format == "FLAC" and sf.info(path).subtype == "PCM_24"

    def test_save_error_wrapped(self, tmp_path) -> None:
        with patch.object(_pydub, "_ffmpeg_encode", side_effect=RuntimeError("boom")):
            with pytest.raises(AudioSaveError, match="Failed to save"):
                _pydub.save_audio(str(tmp_path / "out.mp3"), np.zeros(10, dtype=np.float32), 16000)

    def test_save_unwritable_parent_wrapped(self, tmp_path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("not a directory")
        with pytest.raises(AudioSaveError, match="Failed to save"):
            _pydub.save_audio(str(blocker / "out.wav"), np.zeros(10, dtype=np.float32), 16000)


class TestGetMetadataFfprobe:
    def _ffprobe_json(self) -> str:
        import json

        return json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "audio",
                        "sample_rate": "16000",
                        "channels": "1",
                        "duration": "1.0",
                        "codec_name": "pcm_s16le",
                        "bits_per_sample": "16",
                    }
                ],
                "format": {"duration": "1.0"},
            }
        )

    def test_success(self) -> None:
        result = MagicMock(returncode=0, stdout=self._ffprobe_json())
        with patch("subprocess.run", return_value=result):
            meta = _pydub._get_metadata_ffprobe("/x.wav")
        assert meta["sample_rate"] == 16000
        assert meta["channels"] == 1
        assert meta["subtype"] == "PCM_16"
        assert meta["bit_depth"] == 16

    def test_nonzero_returncode(self) -> None:
        result = MagicMock(returncode=1, stdout="")
        with patch("subprocess.run", return_value=result):
            assert _pydub._get_metadata_ffprobe("/x.wav") is None

    def test_no_audio_stream(self) -> None:
        import json

        result = MagicMock(returncode=0, stdout=json.dumps({"streams": [{"codec_type": "video"}]}))
        with patch("subprocess.run", return_value=result):
            assert _pydub._get_metadata_ffprobe("/x.wav") is None

    def test_subprocess_error(self) -> None:
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.SubprocessError("no ffprobe")):
            assert _pydub._get_metadata_ffprobe("/x.wav") is None

    def test_unknown_codec(self) -> None:
        import json

        data = json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "audio",
                        "sample_rate": "44100",
                        "channels": "2",
                        "duration": "2.0",
                        "codec_name": "weirdcodec",
                    }
                ],
                "format": {"duration": "2.0"},
            }
        )
        result = MagicMock(returncode=0, stdout=data)
        with patch("subprocess.run", return_value=result):
            meta = _pydub._get_metadata_ffprobe("/x.wav")
        assert meta["subtype"] == "WEIRDCODEC"


class TestGetAudioInfo:
    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            _pydub.get_audio_info(str(tmp_path / "missing.wav"))

    def test_ffprobe_path(self, test_audio_path: str) -> None:
        meta = {"sample_rate": 16000, "channels": 1, "duration": 1.0}
        with patch.object(_pydub, "_get_metadata_ffprobe", return_value=meta):
            assert _pydub.get_audio_info(test_audio_path) is meta

    def test_soundfile_reads_the_header_when_ffprobe_cannot(self, test_audio_path: str) -> None:
        with patch.object(_pydub, "_get_metadata_ffprobe", return_value=None):
            meta = _pydub.get_audio_info(test_audio_path)
        assert meta["sample_rate"] == 16000 and meta["channels"] == 1
        assert meta["subtype"] == "PCM_16" and meta["bit_depth"] == 16
        assert meta["samples"] > 0 and meta["duration"] == meta["samples"] / 16000

    def test_no_header_reader_is_an_error(self, tmp_path, monkeypatch) -> None:
        import shutil

        from bioamla.exceptions import DependencyError

        path = tmp_path / "x.mp3"
        path.write_bytes(b"fake")
        with patch.object(_pydub, "_get_metadata_ffprobe", return_value=None):
            with pytest.raises(AudioLoadError, match="neither ffprobe nor soundfile"):
                _pydub.get_audio_info(str(path))
            monkeypatch.setattr(shutil, "which", lambda name: None)
            with pytest.raises(DependencyError, match="ffprobe is required to read .mp3"):
                _pydub.get_audio_info(str(path))

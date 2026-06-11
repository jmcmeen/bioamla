"""Coverage tests for bioamla.audio._pydub (pydub/ffmpeg mocked)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bioamla.audio import _pydub


def _make_segment(samples: np.ndarray, sample_width: int, channels: int, frame_rate: int):
    seg = MagicMock()
    seg.get_array_of_samples.return_value = samples
    seg.sample_width = sample_width
    seg.channels = channels
    seg.frame_rate = frame_rate
    return seg


class TestAudioSegmentToNumpy:
    def test_8bit(self) -> None:
        seg = _make_segment(np.array([128, 192, 64], dtype=np.uint8), 1, 1, 16000)
        audio, sr = _pydub._audiosegment_to_numpy(seg)
        assert sr == 16000
        assert audio.dtype == np.float32
        assert audio[0] == pytest.approx(0.0)

    def test_16bit(self) -> None:
        seg = _make_segment(np.array([0, 16384, -16384], dtype=np.int16), 2, 1, 16000)
        audio, _ = _pydub._audiosegment_to_numpy(seg)
        assert audio[1] == pytest.approx(0.5, abs=1e-3)

    def test_24bit(self) -> None:
        seg = _make_segment(np.array([0, 4194304], dtype=np.int32), 3, 1, 16000)
        audio, _ = _pydub._audiosegment_to_numpy(seg)
        assert audio[1] == pytest.approx(0.5, abs=1e-3)

    def test_32bit_int(self) -> None:
        seg = _make_segment(np.array([0, 1073741824], dtype=np.int32), 4, 1, 16000)
        audio, _ = _pydub._audiosegment_to_numpy(seg)
        assert audio[1] == pytest.approx(0.5, abs=1e-3)

    def test_32bit_float(self) -> None:
        seg = _make_segment(np.array([0.0, 0.5], dtype=np.float32), 4, 1, 16000)
        audio, _ = _pydub._audiosegment_to_numpy(seg)
        assert audio[1] == pytest.approx(0.5)

    def test_unsupported_width_raises(self) -> None:
        seg = _make_segment(np.array([1, 2], dtype=np.int16), 5, 1, 16000)
        with pytest.raises(ValueError):
            _pydub._audiosegment_to_numpy(seg)

    def test_stereo_downmix(self) -> None:
        # interleaved L/R: [0, 1, 0, 1] -> mono [0.5, 0.5] after scaling
        samples = np.array([0, 16384, 0, 16384], dtype=np.int16)
        seg = _make_segment(samples, 2, 2, 16000)
        audio, _ = _pydub._audiosegment_to_numpy(seg)
        assert len(audio) == 2


class TestNumpyToAudioSegment:
    def test_mono(self) -> None:
        with patch.object(_pydub, "AudioSegment") as seg_cls:
            _pydub._numpy_to_audiosegment(np.array([0.0, 0.5, -0.5], dtype=np.float32), 16000)
            seg_cls.assert_called_once()
            _, kwargs = seg_cls.call_args
            assert kwargs["channels"] == 1
            assert kwargs["frame_rate"] == 16000

    def test_2d_single_channel_flattened(self) -> None:
        with patch.object(_pydub, "AudioSegment") as seg_cls:
            _pydub._numpy_to_audiosegment(np.zeros((4, 1), dtype=np.float32), 16000, channels=1)
            seg_cls.assert_called_once()

    def test_clipping(self) -> None:
        with patch.object(_pydub, "AudioSegment") as seg_cls:
            _pydub._numpy_to_audiosegment(np.array([2.0, -2.0], dtype=np.float32), 16000)
            seg_cls.assert_called_once()


class TestLoadAudio:
    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            _pydub.load_audio(str(tmp_path / "missing.wav"))

    def test_wav_via_soundfile(self, test_audio_path: str) -> None:
        audio, sr = _pydub.load_audio(test_audio_path)
        assert sr == 16000
        assert audio.ndim == 1

    def test_soundfile_failure_falls_back_to_pydub(self, test_audio_path: str) -> None:
        seg = _make_segment(np.array([0, 16384], dtype=np.int16), 2, 1, 16000)
        with (
            patch("soundfile.read", side_effect=RuntimeError("sf bad")),
            patch.object(_pydub.AudioSegment, "from_file", return_value=seg),
        ):
            audio, sr = _pydub.load_audio(test_audio_path)
        assert sr == 16000

    def test_non_native_format_uses_pydub(self, tmp_path) -> None:
        path = tmp_path / "x.mp3"
        path.write_bytes(b"fake")
        seg = _make_segment(np.array([0, 16384], dtype=np.int16), 2, 1, 16000)
        with patch.object(_pydub.AudioSegment, "from_file", return_value=seg):
            audio, sr = _pydub.load_audio(str(path))
        assert sr == 16000

    def test_pydub_load_error_wrapped(self, tmp_path) -> None:
        path = tmp_path / "x.mp3"
        path.write_bytes(b"fake")
        with patch.object(_pydub.AudioSegment, "from_file", side_effect=RuntimeError("boom")):
            with pytest.raises(Exception, match="Error opening"):
                _pydub.load_audio(str(path))

    def test_soundfile_stereo_downmix(self, test_audio_path: str) -> None:
        stereo = np.zeros((100, 2), dtype=np.float32)
        with patch("soundfile.read", return_value=(stereo, 16000)):
            audio, _ = _pydub.load_audio(test_audio_path)
        assert audio.ndim == 1


class TestSaveAudio:
    def test_save_wav(self, tmp_path) -> None:
        seg = MagicMock()
        with patch.object(_pydub, "_numpy_to_audiosegment", return_value=seg):
            _pydub.save_audio(str(tmp_path / "out.wav"), np.zeros(10, dtype=np.float32), 16000)
            seg.export.assert_called_once()

    def test_save_m4a_uses_ipod(self, tmp_path) -> None:
        seg = MagicMock()
        with patch.object(_pydub, "_numpy_to_audiosegment", return_value=seg):
            _pydub.save_audio(str(tmp_path / "out.m4a"), np.zeros(10, dtype=np.float32), 16000)
            _, kwargs = seg.export.call_args
            assert kwargs["format"] == "ipod"

    def test_save_explicit_format(self, tmp_path) -> None:
        seg = MagicMock()
        with patch.object(_pydub, "_numpy_to_audiosegment", return_value=seg):
            _pydub.save_audio(
                str(tmp_path / "out.dat"), np.zeros(10, dtype=np.float32), 16000, format="flac"
            )
            _, kwargs = seg.export.call_args
            assert kwargs["format"] == "flac"

    def test_save_error_wrapped(self, tmp_path) -> None:
        with patch.object(_pydub, "_numpy_to_audiosegment", side_effect=RuntimeError("boom")):
            with pytest.raises(Exception, match="Failed to save"):
                _pydub.save_audio(str(tmp_path / "out.wav"), np.zeros(10, dtype=np.float32), 16000)


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
        with pytest.raises(FileNotFoundError):
            _pydub.get_audio_info(str(tmp_path / "missing.wav"))

    def test_ffprobe_path(self, test_audio_path: str) -> None:
        meta = {"sample_rate": 16000, "channels": 1, "duration": 1.0}
        with patch.object(_pydub, "_get_metadata_ffprobe", return_value=meta):
            assert _pydub.get_audio_info(test_audio_path) is meta

    def test_pydub_fallback(self, test_audio_path: str) -> None:
        seg = MagicMock()
        seg.__len__ = lambda self: 1000  # 1000 ms
        seg.frame_rate = 16000
        seg.channels = 1
        seg.sample_width = 2
        with (
            patch.object(_pydub, "_get_metadata_ffprobe", return_value=None),
            patch.object(_pydub.AudioSegment, "from_file", return_value=seg),
        ):
            meta = _pydub.get_audio_info(test_audio_path)
        assert meta["sample_rate"] == 16000
        assert meta["subtype"] == "PCM_16"
        assert meta["bit_depth"] == 16

    def test_pydub_fallback_error(self, test_audio_path: str) -> None:
        with (
            patch.object(_pydub, "_get_metadata_ffprobe", return_value=None),
            patch.object(_pydub.AudioSegment, "from_file", side_effect=RuntimeError("boom")),
        ):
            with pytest.raises(Exception, match="Failed to get audio info"):
                _pydub.get_audio_info(test_audio_path)

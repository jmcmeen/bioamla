"""Coverage tests for bioamla.viz.core."""

import matplotlib

matplotlib.use("Agg")

from unittest.mock import patch  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from bioamla.audio import AudioData  # noqa: E402
from bioamla.exceptions import (  # noqa: E402
    AudioLoadError,
    InvalidInputError,
    NotFoundError,
    ProcessingError,
)
from bioamla.viz.core import (  # noqa: E402
    _get_window_function,
    _load_audio_for_viz,
    compute_mel_spectrogram,
    compute_stft,
    generate_spectrogram,
    spectrogram_to_db,
    spectrogram_to_image,
)


class TestWindowFunction:
    @pytest.mark.parametrize(
        "window",
        ["hann", "hamming", "blackman", "bartlett", "rectangular", "kaiser"],
    )
    def test_known_windows(self, window: str) -> None:
        win = _get_window_function(window, 512)
        assert len(win) == 512

    def test_unknown_falls_back_to_hann(self) -> None:
        win = _get_window_function("unknown", 256)
        assert len(win) == 256


class TestLoadAudioForViz:
    def test_no_resample(self, test_audio_path: str) -> None:
        audio = _load_audio_for_viz(test_audio_path, 16000)
        assert audio.ndim == 1

    def test_resample(self, test_audio_path: str) -> None:
        audio = _load_audio_for_viz(test_audio_path, 8000)
        assert audio.ndim == 1

    def test_resample_failure_raises(self, test_audio_path: str) -> None:
        with patch("bioamla.audio.processing.resample_audio", side_effect=RuntimeError("bad")):
            with pytest.raises(AudioLoadError):
                _load_audio_for_viz(test_audio_path, 8000)


class TestGenerateSpectrogram:
    @pytest.mark.parametrize("viz_type", ["stft", "mel", "mfcc", "waveform"])
    def test_each_type(self, test_audio_path: str, tmp_path, viz_type: str) -> None:
        out = tmp_path / f"{viz_type}.png"
        result = generate_spectrogram(
            test_audio_path, str(out), viz_type=viz_type, backend="librosa"
        )
        assert out.exists()
        assert result == str(out)

    def test_jpeg_output(self, test_audio_path: str, tmp_path) -> None:
        out = tmp_path / "spec.jpg"
        generate_spectrogram(test_audio_path, str(out), backend="librosa")
        assert out.exists()

    def test_no_colorbar(self, test_audio_path: str, tmp_path) -> None:
        out = tmp_path / "spec.png"
        generate_spectrogram(test_audio_path, str(out), show_colorbar=False, backend="librosa")
        assert out.exists()

    def test_db_clipping(self, test_audio_path: str, tmp_path) -> None:
        out = tmp_path / "spec.png"
        generate_spectrogram(
            test_audio_path,
            str(out),
            viz_type="stft",
            db_min=-80,
            db_max=0,
            backend="librosa",
        )
        assert out.exists()

    def test_mel_db_clipping(self, test_audio_path: str, tmp_path) -> None:
        out = tmp_path / "mel.png"
        generate_spectrogram(
            test_audio_path,
            str(out),
            viz_type="mel",
            db_min=-60,
            db_max=0,
            backend="librosa",
        )
        assert out.exists()

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            generate_spectrogram(str(tmp_path / "missing.wav"), str(tmp_path / "out.png"))

    def test_invalid_type_raises(self, test_audio_path: str, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            generate_spectrogram(test_audio_path, str(tmp_path / "out.png"), viz_type="bogus")

    def test_invalid_window_raises(self, test_audio_path: str, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            generate_spectrogram(test_audio_path, str(tmp_path / "out.png"), window="bogus")

    def test_save_failure_raises(self, test_audio_path: str, tmp_path) -> None:
        with patch("matplotlib.figure.Figure.savefig", side_effect=RuntimeError("disk full")):
            with pytest.raises(ProcessingError):
                generate_spectrogram(test_audio_path, str(tmp_path / "out.png"), backend="librosa")


class TestComputeFunctions:
    def test_compute_stft(self, sample_audio_data: AudioData) -> None:
        freqs, times, mag = compute_stft(
            sample_audio_data.samples, sample_audio_data.sample_rate, backend="librosa"
        )
        assert mag.ndim == 2
        assert len(freqs) == mag.shape[0]

    def test_compute_mel_default_fmax(self, sample_audio_data: AudioData) -> None:
        times, mel = compute_mel_spectrogram(
            sample_audio_data.samples,
            sample_audio_data.sample_rate,
            n_mels=32,
            backend="librosa",
        )
        assert mel.shape[0] == 32

    def test_compute_mel_explicit_fmax(self, sample_audio_data: AudioData) -> None:
        _, mel = compute_mel_spectrogram(
            sample_audio_data.samples,
            sample_audio_data.sample_rate,
            n_mels=16,
            fmax=4000,
            backend="librosa",
        )
        assert mel.shape[0] == 16


class TestSpectrogramToDb:
    def test_max_ref(self) -> None:
        spec = np.abs(np.random.randn(10, 10)) + 0.1
        db = spectrogram_to_db(spec)
        assert db.shape == spec.shape
        assert np.all(db <= 1e-6)

    def test_float_ref(self) -> None:
        spec = np.ones((5, 5))
        db = spectrogram_to_db(spec, ref=1.0)
        assert db.shape == (5, 5)

    def test_no_top_db(self) -> None:
        spec = np.abs(np.random.randn(5, 5)) + 0.01
        db = spectrogram_to_db(spec, top_db=None)
        assert db.shape == (5, 5)


class TestSpectrogramToImage:
    def test_writes_png(self, tmp_path) -> None:
        spec = np.random.rand(64, 100)
        out = tmp_path / "spec.png"
        result = spectrogram_to_image(spec, str(out), title="Test", colorbar_label="dB")
        assert out.exists()
        assert result == str(out)

    def test_jpeg(self, tmp_path) -> None:
        spec = np.random.rand(32, 50)
        out = tmp_path / "spec.jpg"
        spectrogram_to_image(spec, str(out), colorbar=False)
        assert out.exists()

    def test_save_failure_raises(self, tmp_path) -> None:
        spec = np.random.rand(8, 8)
        with patch("matplotlib.figure.Figure.savefig", side_effect=RuntimeError("boom")):
            with pytest.raises(ProcessingError):
                spectrogram_to_image(spec, str(tmp_path / "x.png"))

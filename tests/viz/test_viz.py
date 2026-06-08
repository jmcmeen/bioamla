"""Tests for the viz domain (flattened, exception-based API)."""

import numpy as np
import pytest

from bioamla.audio import AudioData
from bioamla.exceptions import DependencyError, NotFoundError
from bioamla.viz import (
    batch_generate_spectrograms,
    compute_mel_spectrogram,
    compute_stft,
    generate_spectrogram,
    spectrogram_to_db,
    spectrogram_to_image,
)


def _torchaudio_available() -> bool:
    try:
        import torchaudio  # noqa: F401

        return True
    except ImportError:
        return False


class TestComputeFunctions:
    def test_compute_stft(self, sample_audio_data: AudioData) -> None:
        freqs, times, mag = compute_stft(
            sample_audio_data.samples, sample_audio_data.sample_rate
        )
        assert mag.ndim == 2
        assert len(freqs) == mag.shape[0]
        assert len(times) == mag.shape[1]

    def test_compute_mel_spectrogram(self, sample_audio_data: AudioData) -> None:
        times, mel = compute_mel_spectrogram(
            sample_audio_data.samples, sample_audio_data.sample_rate, n_mels=64
        )
        assert mel.shape[0] == 64
        assert len(times) == mel.shape[1]

    def test_spectrogram_to_db(self, sample_audio_data: AudioData) -> None:
        _, _, mag = compute_stft(
            sample_audio_data.samples, sample_audio_data.sample_rate
        )
        db = spectrogram_to_db(mag**2)
        assert db.shape == mag.shape
        assert np.all(db <= 0.0 + 1e-6)


class TestSpectrogramToImage:
    def test_writes_image(self, sample_audio_data: AudioData, tmp_path) -> None:
        _, mel = compute_mel_spectrogram(
            sample_audio_data.samples, sample_audio_data.sample_rate, n_mels=64
        )
        out = tmp_path / "spec.png"
        result = spectrogram_to_image(spectrogram_to_db(mel), str(out))
        assert out.exists()
        assert result == str(out)


class TestGenerateSpectrogram:
    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            generate_spectrogram(
                str(tmp_path / "missing.wav"), str(tmp_path / "out.png")
            )

    def test_invalid_type_raises(self, test_audio_path: str, tmp_path) -> None:
        with pytest.raises(ValueError):
            generate_spectrogram(
                test_audio_path, str(tmp_path / "out.png"), viz_type="nope"
            )

    @pytest.mark.skipif(
        not _torchaudio_available(), reason="torchaudio not installed"
    )
    def test_generate_writes_png(self, test_audio_path: str, tmp_path) -> None:
        out = tmp_path / "spec.png"
        result = generate_spectrogram(
            test_audio_path, str(out), viz_type="mel", sample_rate=16000
        )
        assert out.exists()
        assert result == str(out)

    def test_generate_without_torchaudio_raises(
        self, test_audio_path: str, tmp_path
    ) -> None:
        if _torchaudio_available():
            pytest.skip("torchaudio installed; dependency path not exercised")
        with pytest.raises(DependencyError):
            generate_spectrogram(test_audio_path, str(tmp_path / "out.png"))


class TestBatch:
    @pytest.mark.skipif(
        not _torchaudio_available(), reason="torchaudio not installed"
    )
    def test_batch_generate(self, test_audio_dir: str, tmp_path) -> None:
        out_dir = tmp_path / "out"
        result = batch_generate_spectrograms(
            test_audio_dir, str(out_dir), verbose=False
        )
        assert result["files_processed"] == 3
        assert result["files_failed"] == 0

    def test_batch_missing_dir_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            batch_generate_spectrograms(
                str(tmp_path / "nope"), str(tmp_path / "out"), verbose=False
            )

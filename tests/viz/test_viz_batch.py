"""Coverage tests for bioamla.viz.batch."""

import matplotlib

matplotlib.use("Agg")

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from bioamla.exceptions import NotFoundError  # noqa: E402
from bioamla.viz.batch import batch_generate_spectrograms  # noqa: E402


class TestBatchGenerateSpectrograms:
    def test_generates_all(self, test_audio_dir: str, tmp_path) -> None:
        stats = batch_generate_spectrograms(test_audio_dir, str(tmp_path / "out"), verbose=True)
        assert stats["files_processed"] == 3
        assert stats["files_failed"] == 0

    def test_jpg_format(self, test_audio_dir: str, tmp_path) -> None:
        stats = batch_generate_spectrograms(
            test_audio_dir,
            str(tmp_path / "out"),
            format="jpg",
            verbose=False,
        )
        assert stats["files_processed"] == 3

    def test_progress_callback(self, test_audio_dir: str, tmp_path) -> None:
        calls = []
        batch_generate_spectrograms(
            test_audio_dir,
            str(tmp_path / "out"),
            verbose=False,
            on_progress=lambda c, t: calls.append((c, t)),
        )
        assert len(calls) == 3
        assert calls[-1] == (3, 3)

    def test_missing_input_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            batch_generate_spectrograms(str(tmp_path / "nope"), str(tmp_path / "out"))

    def test_empty_dir(self, tmp_path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        stats = batch_generate_spectrograms(str(empty), str(tmp_path / "out"), verbose=True)
        assert stats["files_processed"] == 0

    def test_per_file_failure_counted(self, test_audio_dir: str, tmp_path) -> None:
        with patch(
            "bioamla.viz.batch.generate_spectrogram",
            side_effect=RuntimeError("render fail"),
        ):
            stats = batch_generate_spectrograms(test_audio_dir, str(tmp_path / "out"), verbose=True)
        assert stats["files_failed"] == 3
        assert stats["files_processed"] == 0

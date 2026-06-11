"""Coverage tests for bioamla.ml.batch.

The per-file predict/embed is mocked (ASTInference / EmbeddingExtractor patched)
so the batch wiring is exercised without loading any model.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bioamla.exceptions import NotFoundError
from bioamla.ml.batch import (
    _audio_filter,
    batch_embed_files,
    batch_predict_files,
    batch_predict_segments,
)


class TestAudioFilter:
    def test_filter(self, tmp_path) -> None:
        assert _audio_filter(tmp_path / "a.WAV") is True
        assert _audio_filter(tmp_path / "a.mp3") is True
        assert _audio_filter(tmp_path / "a.txt") is False


class TestBatchPredictFiles:
    def test_missing_dir(self) -> None:
        with pytest.raises(NotFoundError, match="Input directory not found"):
            batch_predict_files("/no/such/dir")

    def test_wires_predictions(self, test_audio_dir) -> None:
        fake_inf = MagicMock()
        fake_inf.predict_topk.return_value = SimpleNamespace(
            predicted_label="a",
            confidence=0.9,
            start_time=0.0,
            end_time=1.0,
            top_k_labels=["a"],
            top_k_scores=[0.9],
        )
        with patch("bioamla.ml.inference.ASTInference", return_value=fake_inf):
            result = batch_predict_files(test_audio_dir, model_path="hub/model")
        assert result.successful == 3
        preds = result.metadata["predictions"]
        assert len(preds) == 3
        assert preds[0]["predicted_label"] == "a"


class TestBatchPredictSegments:
    def test_missing_dir(self) -> None:
        with pytest.raises(NotFoundError, match="Input directory not found"):
            batch_predict_segments("/no/such/dir", segment_duration=3)

    def test_wires_segments_and_min_confidence(self, test_audio_dir) -> None:
        fake_inf = MagicMock()
        fake_inf.predict_segments.return_value = [
            SimpleNamespace(start_time=0.0, end_time=3.0, predicted_label="a", confidence=0.9),
            SimpleNamespace(start_time=3.0, end_time=6.0, predicted_label="b", confidence=0.1),
        ]
        with patch("bioamla.ml.inference.ASTInference", return_value=fake_inf):
            result = batch_predict_segments(test_audio_dir, segment_duration=3, min_confidence=0.5)
        # Each of 3 files yields 2 segments, but only 1 passes the 0.5 threshold.
        segs = result.metadata["segments"]
        assert len(segs) == 3
        assert all(s["confidence"] >= 0.5 for s in segs)


class TestBatchEmbedFiles:
    def test_missing_dir(self, tmp_path) -> None:
        with pytest.raises(NotFoundError, match="Input directory not found"):
            batch_embed_files("/no/such/dir", str(tmp_path / "out"))

    def test_writes_npy_per_file(self, test_audio_dir, tmp_path) -> None:
        out_dir = tmp_path / "out"
        fake_extractor = MagicMock()
        fake_extractor.extract.return_value = SimpleNamespace(
            embeddings=np.ones((1, 8), dtype=np.float32)
        )
        with patch("bioamla.ml.embedding.EmbeddingExtractor", return_value=fake_extractor):
            result = batch_embed_files(test_audio_dir, str(out_dir), model_path="hub/model")
        assert result.successful == 3
        npy_files = list(out_dir.glob("*_embeddings.npy"))
        assert len(npy_files) == 3
        loaded = np.load(str(npy_files[0]))
        assert loaded.shape == (1, 8)

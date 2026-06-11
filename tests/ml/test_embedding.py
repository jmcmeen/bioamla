"""Coverage tests for bioamla.ml.embedding.

The ASTModel backend is mocked (its ``extract_embeddings`` returns a fake
hidden-state array) so no model is downloaded. Covers the extractor, iteration,
batch aggregation, save/load formats, and get_ast_model_info.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bioamla.exceptions import InvalidInputError, ModelError
from bioamla.ml.embedding import (
    BatchEmbeddingResult,
    EmbeddingConfig,
    EmbeddingExtractor,
    extract_embeddings,
    get_ast_model_info,
    load_embeddings,
    save_embeddings,
)


def _extractor_with_fake_model(emb=None, **cfg):
    """Build an EmbeddingExtractor whose backing ASTModel is a mock."""
    if emb is None:
        emb = np.ones((2, 8), dtype=np.float32)
    extractor = EmbeddingExtractor(config=EmbeddingConfig(**cfg))
    fake = MagicMock()
    fake.extract_embeddings.return_value = emb
    extractor._model = fake
    return extractor, fake


class TestResultDataclasses:
    def test_batch_embedding_result_props(self) -> None:
        r = BatchEmbeddingResult(
            embeddings=np.ones((2, 4)),
            filepaths=["a", "b"],
            total_files=4,
            files_processed=2,
            files_failed=2,
        )
        assert r.embedding_dim == 4
        assert r.success_rate == 50.0

    def test_batch_embedding_result_zero_files(self) -> None:
        r = BatchEmbeddingResult(
            embeddings=np.array([]), filepaths=[], total_files=0, files_processed=0, files_failed=0
        )
        assert r.success_rate == 0


class TestExtract:
    def test_extract_array_normalizes(self) -> None:
        extractor, fake = _extractor_with_fake_model(
            emb=np.array([[3.0, 4.0]], dtype=np.float32), normalize=True
        )
        result = extractor.extract(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        # L2-normalized -> unit norm
        np.testing.assert_allclose(np.linalg.norm(result.embeddings, axis=1), [1.0], atol=1e-5)
        assert result.filepath == "<array>"

    def test_extract_1d_reshaped(self) -> None:
        extractor, _ = _extractor_with_fake_model(emb=np.ones(8, dtype=np.float32), normalize=False)
        result = extractor.extract(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        assert result.embeddings.ndim == 2
        assert result.embeddings.shape == (1, 8)

    def test_extract_file_builds_segments(self) -> None:
        extractor, _ = _extractor_with_fake_model(
            emb=np.ones((2, 8), dtype=np.float32), normalize=False, clip_duration=5.0
        )
        with patch("bioamla.audio.get_audio_info", return_value=SimpleNamespace(duration=10.0)):
            result = extractor.extract("a.wav")
        assert result.filepath == "a.wav"
        assert len(result.segments) == 2

    def test_extract_file_segment_failure_swallowed(self) -> None:
        extractor, _ = _extractor_with_fake_model(emb=np.ones((1, 8), dtype=np.float32))
        with patch("bioamla.audio.get_audio_info", side_effect=RuntimeError("bad")):
            result = extractor.extract("a.wav")
        assert result.segments == []


class TestModelLoading:
    def test_unknown_model_type(self) -> None:
        ex = EmbeddingExtractor(config=EmbeddingConfig(model_type="bogus"))
        with pytest.raises(InvalidInputError, match="Unknown model type"):
            ex._get_model()

    def test_ast_model_loaded_once(self) -> None:
        ex = EmbeddingExtractor(config=EmbeddingConfig(model_type="ast"))
        fake_instance = MagicMock()
        with patch("bioamla.ml.ast_model.ASTModel", return_value=fake_instance) as ASTM:
            m1 = ex._get_model()
            m2 = ex._get_model()
        assert m1 is m2 is fake_instance
        ASTM.assert_called_once()
        fake_instance.load.assert_called_once()


class TestExtractIterAndBatch:
    def test_extract_iter_handles_failure(self) -> None:
        extractor, fake = _extractor_with_fake_model()
        fake.extract_embeddings.side_effect = [np.ones((1, 8)), RuntimeError("boom")]
        results = list(extractor.extract_iter(["good.wav", "bad.wav"]))
        assert results[0].embeddings.size > 0
        assert results[1].embeddings.size == 0
        assert "error" in results[1].metadata

    def test_extract_batch_mean(self) -> None:
        extractor, fake = _extractor_with_fake_model(
            emb=np.ones((3, 8), dtype=np.float32), normalize=False
        )
        result = extractor.extract_batch(["a.wav", "b.wav"], aggregate="mean")
        assert result.files_processed == 2
        assert result.embeddings.shape == (2, 8)

    def test_extract_batch_first_and_errors(self) -> None:
        extractor, fake = _extractor_with_fake_model(normalize=False)
        fake.extract_embeddings.side_effect = [np.ones((2, 8)), RuntimeError("boom")]
        result = extractor.extract_batch(["a.wav", "b.wav"], aggregate="first")
        assert result.files_processed == 1
        assert result.files_failed == 1
        assert len(result.errors) == 1

    def test_extract_batch_all_empty(self) -> None:
        extractor, fake = _extractor_with_fake_model()
        fake.extract_embeddings.side_effect = RuntimeError("boom")
        result = extractor.extract_batch(["a.wav"], aggregate="all")
        assert result.embeddings.size == 0
        assert result.files_processed == 0


class TestConvenienceExtract:
    def test_extract_embeddings_fn(self) -> None:
        fake = MagicMock()
        fake.extract_embeddings.return_value = np.ones((1, 8), dtype=np.float32)
        with patch("bioamla.ml.ast_model.ASTModel", return_value=fake):
            out = extract_embeddings(
                np.zeros(16000, dtype=np.float32),
                model_path="hub/model",
                sample_rate=16000,
                normalize=False,
            )
        assert out.shape == (1, 8)


class TestSaveLoadFormats:
    def test_parquet_roundtrip(self, tmp_path) -> None:
        pytest.importorskip("pyarrow")
        emb = np.random.rand(3, 5).astype(np.float32)
        files = ["a.wav", "b.wav", "c.wav"]
        out = tmp_path / "e.parquet"
        save_embeddings(emb, files, str(out), format="parquet")
        loaded, lf = load_embeddings(str(out))
        np.testing.assert_allclose(loaded, emb, rtol=1e-5)
        assert lf == files

    def test_csv_roundtrip(self, tmp_path) -> None:
        emb = np.random.rand(2, 4).astype(np.float32)
        files = ["x.wav", "y.wav"]
        out = tmp_path / "e.csv"
        save_embeddings(emb, files, str(out), format="csv")
        loaded, lf = load_embeddings(str(out))
        np.testing.assert_allclose(loaded, emb, rtol=1e-4)
        assert lf == files

    def test_npy_without_mapping(self, tmp_path) -> None:
        emb = np.ones((2, 3), dtype=np.float32)
        out = tmp_path / "e.npy"
        np.save(str(out), emb)  # no .files.txt
        loaded, lf = load_embeddings(str(out))
        np.testing.assert_allclose(loaded, emb)
        assert lf == []


class TestModelInfo:
    def test_get_ast_model_info(self) -> None:
        cfg = SimpleNamespace(
            id2label={i: f"c{i}" for i in range(15)},
            model_type="audio-spectrogram-transformer",
            num_labels=15,
            hidden_size=768,
        )
        with patch("transformers.AutoConfig.from_pretrained", return_value=cfg):
            info = get_ast_model_info("hub/model")
        assert info["num_classes"] == 15
        assert info["has_more_classes"] is True
        assert len(info["classes"]) == 10
        assert info["hidden_size"] == 768

    def test_get_ast_model_info_no_labels(self) -> None:
        cfg = SimpleNamespace(model_type="ast", num_labels=5, hidden_size=384)
        with patch("transformers.AutoConfig.from_pretrained", return_value=cfg):
            info = get_ast_model_info("hub/model")
        assert info["num_classes"] == 5
        assert info["classes"] == []
        assert info["has_more_classes"] is False

    def test_get_ast_model_info_failure(self) -> None:
        with patch("transformers.AutoConfig.from_pretrained", side_effect=RuntimeError("nope")):
            with pytest.raises(ModelError, match="Failed to load model config"):
                get_ast_model_info("hub/model")

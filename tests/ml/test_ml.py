"""Tests for the ml domain (flattened, exception-based API).

These tests deliberately avoid downloading heavy models. They exercise:
- the curated public API surface,
- plumbing / config dataclasses,
- error paths (NotFoundError, InvalidInputError, ModelError),
- save/load of embeddings (pure numpy, no model),
- batch wrappers' directory-not-found handling.
"""

import numpy as np
import pytest

from bioamla.exceptions import (
    InvalidInputError,
    ModelError,
    NotFoundError,
)

# =============================================================================
# Public API surface
# =============================================================================


class TestPublicApi:
    def test_imports(self) -> None:
        import bioamla.ml as ml

        for name in (
            # foundations preserved
            "get_device",
            "BaseAudioModel",
            "ModelConfig",
            "PredictionResult",
            # AST core
            "ASTModel",
            "InferenceConfig",
            "ast_predict",
            "load_pretrained_ast_model",
            # service-level
            "predict_file",
            "evaluate_directory",
            "extract_embeddings_file",
            "get_model_info",
            "EvaluationResult",
            "TrainResult",
            # inference
            "ASTInference",
            "ASTPredictionResult",
            # embeddings
            "EmbeddingConfig",
            "EmbeddingResult",
            "EmbeddingExtractor",
            "extract_embeddings",
            "save_embeddings",
            "load_embeddings",
            # batch
            "batch_predict_files",
            "batch_predict_segments",
            "batch_embed_files",
        ):
            assert hasattr(ml, name), f"missing public export: {name}"


# =============================================================================
# Config / dataclass plumbing
# =============================================================================


class TestConfigPlumbing:
    def test_inference_config_defaults(self) -> None:
        from bioamla.ml import InferenceConfig

        cfg = InferenceConfig()
        assert cfg.batch_size == 8
        assert cfg.use_fp16 is False
        assert cfg.num_workers == 1

    def test_embedding_config_model_path_override(self) -> None:
        from bioamla.ml import EmbeddingConfig, EmbeddingExtractor

        extractor = EmbeddingExtractor(model_path="some/model")
        assert extractor.config.model_path == "some/model"

        cfg = EmbeddingConfig(model_path="other/model", normalize=False)
        extractor2 = EmbeddingExtractor(config=cfg)
        assert extractor2.config.model_path == "other/model"
        assert extractor2.config.normalize is False

    def test_evaluation_result_to_dict(self) -> None:
        from bioamla.ml import EvaluationResult

        r = EvaluationResult(
            accuracy=0.9, precision=0.8, recall=0.7, f1_score=0.75, total_samples=10
        )
        d = r.to_dict()
        assert d["accuracy"] == 0.9
        assert d["total_samples"] == 10
        assert "confusion_matrix" not in d

    def test_train_result_to_dict(self) -> None:
        from bioamla.ml import TrainResult

        r = TrainResult(model_path="/tmp/m", epochs=3)
        d = r.to_dict()
        assert d["model_path"] == "/tmp/m"
        assert d["epochs"] == 3
        assert d["final_accuracy"] is None


# =============================================================================
# EmbeddingResult helpers (pure numpy)
# =============================================================================


class TestEmbeddingResult:
    def test_embedding_result_properties(self) -> None:
        from bioamla.ml import EmbeddingResult

        emb = np.ones((3, 8), dtype=np.float32)
        result = EmbeddingResult(filepath="a.wav", embeddings=emb, sample_rate=16000)
        assert result.embedding_dim == 8
        assert result.num_segments == 3
        assert result.mean_embedding().shape == (8,)

    def test_embedding_result_1d(self) -> None:
        from bioamla.ml import EmbeddingResult

        emb = np.ones(8, dtype=np.float32)
        result = EmbeddingResult(filepath="a.wav", embeddings=emb, sample_rate=16000)
        assert result.embedding_dim == 8
        assert result.num_segments == 1


# =============================================================================
# Save / load embeddings (no model required)
# =============================================================================


class TestSaveLoadEmbeddings:
    def test_roundtrip_npy(self, tmp_path) -> None:
        from bioamla.ml import load_embeddings, save_embeddings

        emb = np.random.rand(4, 16).astype(np.float32)
        files = ["a.wav", "b.wav", "c.wav", "d.wav"]
        out = tmp_path / "emb.npy"

        saved = save_embeddings(emb, files, str(out), format="npy")
        assert out.exists()
        assert (tmp_path / "emb.files.txt").exists()

        loaded, loaded_files = load_embeddings(saved)
        np.testing.assert_allclose(loaded, emb)
        assert loaded_files == files

    def test_roundtrip_npz(self, tmp_path) -> None:
        from bioamla.ml import load_embeddings, save_embeddings

        emb = np.random.rand(2, 8).astype(np.float32)
        files = ["x.wav", "y.wav"]
        out = tmp_path / "emb.npz"

        save_embeddings(emb, files, str(out), format="npz")
        loaded, loaded_files = load_embeddings(str(out))
        np.testing.assert_allclose(loaded, emb)
        assert list(loaded_files) == files

    def test_save_unknown_format(self, tmp_path) -> None:
        from bioamla.ml import save_embeddings

        emb = np.zeros((1, 4), dtype=np.float32)
        with pytest.raises(InvalidInputError, match="Unknown format"):
            save_embeddings(emb, ["a.wav"], str(tmp_path / "x.bogus"), format="bogus")

    def test_load_unknown_format(self, tmp_path) -> None:
        from bioamla.ml import load_embeddings

        bogus = tmp_path / "x.bogus"
        bogus.write_text("nope")
        with pytest.raises(InvalidInputError, match="Unknown format"):
            load_embeddings(str(bogus))


# =============================================================================
# Error paths (no heavy model downloads)
# =============================================================================


class TestErrorPaths:
    def test_predict_file_missing(self) -> None:
        from bioamla.ml import predict_file

        with pytest.raises(NotFoundError, match="not found"):
            predict_file("/no/such/file.wav", model_path="bioamla/scp-frogs")

    def test_extract_embeddings_file_missing(self) -> None:
        from bioamla.ml import extract_embeddings_file

        with pytest.raises(NotFoundError, match="not found"):
            extract_embeddings_file("/no/such/file.wav", model_path="bioamla/scp-frogs")

    def test_evaluate_directory_missing_audio_dir(self, tmp_path) -> None:
        from bioamla.ml import evaluate_directory

        gt = tmp_path / "gt.csv"
        gt.write_text("file_name,label\na.wav,frog\n")
        with pytest.raises(NotFoundError, match="Audio directory not found"):
            evaluate_directory(
                audio_dir="/no/such/dir",
                model_path="bioamla/scp-frogs",
                ground_truth_csv=str(gt),
            )

    def test_evaluate_directory_missing_csv(self, test_audio_dir: str) -> None:
        from bioamla.ml import evaluate_directory

        with pytest.raises(NotFoundError, match="Ground truth file not found"):
            evaluate_directory(
                audio_dir=test_audio_dir,
                model_path="bioamla/scp-frogs",
                ground_truth_csv="/no/such/gt.csv",
            )

    def test_batch_predict_files_missing_dir(self) -> None:
        from bioamla.ml import batch_predict_files

        with pytest.raises(NotFoundError, match="Input directory not found"):
            batch_predict_files("/no/such/dir")

    def test_batch_predict_segments_missing_dir(self) -> None:
        from bioamla.ml import batch_predict_segments

        with pytest.raises(NotFoundError, match="Input directory not found"):
            batch_predict_segments("/no/such/dir", segment_duration=3)

    def test_batch_embed_files_missing_dir(self, tmp_path) -> None:
        from bioamla.ml import batch_embed_files

        with pytest.raises(NotFoundError, match="Input directory not found"):
            batch_embed_files("/no/such/dir", str(tmp_path / "out"))


# =============================================================================
# Extractor model-type validation (no download)
# =============================================================================


class TestExtractorValidation:
    def test_unknown_model_type(self) -> None:
        from bioamla.ml import EmbeddingConfig, EmbeddingExtractor

        cfg = EmbeddingConfig(model_type="bogus")
        extractor = EmbeddingExtractor(config=cfg)
        with pytest.raises(InvalidInputError, match="Unknown model type"):
            extractor._get_model()


# =============================================================================
# AST low-level helpers raise ModelError on bad model paths
# =============================================================================


class TestAstHelpers:
    def test_load_pretrained_bad_local_path(self) -> None:
        pytest.importorskip("torch")  # loading a model needs torch
        from bioamla.ml import load_pretrained_ast_model

        # Looks local (starts with ./) and does not exist -> transformers fails.
        with pytest.raises(ModelError, match="Failed to load AST model"):
            load_pretrained_ast_model("./definitely-not-a-real-model-dir-xyz")

# tests/controllers/test_embedding.py
"""
Tests for EmbeddingController.
"""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from bioamla.controllers.embedding import (
    BatchEmbeddingSummary,
    EmbeddingController,
    EmbeddingInfo,
)


class TestEmbeddingController:
    """Tests for EmbeddingController."""

    @pytest.fixture
    def controller(self):
        return EmbeddingController(model_path="test-model")

    @pytest.fixture
    def mock_extractor(self, mocker):
        """Mock the EmbeddingExtractor."""
        mocker.patch("bioamla.core.ml.embeddings.EmbeddingConfig")
        mock_extractor_class = mocker.patch("bioamla.core.ml.embeddings.EmbeddingExtractor")

        # Create a mock result object
        mock_result = MagicMock()
        mock_result.embeddings = np.random.randn(1, 768).astype(np.float32)
        mock_result.embedding_dim = 768
        mock_result.num_segments = 1
        mock_result.segments = [(0.0, 1.0)]
        mock_result.mean_embedding.return_value = np.random.randn(768).astype(np.float32)

        instance = Mock()
        instance.extract.return_value = mock_result
        mock_extractor_class.return_value = instance

        return instance

    def test_extract_valid_file_success(self, controller, tmp_audio_file, mock_extractor):
        """Test that extraction on a valid audio file succeeds."""
        result = controller.extract(tmp_audio_file)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, EmbeddingInfo)
        assert result.data.embedding_dim == 768
        assert result.data.num_segments == 1

    def test_extract_nonexistent_file_fails(self, controller):
        """Test that extraction on a nonexistent file fails with error."""
        result = controller.extract("/nonexistent/path/audio.wav")

        assert result.success is False
        assert result.error is not None
        assert "does not exist" in result.error

    def test_extract_batch_success(
        self, controller, tmp_dir_with_audio_files, mock_extractor, mocker
    ):
        """Test that batch extraction on multiple files succeeds."""
        result = controller.extract_batch(tmp_dir_with_audio_files)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, BatchEmbeddingSummary)
        assert result.data.total_files == 3
        assert result.data.files_processed == 3
        assert result.data.files_failed == 0


class TestDimensionalityReduction:
    """Tests for dimensionality reduction methods."""

    @pytest.fixture
    def controller(self):
        return EmbeddingController()

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing reduction."""
        return np.random.randn(10, 768).astype(np.float32)

    def test_reduce_dimensions_success(self, controller, sample_embeddings, mocker):
        """Test that dimensionality reduction succeeds."""
        mock_reduce = mocker.patch("bioamla.core.analysis.clustering.reduce_dimensions")
        mock_reduce.return_value = np.random.randn(10, 2).astype(np.float32)

        result = controller.reduce_dimensions(sample_embeddings, method="pca", n_components=2)

        assert result.success is True
        assert result.data is not None
        assert result.data["reduced_shape"] == (10, 2)
        assert result.data["method"] == "pca"

    def test_reduce_for_visualization_returns_coordinates(
        self, controller, sample_embeddings, mocker
    ):
        """Test that visualization reduction returns x,y coordinates."""
        mock_reduce = mocker.patch("bioamla.core.analysis.clustering.reduce_dimensions")
        mock_reduce.return_value = np.random.randn(10, 2).astype(np.float32)

        result = controller.reduce_for_visualization(sample_embeddings)

        assert result.success is True
        assert "x" in result.data
        assert "y" in result.data
        assert len(result.data["x"]) == 10
        assert len(result.data["y"]) == 10


class TestLoadSaveEmbeddings:
    """Tests for loading and saving embeddings."""

    @pytest.fixture
    def controller(self):
        return EmbeddingController()

    def test_load_embeddings_nonexistent_fails(self, controller):
        """Test that loading nonexistent file fails."""
        result = controller.load_embeddings("/nonexistent/embeddings.npy")

        assert result.success is False
        assert "does not exist" in result.error

    def test_save_embeddings_success(self, controller, tmp_path, mocker):
        """Test that saving embeddings succeeds."""
        mock_save = mocker.patch("bioamla.core.ml.embeddings.save_embeddings")
        output_path = str(tmp_path / "embeddings.npy")
        mock_save.return_value = output_path

        embeddings = np.random.randn(5, 768).astype(np.float32)
        filepaths = [f"audio_{i}.wav" for i in range(5)]

        result = controller.save_embeddings(embeddings, filepaths, output_path)

        assert result.success is True
        assert result.data["output_path"] == output_path
        assert result.data["shape"] == (5, 768)

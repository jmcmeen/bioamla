# tests/controllers/test_inference.py
"""
Tests for InferenceController.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from bioamla.controllers.inference import (
    BatchInferenceResult,
    InferenceController,
    PredictionResult,
)


class TestInferenceController:
    """Tests for InferenceController."""

    @pytest.fixture
    def controller(self):
        return InferenceController(model_path="test-model")

    @pytest.fixture
    def mock_model(self, mocker):
        """Mock the ASTInference model."""
        mock = mocker.patch("bioamla.core.ml.inference.ASTInference")
        instance = Mock()
        instance.predict.return_value = [
            {
                "label": "speech",
                "confidence": 0.95,
                "start_time": 0.0,
                "end_time": 1.0,
                "top_k_labels": ["speech", "music"],
                "top_k_scores": [0.95, 0.03],
            }
        ]
        instance.get_embeddings.return_value = np.random.randn(1, 768).astype(np.float32)
        mock.return_value = instance
        return instance

    def test_predict_valid_file_success(self, controller, tmp_audio_file, mock_model):
        """Test that prediction on a valid audio file succeeds."""
        result = controller.predict(tmp_audio_file)

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 1
        assert isinstance(result.data[0], PredictionResult)
        assert result.data[0].predicted_label == "speech"
        assert result.data[0].confidence == 0.95

    def test_predict_nonexistent_file_fails(self, controller):
        """Test that prediction on a nonexistent file fails with error."""
        result = controller.predict("/nonexistent/path/audio.wav")

        assert result.success is False
        assert result.error is not None
        assert "does not exist" in result.error

    def test_predict_batch_success(self, controller, tmp_dir_with_audio_files, mock_model, mocker):
        """Test that batch prediction on multiple files succeeds."""
        # Mock storage for run tracking
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_complete_run")
        mocker.patch("bioamla.core.files.TextFile")

        result = controller.predict_batch(tmp_dir_with_audio_files)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, BatchInferenceResult)
        assert result.data.summary.total_files == 3
        assert len(result.data.predictions) == 3  # One prediction per file


class TestExtractEmbeddings:
    """Tests for embedding extraction methods."""

    @pytest.fixture
    def controller(self):
        return InferenceController(model_path="test-model")

    @pytest.fixture
    def mock_model(self, mocker):
        """Mock the ASTInference model."""
        mock = mocker.patch("bioamla.core.ml.inference.ASTInference")
        instance = Mock()
        instance.get_embeddings.return_value = np.random.randn(1, 768).astype(np.float32)
        mock.return_value = instance
        return instance

    def test_extract_embeddings_success(self, controller, tmp_audio_file, mock_model):
        """Test that embedding extraction succeeds."""
        result = controller.extract_embeddings(tmp_audio_file)

        assert result.success is True
        assert result.data is not None
        assert "shape" in result.data
        assert result.data["shape"] == (1, 768)

    def test_extract_embeddings_nonexistent_file_fails(self, controller):
        """Test that embedding extraction on nonexistent file fails."""
        result = controller.extract_embeddings("/nonexistent/path/audio.wav")

        assert result.success is False
        assert "does not exist" in result.error

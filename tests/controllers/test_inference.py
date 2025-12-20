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

    def test_extract_embeddings_saves_to_file(self, controller, tmp_audio_file, tmp_path, mock_model):
        """Test that embeddings can be saved to file."""
        output_path = str(tmp_path / "embeddings.npy")
        result = controller.extract_embeddings(tmp_audio_file, output_path=output_path)

        assert result.success is True
        assert (tmp_path / "embeddings.npy").exists()


class TestBatchEmbeddingExtraction:
    """Tests for batch embedding extraction."""

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

    def test_extract_embeddings_batch_success(
        self, controller, tmp_dir_with_audio_files, tmp_path, mock_model, mocker
    ):
        """Test that batch embedding extraction succeeds."""
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_complete_run")

        output_path = str(tmp_path / "embeddings.npy")
        result = controller.extract_embeddings_batch(
            tmp_dir_with_audio_files,
            output_path=output_path,
            format="npy",
        )

        assert result.success is True
        assert result.data["extracted"] == 3
        assert (tmp_path / "embeddings.npy").exists()

    def test_extract_embeddings_batch_empty_dir_fails(
        self, controller, tmp_path, mock_model, mocker
    ):
        """Test that batch embedding fails with empty directory."""
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_fail_run")

        # Empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = controller.extract_embeddings_batch(
            str(empty_dir),
            output_path=str(tmp_path / "out.npy"),
        )

        assert result.success is False
        assert "No audio files" in result.error


class TestModelInfo:
    """Tests for model information methods."""

    @pytest.fixture
    def controller(self):
        return InferenceController(model_path="test-model")

    @pytest.fixture
    def mock_model(self, mocker):
        """Mock the ASTInference model."""
        mock = mocker.patch("bioamla.core.ml.inference.ASTInference")
        instance = Mock()
        instance.num_labels = 527
        instance.labels = ["speech", "music", "noise"]
        mock.return_value = instance
        return instance

    def test_get_model_info_success(self, controller, mock_model):
        """Test that model info is returned."""
        result = controller.get_model_info()

        assert result.success is True
        assert result.data is not None
        assert result.data["model_path"] == "test-model"
        assert result.data["num_labels"] == 527

    def test_list_available_models_success(self, controller):
        """Test that list_available_models returns models."""
        result = controller.list_available_models()

        assert result.success is True
        assert result.data is not None
        assert len(result.data) > 0
        assert all("name" in m and "description" in m for m in result.data)


class TestMinimumConfidenceFiltering:
    """Tests for minimum confidence filtering."""

    @pytest.fixture
    def controller(self):
        return InferenceController(model_path="test-model")

    @pytest.fixture
    def mock_model(self, mocker):
        """Mock the ASTInference model with varied confidence scores."""
        mock = mocker.patch("bioamla.core.ml.inference.ASTInference")
        instance = Mock()
        instance.predict.return_value = [
            {"label": "speech", "confidence": 0.95, "start_time": 0.0, "end_time": 1.0},
            {"label": "music", "confidence": 0.30, "start_time": 1.0, "end_time": 2.0},
            {"label": "noise", "confidence": 0.10, "start_time": 2.0, "end_time": 3.0},
        ]
        mock.return_value = instance
        return instance

    def test_predict_with_min_confidence_filters(self, controller, tmp_audio_file, mock_model):
        """Test that min_confidence filters out low-confidence predictions."""
        result = controller.predict(tmp_audio_file, min_confidence=0.5)

        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0].predicted_label == "speech"

    def test_predict_with_high_min_confidence_empty(self, controller, tmp_audio_file, mock_model):
        """Test that high min_confidence returns empty results."""
        result = controller.predict(tmp_audio_file, min_confidence=0.99)

        assert result.success is True
        assert len(result.data) == 0


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_prediction_result_fields(self):
        """Test that PredictionResult has all expected fields."""
        pred = PredictionResult(
            filepath="/path/to/file.wav",
            start_time=0.0,
            end_time=1.0,
            predicted_label="speech",
            confidence=0.95,
            top_k_labels=["speech", "music"],
            top_k_scores=[0.95, 0.03],
        )

        assert pred.filepath == "/path/to/file.wav"
        assert pred.start_time == 0.0
        assert pred.end_time == 1.0
        assert pred.predicted_label == "speech"
        assert pred.confidence == 0.95
        assert pred.top_k_labels == ["speech", "music"]
        assert pred.top_k_scores == [0.95, 0.03]


class TestInferenceControllerNoModel:
    """Tests for InferenceController without model specified."""

    def test_predict_without_model_fails(self, tmp_audio_file):
        """Test that predict fails when no model is specified."""
        controller = InferenceController()

        result = controller.predict(tmp_audio_file)

        assert result.success is False
        assert "No model path" in result.error or "model" in result.error.lower()

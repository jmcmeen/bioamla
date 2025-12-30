"""Tests for CNNService - CNN model operations via adapter."""

import pytest

from bioamla.services.cnn import CNNPrediction, CNNService, CNNTrainResult


class TestCNNPrediction:
    """Tests for CNNPrediction dataclass."""

    def test_create_prediction(self) -> None:
        """Test creating a CNNPrediction."""
        pred = CNNPrediction(
            filepath="/test/audio.wav",
            start_time=0.0,
            end_time=3.0,
            label="bird",
            confidence=0.95,
        )

        assert pred.filepath == "/test/audio.wav"
        assert pred.start_time == 0.0
        assert pred.end_time == 3.0
        assert pred.label == "bird"
        assert pred.confidence == 0.95

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        pred = CNNPrediction(
            filepath="/test/audio.wav",
            start_time=1.0,
            end_time=4.0,
            label="frog",
            confidence=0.85,
        )

        d = pred.to_dict()

        assert d["filepath"] == "/test/audio.wav"
        assert d["start_time"] == 1.0
        assert d["end_time"] == 4.0
        assert d["label"] == "frog"
        assert d["confidence"] == 0.85


class TestCNNTrainResult:
    """Tests for CNNTrainResult dataclass."""

    def test_create_train_result(self) -> None:
        """Test creating a CNNTrainResult."""
        result = CNNTrainResult(
            model_path="/models/model.pt",
            epochs=10,
            num_classes=5,
            architecture="resnet18",
        )

        assert result.model_path == "/models/model.pt"
        assert result.epochs == 10
        assert result.num_classes == 5
        assert result.architecture == "resnet18"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        result = CNNTrainResult(
            model_path="/models/model.pt",
            epochs=20,
            num_classes=3,
            architecture="efficientnet_b0",
        )

        d = result.to_dict()

        assert d["model_path"] == "/models/model.pt"
        assert d["epochs"] == 20
        assert d["num_classes"] == 3
        assert d["architecture"] == "efficientnet_b0"


class TestCNNServiceUnit:
    """Unit tests for CNNService (no model loading)."""

    def test_service_init(self, mock_repository) -> None:
        """Test service initialization."""
        service = CNNService(mock_repository)

        assert service._adapter is None
        assert service._model_path is None

    def test_list_architectures(self, mock_repository) -> None:
        """Test listing available architectures."""
        service = CNNService(mock_repository)

        result = service.list_architectures()

        assert result.success
        assert isinstance(result.data, list)
        assert "resnet18" in result.data
        assert "resnet50" in result.data
        assert "efficientnet_b0" in result.data
        assert "densenet121" in result.data
        assert "inception_v3" in result.data

    def test_predict_invalid_filepath(self, mock_repository) -> None:
        """Test predict with non-existent file."""
        service = CNNService(mock_repository)

        result = service.predict(
            filepath="/nonexistent/audio.wav",
            model_path="/some/model.pt",
        )

        assert not result.success
        assert "does not exist" in result.error

    def test_predict_invalid_model_path(self, mock_repository, test_audio_path) -> None:
        """Test predict with non-existent model."""
        service = CNNService(mock_repository)

        result = service.predict(
            filepath=test_audio_path,
            model_path="/nonexistent/model.pt",
        )

        assert not result.success
        assert "does not exist" in result.error

    def test_extract_embeddings_invalid_filepath(self, mock_repository) -> None:
        """Test extract_embeddings with non-existent file."""
        service = CNNService(mock_repository)

        result = service.extract_embeddings(
            filepath="/nonexistent/audio.wav",
            model_path="/some/model.pt",
        )

        assert not result.success
        assert "does not exist" in result.error

    def test_get_model_info_invalid_path(self, mock_repository) -> None:
        """Test get_model_info with non-existent model."""
        service = CNNService(mock_repository)

        result = service.get_model_info("/nonexistent/model.pt")

        assert not result.success
        assert "does not exist" in result.error

    def test_train_without_model(self, mock_repository, test_audio_path) -> None:
        """Test train fails without creating model first."""
        service = CNNService(mock_repository)

        result = service.train(
            train_csv=test_audio_path,  # Using any existing file as placeholder
            output_dir="/tmp/output",
        )

        assert not result.success
        assert "No model loaded" in result.error

    def test_train_invalid_csv_path(self, mock_repository) -> None:
        """Test train with non-existent CSV."""
        service = CNNService(mock_repository)
        # Manually set adapter to bypass the "no model" check
        service._adapter = object()

        result = service.train(
            train_csv="/nonexistent/train.csv",
            output_dir="/tmp/output",
        )

        assert not result.success
        assert "does not exist" in result.error


@pytest.mark.slow
class TestCNNServiceCreateModel:
    """Tests for CNNService model creation (requires opensoundscape)."""

    def test_create_model_basic(self, mock_repository) -> None:
        """Test creating a basic CNN model."""
        service = CNNService(mock_repository)

        result = service.create_model(
            classes=["bird", "frog", "insect"],
            architecture="resnet18",
            sample_duration=3.0,
            sample_rate=16000,
        )

        assert result.success
        assert result.data is not None
        assert result.data["architecture"] == "resnet18"
        assert result.data["num_classes"] == 3
        assert result.data["classes"] == ["bird", "frog", "insect"]
        assert result.data["sample_duration"] == 3.0
        assert service._adapter is not None

    def test_create_model_different_architecture(self, mock_repository) -> None:
        """Test creating model with different architecture."""
        service = CNNService(mock_repository)

        result = service.create_model(
            classes=["class_a", "class_b"],
            architecture="resnet50",
            sample_duration=5.0,
            sample_rate=22050,
        )

        assert result.success
        assert result.data["architecture"] == "resnet50"
        assert result.data["num_classes"] == 2
        assert result.data["sample_duration"] == 5.0

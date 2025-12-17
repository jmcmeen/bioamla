"""
Unit tests for bioamla.models module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from bioamla.models.base import (
    AudioDataset,
    BaseAudioModel,
    BatchPredictionResult,
    ModelBackend,
    ModelConfig,
    PredictionResult,
    create_dataloader,
    get_model_class,
    list_models,
    register_model,
)
from bioamla.models.opensoundscape import OpenSoundscapeModel, SpectrogramCNN
from bioamla.models.trainer import (
    ModelTrainer,
    SpectrogramDataset,
    TrainingConfig,
    TrainingMetrics,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()

        assert config.sample_rate == 16000
        assert config.clip_duration == 3.0
        assert config.overlap == 0.0
        assert config.min_confidence == 0.0
        assert config.top_k == 1
        assert config.batch_size == 8
        assert config.num_workers == 4
        assert config.use_fp16 is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            sample_rate=48000,
            clip_duration=5.0,
            min_confidence=0.5,
            top_k=5,
            batch_size=16,
        )

        assert config.sample_rate == 48000
        assert config.clip_duration == 5.0
        assert config.min_confidence == 0.5
        assert config.top_k == 5
        assert config.batch_size == 16

    def test_get_device_cpu(self):
        """Test get_device returns CPU when specified."""
        config = ModelConfig(device="cpu")
        device = config.get_device()
        assert device.type == "cpu"

    def test_get_device_auto(self):
        """Test get_device auto-detects device."""
        config = ModelConfig()
        device = config.get_device()
        assert device.type in ["cpu", "cuda"]


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_basic_result(self):
        """Test creating a basic prediction result."""
        result = PredictionResult(
            label="bird",
            confidence=0.95,
            start_time=0.0,
            end_time=3.0,
        )

        assert result.label == "bird"
        assert result.confidence == 0.95
        assert result.start_time == 0.0
        assert result.end_time == 3.0

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = PredictionResult(
            label="frog",
            confidence=0.85,
            start_time=1.0,
            end_time=4.0,
            filepath="/path/to/audio.wav",
        )

        d = result.to_dict()

        assert d["label"] == "frog"
        assert d["confidence"] == 0.85
        assert d["start_time"] == 1.0
        assert d["end_time"] == 4.0
        assert d["filepath"] == "/path/to/audio.wav"

    def test_with_embeddings(self):
        """Test result with embeddings."""
        embeddings = np.random.randn(512).astype(np.float32)
        result = PredictionResult(
            label="insect",
            confidence=0.75,
            embeddings=embeddings,
        )

        assert result.embeddings is not None
        assert result.embeddings.shape == (512,)

    def test_with_metadata(self):
        """Test result with custom metadata."""
        result = PredictionResult(
            label="bird",
            confidence=0.9,
            metadata={"species": "robin", "region": "northeast"},
        )

        assert result.metadata["species"] == "robin"
        assert result.metadata["region"] == "northeast"


class TestBatchPredictionResult:
    """Tests for BatchPredictionResult dataclass."""

    def test_batch_result(self):
        """Test creating batch prediction result."""
        predictions = [
            PredictionResult(label="bird", confidence=0.9),
            PredictionResult(label="frog", confidence=0.8),
        ]

        batch_result = BatchPredictionResult(
            predictions=predictions,
            total_files=5,
            files_processed=4,
            files_failed=1,
            processing_time=10.5,
        )

        assert len(batch_result.predictions) == 2
        assert batch_result.total_files == 5
        assert batch_result.files_processed == 4
        assert batch_result.files_failed == 1
        assert batch_result.processing_time == 10.5

    def test_to_dict(self):
        """Test converting batch result to dictionary."""
        predictions = [
            PredictionResult(label="bird", confidence=0.9),
        ]

        batch_result = BatchPredictionResult(
            predictions=predictions,
            total_files=1,
            files_processed=1,
            files_failed=0,
            processing_time=1.0,
        )

        d = batch_result.to_dict()

        assert len(d["predictions"]) == 1
        assert d["total_files"] == 1
        assert d["processing_time"] == 1.0


class TestModelRegistry:
    """Tests for model registry functions."""

    def test_list_models_includes_registered(self):
        """Test that registered models are listed."""
        # Import model classes to trigger registration
        from bioamla.models.ast_model import ASTModel
        from bioamla.models.birdnet import BirdNETModel

        models = list_models()

        # Should include our registered models
        assert "ast" in models
        assert "birdnet" in models
        assert "opensoundscape" in models

    def test_get_model_class_ast(self):
        """Test getting AST model class."""
        # Import to trigger registration
        from bioamla.models.ast_model import ASTModel
        cls = get_model_class("ast")
        assert cls.__name__ == "ASTModel"

    def test_get_model_class_birdnet(self):
        """Test getting BirdNET model class."""
        # Import to trigger registration
        from bioamla.models.birdnet import BirdNETModel
        cls = get_model_class("birdnet")
        assert cls.__name__ == "BirdNETModel"

    def test_get_model_class_opensoundscape(self):
        """Test getting OpenSoundscape model class."""
        cls = get_model_class("opensoundscape")
        assert cls.__name__ == "OpenSoundscapeModel"

    def test_get_model_class_unknown(self):
        """Test error on unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_class("unknown_model")

    def test_register_model_decorator(self):
        """Test registering a custom model."""
        @register_model("test_model")
        class TestModel(BaseAudioModel):
            @property
            def backend(self):
                return ModelBackend.CUSTOM_CNN

            def load(self, model_path, **kwargs):
                return self

            def predict(self, audio, sample_rate=None):
                return []

            def extract_embeddings(self, audio, sample_rate=None, layer=None):
                return np.zeros(512)

        assert "test_model" in list_models()


class TestSpectrogramCNN:
    """Tests for SpectrogramCNN architecture."""

    def test_create_resnet18(self):
        """Test creating ResNet18 model."""
        model = SpectrogramCNN(num_classes=10, architecture="resnet18")

        assert model.num_classes == 10
        assert model.architecture == "resnet18"

    def test_create_resnet50(self):
        """Test creating ResNet50 model."""
        model = SpectrogramCNN(num_classes=20, architecture="resnet50")

        assert model.num_classes == 20
        assert model.architecture == "resnet50"

    def test_forward_pass(self):
        """Test forward pass through model."""
        model = SpectrogramCNN(num_classes=5, architecture="resnet18", pretrained=False)
        model.eval()

        # Create dummy input (batch, channels, height, width)
        x = torch.randn(2, 1, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 5)

    def test_get_embeddings(self):
        """Test extracting embeddings."""
        model = SpectrogramCNN(num_classes=5, architecture="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(2, 1, 224, 224)

        with torch.no_grad():
            embeddings = model.get_embeddings(x)

        # Default embedding dim for resnet18 is 512
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0

    def test_freeze_backbone(self):
        """Test freezing backbone weights."""
        model = SpectrogramCNN(num_classes=5, architecture="resnet18", pretrained=False)

        model.freeze_backbone()

        # Check that backbone params are frozen
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_unfreeze_backbone(self):
        """Test unfreezing backbone weights."""
        model = SpectrogramCNN(num_classes=5, architecture="resnet18", pretrained=False)

        model.freeze_backbone()
        model.unfreeze_backbone()

        # Check that backbone params are unfrozen
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_unsupported_architecture(self):
        """Test error on unsupported architecture."""
        with pytest.raises(ValueError, match="Unsupported architecture"):
            SpectrogramCNN(num_classes=5, architecture="vgg16")


class TestOpenSoundscapeModel:
    """Tests for OpenSoundscapeModel."""

    def test_create_model(self):
        """Test creating a new model for training."""
        model = OpenSoundscapeModel.create(
            num_classes=3,
            architecture="resnet18",
            class_names=["bird", "frog", "insect"],
            pretrained=False,
        )

        assert model.num_classes == 3
        assert model.backend == ModelBackend.OPENSOUNDSCAPE
        assert model.id2label == {0: "bird", 1: "frog", 2: "insect"}

    def test_filter_predictions(self):
        """Test filtering predictions by confidence."""
        model = OpenSoundscapeModel()
        model.config.min_confidence = 0.5

        predictions = [
            PredictionResult(label="bird", confidence=0.9),
            PredictionResult(label="frog", confidence=0.3),
            PredictionResult(label="insect", confidence=0.7),
        ]

        filtered = model.filter_predictions(predictions)

        assert len(filtered) == 2
        assert all(p.confidence >= 0.5 for p in filtered)

    def test_filter_predictions_by_label(self):
        """Test filtering predictions by label."""
        model = OpenSoundscapeModel()

        predictions = [
            PredictionResult(label="bird", confidence=0.9),
            PredictionResult(label="frog", confidence=0.8),
            PredictionResult(label="insect", confidence=0.7),
        ]

        filtered = model.filter_predictions(predictions, labels=["bird", "frog"])

        assert len(filtered) == 2
        assert all(p.label in ["bird", "frog"] for p in filtered)

    def test_filter_predictions_exclude_label(self):
        """Test filtering predictions excluding labels."""
        model = OpenSoundscapeModel()

        predictions = [
            PredictionResult(label="bird", confidence=0.9),
            PredictionResult(label="frog", confidence=0.8),
            PredictionResult(label="noise", confidence=0.7),
        ]

        filtered = model.filter_predictions(predictions, exclude_labels=["noise"])

        assert len(filtered) == 2
        assert all(p.label != "noise" for p in filtered)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default training configuration."""
        config = TrainingConfig()

        assert config.batch_size == 32
        assert config.num_epochs == 10
        assert config.learning_rate == 1e-4
        assert config.architecture == "resnet18"
        assert config.pretrained is True

    def test_custom_values(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            train_dir="./train",
            val_dir="./val",
            class_names=["a", "b", "c"],
            architecture="resnet50",
            num_epochs=50,
            batch_size=64,
        )

        assert config.train_dir == "./train"
        assert config.val_dir == "./val"
        assert len(config.class_names) == 3
        assert config.architecture == "resnet50"
        assert config.num_epochs == 50
        assert config.batch_size == 64


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = TrainingMetrics(
            epoch=5,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=0.6,
            val_accuracy=0.80,
            learning_rate=1e-4,
            epoch_time=30.5,
        )

        d = metrics.to_dict()

        assert d["epoch"] == 5
        assert d["train_loss"] == 0.5
        assert d["train_accuracy"] == 0.85
        assert d["val_loss"] == 0.6
        assert d["val_accuracy"] == 0.80
        assert d["learning_rate"] == 1e-4
        assert d["epoch_time"] == 30.5


class TestAudioDataset:
    """Tests for AudioDataset class."""

    def test_dataset_length(self, temp_dir, mock_audio_files):
        """Test dataset length matches file count."""
        dataset = AudioDataset(
            filepaths=mock_audio_files,
            sample_rate=16000,
            clip_duration=1.0,
        )

        assert len(dataset) == len(mock_audio_files)

    def test_dataset_getitem(self, temp_dir, mock_audio_files):
        """Test getting item from dataset."""
        dataset = AudioDataset(
            filepaths=mock_audio_files,
            sample_rate=16000,
            clip_duration=1.0,
        )

        waveform, filepath = dataset[0]

        assert isinstance(waveform, torch.Tensor)
        assert isinstance(filepath, str)


class TestCreateDataloader:
    """Tests for create_dataloader function."""

    def test_create_dataloader(self, temp_dir, mock_audio_files):
        """Test creating a dataloader."""
        config = ModelConfig(
            sample_rate=16000,
            clip_duration=1.0,
            batch_size=2,
            num_workers=0,
        )

        dataloader = create_dataloader(mock_audio_files, config)

        assert dataloader is not None
        assert dataloader.batch_size == 2


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def mock_audio_files(temp_dir):
    """Create mock audio files for testing."""
    import soundfile as sf

    files = []
    for i in range(3):
        filepath = temp_dir / f"test_{i}.wav"
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        sf.write(str(filepath), audio, 16000)
        files.append(str(filepath))

    return files

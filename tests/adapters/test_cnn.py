"""Tests for CNNAdapter."""

import numpy as np
import pandas as pd
import pytest
import scipy.io.wavfile as wav
import torch

from bioamla.adapters.opensoundscape import CNNAdapter


class TestCNNAdapterCreate:
    """Tests for CNNAdapter.create() factory method."""

    def test_create_default_architecture(self) -> None:
        """Test creating adapter with default resnet18."""
        adapter = CNNAdapter.create(
            classes=["bird", "frog", "insect"],
            sample_duration=3.0,
        )

        assert adapter.architecture == "resnet18"
        assert adapter.classes == ["bird", "frog", "insect"]
        assert adapter.num_classes == 3
        assert adapter.sample_duration == 3.0

    def test_create_resnet50(self) -> None:
        """Test creating adapter with resnet50 architecture."""
        adapter = CNNAdapter.create(
            classes=["class_a", "class_b"],
            architecture="resnet50",
            sample_duration=5.0,
        )

        assert adapter.architecture == "resnet50"
        assert adapter.num_classes == 2
        assert adapter.sample_duration == 5.0

    def test_create_efficientnet(self) -> None:
        """Test creating adapter with efficientnet architecture."""
        adapter = CNNAdapter.create(
            classes=["species1", "species2", "species3", "species4"],
            architecture="efficientnet_b0",
            sample_duration=2.0,
        )

        assert adapter.architecture == "efficientnet_b0"
        assert adapter.num_classes == 4

    def test_create_single_target(self) -> None:
        """Test creating single-target (single-label) classifier."""
        adapter = CNNAdapter.create(
            classes=["a", "b", "c"],
            single_target=True,
        )

        assert adapter.num_classes == 3


class TestCNNAdapterProperties:
    """Tests for CNNAdapter properties."""

    @pytest.fixture
    def adapter(self) -> CNNAdapter:
        """Create a test adapter."""
        return CNNAdapter.create(
            classes=["bird", "frog"],
            architecture="resnet18",
            sample_duration=3.0,
            sample_rate=16000,
        )

    def test_classes_property(self, adapter: CNNAdapter) -> None:
        """Test classes property returns list."""
        assert adapter.classes == ["bird", "frog"]
        assert isinstance(adapter.classes, list)

    def test_num_classes_property(self, adapter: CNNAdapter) -> None:
        """Test num_classes property."""
        assert adapter.num_classes == 2

    def test_architecture_property(self, adapter: CNNAdapter) -> None:
        """Test architecture property."""
        assert adapter.architecture == "resnet18"

    def test_sample_duration_property(self, adapter: CNNAdapter) -> None:
        """Test sample_duration property."""
        assert adapter.sample_duration == 3.0

    def test_device_property(self, adapter: CNNAdapter) -> None:
        """Test device property returns torch.device."""
        device = adapter.device
        assert isinstance(device, torch.device)

    def test_config_method(self, adapter: CNNAdapter) -> None:
        """Test config method returns dict with expected keys."""
        config = adapter.config()

        assert "architecture" in config
        assert "classes" in config
        assert "num_classes" in config
        assert "sample_duration" in config
        assert config["architecture"] == "resnet18"
        assert config["num_classes"] == 2


class TestCNNAdapterMethods:
    """Tests for CNNAdapter methods."""

    @pytest.fixture
    def adapter(self) -> CNNAdapter:
        """Create a test adapter."""
        return CNNAdapter.create(
            classes=["bird", "frog"],
            architecture="resnet18",
            sample_duration=3.0,
        )

    def test_eval_mode(self, adapter: CNNAdapter) -> None:
        """Test setting model to eval mode."""
        result = adapter.eval()

        # Should return self for chaining
        assert result is adapter

    def test_to_device(self, adapter: CNNAdapter) -> None:
        """Test moving model to device."""
        result = adapter.to("cpu")

        # Should return self for chaining
        assert result is adapter
        assert adapter.device == torch.device("cpu")

    def test_freeze_backbone(self, adapter: CNNAdapter) -> None:
        """Test freezing backbone weights."""
        # Should not raise
        adapter.freeze_backbone()

    def test_unfreeze(self, adapter: CNNAdapter) -> None:
        """Test unfreezing all weights."""
        adapter.freeze_backbone()
        adapter.unfreeze()
        # Should not raise


class TestCNNAdapterSaveLoad:
    """Tests for CNNAdapter save/load functionality."""

    @pytest.fixture
    def adapter(self) -> CNNAdapter:
        """Create a test adapter."""
        return CNNAdapter.create(
            classes=["bird", "frog", "insect"],
            architecture="resnet18",
            sample_duration=3.0,
        )

    def test_save_and_load(self, adapter: CNNAdapter, tmp_path) -> None:
        """Test saving and loading model."""
        model_path = str(tmp_path / "test_model.model")

        # Save
        saved_path = adapter.save(model_path)
        assert saved_path == model_path

        # Load
        loaded = CNNAdapter.load(model_path)

        # Architecture is inferred from network class on load (e.g., "resnet" not "resnet18")
        assert "resnet" in loaded.architecture.lower()
        assert loaded.classes == adapter.classes
        assert loaded.num_classes == adapter.num_classes
        assert loaded.sample_duration == adapter.sample_duration

    def test_load_preserves_config(self, adapter: CNNAdapter, tmp_path) -> None:
        """Test that loaded model preserves core config."""
        model_path = str(tmp_path / "config_test.model")
        adapter.save(model_path)

        loaded = CNNAdapter.load(model_path)

        # Classes and num_classes should match exactly
        assert loaded.classes == adapter.classes
        assert loaded.num_classes == adapter.num_classes
        assert loaded.sample_duration == adapter.sample_duration
        # Architecture is inferred on load - just check it's valid
        assert loaded.architecture in ["resnet", "efficientnet", "densenet", "unknown"]


class TestCNNAdapterPrediction:
    """Tests for CNNAdapter prediction functionality."""

    @pytest.fixture
    def sample_audio_path(self, tmp_path) -> str:
        """Create a temporary test audio file."""
        sample_rate = 16000
        duration = 3.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

        audio_path = tmp_path / "test_audio.wav"
        wav.write(str(audio_path), sample_rate, samples)

        return str(audio_path)

    @pytest.fixture
    def adapter(self) -> CNNAdapter:
        """Create a test adapter."""
        return CNNAdapter.create(
            classes=["bird", "frog"],
            architecture="resnet18",
            sample_duration=3.0,
            sample_rate=16000,
        )

    def test_predict_single_file(
        self,
        adapter: CNNAdapter,
        sample_audio_path: str,
    ) -> None:
        """Test prediction on single file."""
        adapter.eval()
        predictions = adapter.predict(
            sample_audio_path,
            batch_size=1,
            num_workers=0,
        )

        assert isinstance(predictions, pd.DataFrame)
        # Should have class columns
        assert "bird" in predictions.columns
        assert "frog" in predictions.columns

    def test_predict_multiple_files(
        self,
        adapter: CNNAdapter,
        sample_audio_path: str,
        tmp_path,
    ) -> None:
        """Test prediction on multiple files."""
        # Create additional test file
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = (0.5 * np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
        audio_path2 = tmp_path / "test_audio2.wav"
        wav.write(str(audio_path2), sample_rate, samples)

        adapter.eval()
        predictions = adapter.predict(
            [sample_audio_path, str(audio_path2)],
            batch_size=2,
            num_workers=0,
        )

        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) >= 2  # At least one prediction per file

    def test_predict_with_softmax(
        self,
        adapter: CNNAdapter,
        sample_audio_path: str,
    ) -> None:
        """Test prediction with softmax activation."""
        adapter.eval()
        predictions = adapter.predict(
            sample_audio_path,
            activation="softmax",
            num_workers=0,
        )

        # Softmax outputs should sum to ~1 per row
        row_sums = predictions[["bird", "frog"]].sum(axis=1)
        assert all(abs(s - 1.0) < 0.01 for s in row_sums)

    def test_predict_with_sigmoid(
        self,
        adapter: CNNAdapter,
        sample_audio_path: str,
    ) -> None:
        """Test prediction with sigmoid activation."""
        adapter.eval()
        predictions = adapter.predict(
            sample_audio_path,
            activation="sigmoid",
            num_workers=0,
        )

        assert isinstance(predictions, pd.DataFrame)
        # Sigmoid outputs are independent, don't need to sum to 1

    def test_predict_no_activation(
        self,
        adapter: CNNAdapter,
        sample_audio_path: str,
    ) -> None:
        """Test prediction with no activation (raw logits)."""
        adapter.eval()
        predictions = adapter.predict(
            sample_audio_path,
            activation=None,
            num_workers=0,
        )

        assert isinstance(predictions, pd.DataFrame)


class TestCNNAdapterEmbeddings:
    """Tests for CNNAdapter embedding extraction."""

    @pytest.fixture
    def sample_audio_path(self, tmp_path) -> str:
        """Create a temporary test audio file."""
        sample_rate = 16000
        duration = 3.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

        audio_path = tmp_path / "test_audio.wav"
        wav.write(str(audio_path), sample_rate, samples)

        return str(audio_path)

    @pytest.fixture
    def adapter(self) -> CNNAdapter:
        """Create a test adapter."""
        return CNNAdapter.create(
            classes=["bird", "frog"],
            architecture="resnet18",
            sample_duration=3.0,
            sample_rate=16000,
        )

    def test_extract_embeddings(
        self,
        adapter: CNNAdapter,
        sample_audio_path: str,
    ) -> None:
        """Test embedding extraction."""
        adapter.eval()
        embeddings = adapter.extract_embeddings(
            sample_audio_path,
            num_workers=0,
        )

        assert isinstance(embeddings, np.ndarray)
        # ResNet18 produces 512-dimensional embeddings
        assert embeddings.shape[-1] == 512

    def test_extract_embeddings_multiple_files(
        self,
        adapter: CNNAdapter,
        sample_audio_path: str,
        tmp_path,
    ) -> None:
        """Test embedding extraction from multiple files."""
        # Create additional test file
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = (0.5 * np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
        audio_path2 = tmp_path / "test_audio2.wav"
        wav.write(str(audio_path2), sample_rate, samples)

        adapter.eval()
        embeddings = adapter.extract_embeddings(
            [sample_audio_path, str(audio_path2)],
            num_workers=0,
        )

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim >= 1


class TestCNNAdapterIntegration:
    """Integration tests for CNNAdapter workflow."""

    @pytest.fixture
    def sample_audio_paths(self, tmp_path) -> list[str]:
        """Create multiple test audio files."""
        sample_rate = 16000
        duration = 3.0
        paths = []

        for i, freq in enumerate([440, 880, 1320]):
            t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
            samples = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            path = tmp_path / f"audio_{i}.wav"
            wav.write(str(path), sample_rate, samples)
            paths.append(str(path))

        return paths

    def test_inference_workflow(
        self,
        sample_audio_paths: list[str],
    ) -> None:
        """Test typical inference workflow."""
        # Create model
        adapter = CNNAdapter.create(
            classes=["species_a", "species_b", "species_c"],
            architecture="resnet18",
            sample_duration=3.0,
        )

        # Set to eval mode
        adapter.eval()

        # Run predictions
        predictions = adapter.predict(
            sample_audio_paths,
            batch_size=2,
            num_workers=0,
            activation="softmax",
        )

        assert isinstance(predictions, pd.DataFrame)
        assert set(predictions.columns) >= {"species_a", "species_b", "species_c"}

    def test_embedding_workflow(
        self,
        sample_audio_paths: list[str],
    ) -> None:
        """Test typical embedding extraction workflow."""
        adapter = CNNAdapter.create(
            classes=["a", "b"],
            architecture="resnet18",
            sample_duration=3.0,
        )

        adapter.eval()

        # Extract embeddings
        embeddings = adapter.extract_embeddings(
            sample_audio_paths,
            num_workers=0,
        )

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[-1] == 512  # ResNet18 embedding dim

    def test_save_load_predict_workflow(
        self,
        sample_audio_paths: list[str],
        tmp_path,
    ) -> None:
        """Test save, load, then predict workflow."""
        # Create and save
        adapter = CNNAdapter.create(
            classes=["bird", "frog"],
            architecture="resnet18",
            sample_duration=3.0,
        )
        model_path = str(tmp_path / "workflow_model.model")
        adapter.save(model_path)

        # Load and predict
        loaded = CNNAdapter.load(model_path)
        loaded.eval()
        predictions = loaded.predict(
            sample_audio_paths[0],
            num_workers=0,
        )

        assert isinstance(predictions, pd.DataFrame)
        assert "bird" in predictions.columns
        assert "frog" in predictions.columns

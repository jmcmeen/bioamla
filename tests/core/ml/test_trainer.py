"""Tests for model training pipeline with BioamlaPreprocessor integration."""

import numpy as np
import pytest
import scipy.io.wavfile as wav
import torch
from pathlib import Path

from bioamla.core.ml.trainer import (
    SpectrogramDataset,
    TrainingConfig,
    ModelTrainer,
    SpectrogramPreprocessorProtocol,
)


class TestSpectrogramDataset:
    """Tests for SpectrogramDataset class."""

    @pytest.fixture
    def sample_data_dir(self, tmp_path) -> Path:
        """Create a temporary directory with sample audio files organized by class."""
        sample_rate = 16000
        duration = 3.0

        # Create class directories
        for class_name in ["class_a", "class_b"]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()

            # Create 2 audio files per class
            for i in range(2):
                frequency = 440 if class_name == "class_a" else 880
                t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
                samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

                audio_path = class_dir / f"audio_{i}.wav"
                wav.write(str(audio_path), sample_rate, samples)

        return tmp_path

    def test_dataset_without_preprocessor(self, sample_data_dir: Path) -> None:
        """Test dataset with built-in torchaudio processing."""
        dataset = SpectrogramDataset(
            data_dir=str(sample_data_dir),
            class_names=["class_a", "class_b"],
            sample_rate=16000,
            clip_duration=3.0,
        )

        assert len(dataset) == 4  # 2 files per class * 2 classes

        spectrogram, label = dataset[0]

        assert isinstance(spectrogram, torch.Tensor)
        assert spectrogram.dim() == 2  # (height, width)
        assert label in [0, 1]

    def test_dataset_with_preprocessor(self, sample_data_dir: Path) -> None:
        """Test dataset with BioamlaPreprocessor."""
        from bioamla.adapters.opensoundscape import BioamlaPreprocessor

        preprocessor = BioamlaPreprocessor(
            sample_duration=3.0,
            sample_rate=16000,
            height=224,
            width=224,
        )

        dataset = SpectrogramDataset(
            data_dir=str(sample_data_dir),
            class_names=["class_a", "class_b"],
            sample_rate=16000,
            clip_duration=3.0,
            preprocessor=preprocessor,
        )

        assert len(dataset) == 4

        spectrogram, label = dataset[0]

        assert isinstance(spectrogram, torch.Tensor)
        assert spectrogram.dim() == 2
        assert spectrogram.shape == (224, 224)
        assert label in [0, 1]

    def test_dataset_with_augmented_preprocessor(self, sample_data_dir: Path) -> None:
        """Test dataset with BioamlaPreprocessor and augmentation enabled."""
        from bioamla.adapters.opensoundscape import (
            AugmentationConfig,
            BioamlaPreprocessor,
        )

        preprocessor = BioamlaPreprocessor(
            sample_duration=3.0,
            sample_rate=16000,
            height=224,
            width=224,
        )

        aug_config = AugmentationConfig(
            time_mask=True,
            frequency_mask=True,
        )
        preprocessor.enable_augmentation(aug_config)

        dataset = SpectrogramDataset(
            data_dir=str(sample_data_dir),
            class_names=["class_a", "class_b"],
            sample_rate=16000,
            clip_duration=3.0,
            preprocessor=preprocessor,
            augment=True,
        )

        spectrogram, label = dataset[0]

        assert isinstance(spectrogram, torch.Tensor)
        assert spectrogram.shape == (224, 224)

    def test_dataloader_compatibility(self, sample_data_dir: Path) -> None:
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        from bioamla.adapters.opensoundscape import BioamlaPreprocessor

        preprocessor = BioamlaPreprocessor(
            sample_duration=3.0,
            sample_rate=16000,
            height=128,
            width=128,
        )

        dataset = SpectrogramDataset(
            data_dir=str(sample_data_dir),
            class_names=["class_a", "class_b"],
            sample_rate=16000,
            clip_duration=3.0,
            preprocessor=preprocessor,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )

        batch = next(iter(dataloader))
        spectrograms, labels = batch

        assert spectrograms.shape == (2, 128, 128)
        assert labels.shape == (2,)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.use_oss_preprocessor is False
        assert config.sample_rate == 16000
        assert config.clip_duration == 3.0
        assert config.augment is True

    def test_oss_preprocessor_config(self) -> None:
        """Test configuration with OSS preprocessor enabled."""
        config = TrainingConfig(
            use_oss_preprocessor=True,
            sample_rate=22050,
            clip_duration=5.0,
        )

        assert config.use_oss_preprocessor is True
        assert config.sample_rate == 22050
        assert config.clip_duration == 5.0


class TestModelTrainerPreprocessor:
    """Tests for ModelTrainer preprocessor integration."""

    @pytest.fixture
    def sample_data_dir(self, tmp_path) -> Path:
        """Create a temporary directory with sample audio files."""
        sample_rate = 16000
        duration = 3.0

        for class_name in ["class_a", "class_b"]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()

            for i in range(2):
                frequency = 440 + i * 110
                t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
                samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

                audio_path = class_dir / f"audio_{i}.wav"
                wav.write(str(audio_path), sample_rate, samples)

        return tmp_path

    def test_create_preprocessors(self, sample_data_dir: Path) -> None:
        """Test that _create_preprocessors creates correct preprocessors."""
        config = TrainingConfig(
            train_dir=str(sample_data_dir),
            class_names=["class_a", "class_b"],
            use_oss_preprocessor=True,
            augment=True,
        )

        trainer = ModelTrainer(config)
        train_pp, val_pp = trainer._create_preprocessors()

        # Both should be BioamlaPreprocessor instances
        from bioamla.adapters.opensoundscape import BioamlaPreprocessor
        assert isinstance(train_pp, BioamlaPreprocessor)
        assert isinstance(val_pp, BioamlaPreprocessor)

        # Train should have augmentation enabled, val should not
        assert train_pp.augmentation_enabled is True
        assert val_pp.augmentation_enabled is False

    def test_trainer_setup_with_oss_preprocessor(self, sample_data_dir: Path, tmp_path) -> None:
        """Test trainer setup with OSS preprocessor."""
        config = TrainingConfig(
            train_dir=str(sample_data_dir),
            output_dir=str(tmp_path / "output"),
            class_names=["class_a", "class_b"],
            use_oss_preprocessor=True,
            augment=True,
            num_workers=0,  # Avoid multiprocessing issues in tests
        )

        trainer = ModelTrainer(config)
        trainer.setup()

        # Verify dataset uses preprocessor
        assert trainer.train_loader is not None
        dataset = trainer.train_loader.dataset
        assert dataset.preprocessor is not None

    def test_trainer_setup_without_oss_preprocessor(self, sample_data_dir: Path, tmp_path) -> None:
        """Test trainer setup with default torchaudio processing."""
        config = TrainingConfig(
            train_dir=str(sample_data_dir),
            output_dir=str(tmp_path / "output"),
            class_names=["class_a", "class_b"],
            use_oss_preprocessor=False,
            num_workers=0,
        )

        trainer = ModelTrainer(config)
        trainer.setup()

        # Verify dataset does not use preprocessor
        dataset = trainer.train_loader.dataset
        assert dataset.preprocessor is None


class TestSpectrogramPreprocessorProtocol:
    """Tests for the preprocessor protocol."""

    def test_bioamla_preprocessor_implements_protocol(self) -> None:
        """Verify BioamlaPreprocessor satisfies the protocol."""
        from bioamla.adapters.opensoundscape import BioamlaPreprocessor

        preprocessor = BioamlaPreprocessor()

        # Check it has the required methods
        assert hasattr(preprocessor, "process_file")
        assert hasattr(preprocessor, "to_tensor")
        assert callable(preprocessor.process_file)
        assert callable(preprocessor.to_tensor)

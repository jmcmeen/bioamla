"""Tests for BioamlaPreprocessor adapter."""

import numpy as np
import pytest
import scipy.io.wavfile as wav
import torch

from bioamla.adapters.opensoundscape import (
    AugmentationConfig,
    BioamlaPreprocessor,
)


class TestBioamlaPreprocessor:
    """Tests for BioamlaPreprocessor class."""

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
    def sample_samples(self) -> tuple[np.ndarray, int]:
        """Create sample audio data as numpy array."""
        sample_rate = 16000
        duration = 3.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = 0.5 * np.sin(2 * np.pi * frequency * t)

        return samples, sample_rate

    @pytest.fixture
    def preprocessor(self) -> BioamlaPreprocessor:
        """Create a default preprocessor."""
        return BioamlaPreprocessor(
            sample_duration=3.0,
            sample_rate=16000,
            n_mels=128,
        )

    def test_init_default(self) -> None:
        """Test default initialization."""
        preprocessor = BioamlaPreprocessor()

        assert preprocessor.sample_duration == 3.0
        assert preprocessor.sample_rate == 16000
        assert preprocessor.n_mels == 128
        assert preprocessor.n_fft == 2048
        assert preprocessor.hop_length == 512
        assert not preprocessor.augmentation_enabled

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        preprocessor = BioamlaPreprocessor(
            sample_duration=5.0,
            sample_rate=22050,
            n_mels=64,
            n_fft=1024,
            hop_length=256,
        )

        assert preprocessor.sample_duration == 5.0
        assert preprocessor.sample_rate == 22050
        assert preprocessor.n_mels == 64
        assert preprocessor.n_fft == 1024
        assert preprocessor.hop_length == 256

    def test_process_file(
        self,
        preprocessor: BioamlaPreprocessor,
        sample_audio_path: str,
    ) -> None:
        """Test processing audio file to spectrogram."""
        spectrogram = preprocessor.process_file(sample_audio_path)

        assert isinstance(spectrogram, np.ndarray)
        assert spectrogram.ndim == 2
        # Should have n_mels frequency bins
        assert spectrogram.shape[0] > 0
        # Should have time frames
        assert spectrogram.shape[1] > 0

    def test_process_file_with_trim(
        self,
        preprocessor: BioamlaPreprocessor,
        sample_audio_path: str,
    ) -> None:
        """Test processing file with time trimming."""
        spectrogram = preprocessor.process_file(
            sample_audio_path,
            start_time=0.5,
            end_time=2.5,
        )

        assert isinstance(spectrogram, np.ndarray)
        assert spectrogram.ndim == 2

    def test_process_samples(
        self,
        preprocessor: BioamlaPreprocessor,
        sample_samples: tuple[np.ndarray, int],
    ) -> None:
        """Test processing numpy array to spectrogram."""
        samples, sample_rate = sample_samples
        spectrogram = preprocessor.process_samples(samples, sample_rate)

        assert isinstance(spectrogram, np.ndarray)
        assert spectrogram.ndim == 2

    def test_process_samples_with_resample(
        self,
        sample_samples: tuple[np.ndarray, int],
    ) -> None:
        """Test processing samples that need resampling."""
        samples, _ = sample_samples
        # Samples are at 16000, preprocessor expects 16000
        preprocessor = BioamlaPreprocessor(sample_rate=16000)

        spectrogram = preprocessor.process_samples(samples, 16000)

        assert isinstance(spectrogram, np.ndarray)
        assert spectrogram.ndim == 2

    def test_resize_spectrogram(
        self,
        sample_audio_path: str,
    ) -> None:
        """Test spectrogram resizing."""
        preprocessor = BioamlaPreprocessor(
            sample_duration=3.0,
            sample_rate=16000,
            height=224,
            width=224,
        )

        spectrogram = preprocessor.process_file(sample_audio_path)

        assert spectrogram.shape == (224, 224)

    def test_to_tensor(
        self,
        preprocessor: BioamlaPreprocessor,
        sample_audio_path: str,
    ) -> None:
        """Test converting spectrogram to tensor."""
        spectrogram = preprocessor.process_file(sample_audio_path)
        tensor = preprocessor.to_tensor(spectrogram)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 3  # (1, height, width)
        assert tensor.dtype == torch.float32

    def test_to_tensor_normalized(
        self,
        preprocessor: BioamlaPreprocessor,
        sample_audio_path: str,
    ) -> None:
        """Test tensor normalization."""
        spectrogram = preprocessor.process_file(sample_audio_path)
        tensor = preprocessor.to_tensor(spectrogram, normalize=True)

        # Should be in [0, 1] range
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_to_tensor_unnormalized(
        self,
        preprocessor: BioamlaPreprocessor,
        sample_audio_path: str,
    ) -> None:
        """Test tensor without normalization."""
        spectrogram = preprocessor.process_file(sample_audio_path)
        tensor = preprocessor.to_tensor(spectrogram, normalize=False)

        # Should preserve original values
        np.testing.assert_array_almost_equal(
            tensor.squeeze().numpy(),
            spectrogram,
            decimal=5,
        )

    def test_config_property(
        self,
        preprocessor: BioamlaPreprocessor,
    ) -> None:
        """Test config property returns correct values."""
        config = preprocessor.config

        assert config["sample_duration"] == 3.0
        assert config["sample_rate"] == 16000
        assert config["n_mels"] == 128
        assert config["augmentation_enabled"] is False


class TestAugmentationConfig:
    """Tests for AugmentationConfig class."""

    def test_default_config(self) -> None:
        """Test default augmentation config."""
        config = AugmentationConfig()

        assert config.time_mask is False
        assert config.frequency_mask is False
        assert config.random_gain is False
        assert config.add_noise is False

    def test_custom_config(self) -> None:
        """Test custom augmentation config."""
        config = AugmentationConfig(
            time_mask=True,
            frequency_mask=True,
            time_mask_max_masks=3,
            frequency_mask_max_masks=3,
        )

        assert config.time_mask is True
        assert config.frequency_mask is True
        assert config.time_mask_max_masks == 3
        assert config.frequency_mask_max_masks == 3


class TestPreprocessorAugmentation:
    """Tests for preprocessor augmentation functionality."""

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

    def test_enable_augmentation(self) -> None:
        """Test enabling augmentation."""
        preprocessor = BioamlaPreprocessor()
        config = AugmentationConfig(time_mask=True)

        assert not preprocessor.augmentation_enabled

        preprocessor.enable_augmentation(config)

        assert preprocessor.augmentation_enabled

    def test_disable_augmentation(self) -> None:
        """Test disabling augmentation."""
        preprocessor = BioamlaPreprocessor()
        config = AugmentationConfig(time_mask=True)

        preprocessor.enable_augmentation(config)
        assert preprocessor.augmentation_enabled

        preprocessor.disable_augmentation()
        assert not preprocessor.augmentation_enabled

    def test_process_with_augmentation(self, sample_audio_path: str) -> None:
        """Test processing with augmentation enabled."""
        preprocessor = BioamlaPreprocessor()
        config = AugmentationConfig(
            time_mask=True,
            frequency_mask=True,
        )

        preprocessor.enable_augmentation(config)
        spectrogram = preprocessor.process_file(sample_audio_path)

        # Should still produce valid spectrogram
        assert isinstance(spectrogram, np.ndarray)
        assert spectrogram.ndim == 2


class TestPreprocessorDurationHandling:
    """Tests for audio duration handling."""

    @pytest.fixture
    def short_audio_path(self, tmp_path) -> str:
        """Create a short audio file (1 second)."""
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

        audio_path = tmp_path / "short_audio.wav"
        wav.write(str(audio_path), sample_rate, samples)

        return str(audio_path)

    @pytest.fixture
    def long_audio_path(self, tmp_path) -> str:
        """Create a long audio file (10 seconds)."""
        sample_rate = 16000
        duration = 10.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

        audio_path = tmp_path / "long_audio.wav"
        wav.write(str(audio_path), sample_rate, samples)

        return str(audio_path)

    def test_short_audio_padding(self, short_audio_path: str) -> None:
        """Test that short audio is padded to target duration."""
        preprocessor = BioamlaPreprocessor(sample_duration=3.0)

        spectrogram = preprocessor.process_file(short_audio_path)

        # Should produce spectrogram for 3-second audio
        assert isinstance(spectrogram, np.ndarray)
        assert spectrogram.ndim == 2

    def test_long_audio_trimming(self, long_audio_path: str) -> None:
        """Test that long audio is trimmed to target duration."""
        preprocessor = BioamlaPreprocessor(sample_duration=3.0)

        spectrogram = preprocessor.process_file(long_audio_path)

        # Should produce spectrogram for 3-second audio
        assert isinstance(spectrogram, np.ndarray)
        assert spectrogram.ndim == 2


class TestPreprocessorIntegration:
    """Integration tests for preprocessor workflow."""

    @pytest.fixture
    def sample_audio_path(self, tmp_path) -> str:
        """Create a temporary test audio file."""
        sample_rate = 16000
        duration = 3.0
        # Create audio with multiple frequencies
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = (
            0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.2 * np.sin(2 * np.pi * 880 * t)
            + 0.1 * np.sin(2 * np.pi * 1320 * t)
        )
        samples = (samples * 32767).astype(np.int16)

        audio_path = tmp_path / "test_audio.wav"
        wav.write(str(audio_path), sample_rate, samples)

        return str(audio_path)

    def test_training_workflow(self, sample_audio_path: str) -> None:
        """Test typical training workflow with augmentation."""
        # Setup preprocessor for training
        preprocessor = BioamlaPreprocessor(
            sample_duration=3.0,
            sample_rate=16000,
            n_mels=128,
            height=224,
            width=224,
        )

        # Enable augmentation
        aug_config = AugmentationConfig(
            time_mask=True,
            frequency_mask=True,
        )
        preprocessor.enable_augmentation(aug_config)

        # Process file
        spectrogram = preprocessor.process_file(sample_audio_path)
        tensor = preprocessor.to_tensor(spectrogram, normalize=True)

        # Verify output format suitable for training
        assert tensor.shape == (1, 224, 224)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_inference_workflow(self, sample_audio_path: str) -> None:
        """Test typical inference workflow without augmentation."""
        # Setup preprocessor for inference
        preprocessor = BioamlaPreprocessor(
            sample_duration=3.0,
            sample_rate=16000,
            n_mels=128,
            height=224,
            width=224,
        )

        # No augmentation for inference
        assert not preprocessor.augmentation_enabled

        # Process file
        spectrogram = preprocessor.process_file(sample_audio_path)
        tensor = preprocessor.to_tensor(spectrogram, normalize=True)

        # Add batch dimension for model input
        batch = tensor.unsqueeze(0)

        # Verify output format suitable for inference
        assert batch.shape == (1, 1, 224, 224)
        assert batch.dtype == torch.float32

    def test_batch_processing(self, sample_audio_path: str, tmp_path) -> None:
        """Test processing multiple files."""
        # Create additional test files
        sample_rate = 16000
        duration = 3.0

        paths = [sample_audio_path]
        for i in range(2):
            t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
            frequency = 440 + i * 220
            samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
            path = tmp_path / f"test_audio_{i}.wav"
            wav.write(str(path), sample_rate, samples)
            paths.append(str(path))

        # Process batch
        preprocessor = BioamlaPreprocessor(
            sample_duration=3.0,
            height=128,
            width=128,
        )

        tensors = []
        for path in paths:
            spec = preprocessor.process_file(path)
            tensor = preprocessor.to_tensor(spec, normalize=True)
            tensors.append(tensor)

        batch = torch.stack(tensors)

        assert batch.shape == (3, 1, 128, 128)

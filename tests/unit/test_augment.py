"""
Unit tests for bioamla.core.augment module.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from bioamla.core.augment import (
    AugmentationConfig,
    augment_audio,
    augment_file,
    batch_augment,
    create_augmentation_pipeline,
)


def _create_mock_wav(path: Path) -> None:
    """Create a minimal WAV file for testing."""
    sample_rate = 16000
    duration = 1
    num_samples = sample_rate * duration
    bits_per_sample = 16
    num_channels = 1

    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    file_size = 36 + data_size

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", file_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        # Write some actual audio data (sine wave)
        for i in range(num_samples):
            sample = int(32767 * np.sin(2 * np.pi * 440 * i / sample_rate))
            f.write(struct.pack("<h", sample))


class TestAugmentationConfig:
    """Tests for AugmentationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AugmentationConfig()

        assert config.add_noise is False
        assert config.time_stretch is False
        assert config.pitch_shift is False
        assert config.gain is False
        assert config.sample_rate == 16000
        assert config.multiply == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AugmentationConfig(
            add_noise=True,
            noise_min_snr=5.0,
            noise_max_snr=20.0,
            time_stretch=True,
            pitch_shift=True,
            multiply=5,
        )

        assert config.add_noise is True
        assert config.noise_min_snr == 5.0
        assert config.time_stretch is True
        assert config.pitch_shift is True
        assert config.multiply == 5


class TestCreateAugmentationPipeline:
    """Tests for create_augmentation_pipeline function."""

    def test_returns_none_when_no_augmentations(self):
        """Test that None is returned when no augmentations are enabled."""
        config = AugmentationConfig()
        pipeline = create_augmentation_pipeline(config)

        assert pipeline is None

    def test_creates_pipeline_with_noise(self):
        """Test creating pipeline with noise augmentation."""
        config = AugmentationConfig(add_noise=True)
        pipeline = create_augmentation_pipeline(config)

        assert pipeline is not None

    def test_creates_pipeline_with_time_stretch(self):
        """Test creating pipeline with time stretch augmentation."""
        config = AugmentationConfig(time_stretch=True)
        pipeline = create_augmentation_pipeline(config)

        assert pipeline is not None

    def test_creates_pipeline_with_pitch_shift(self):
        """Test creating pipeline with pitch shift augmentation."""
        config = AugmentationConfig(pitch_shift=True)
        pipeline = create_augmentation_pipeline(config)

        assert pipeline is not None

    def test_creates_pipeline_with_gain(self):
        """Test creating pipeline with gain augmentation."""
        config = AugmentationConfig(gain=True)
        pipeline = create_augmentation_pipeline(config)

        assert pipeline is not None

    def test_creates_pipeline_with_multiple_augmentations(self):
        """Test creating pipeline with multiple augmentations."""
        config = AugmentationConfig(
            add_noise=True,
            time_stretch=True,
            pitch_shift=True,
            gain=True,
        )
        pipeline = create_augmentation_pipeline(config)

        assert pipeline is not None


class TestAugmentAudio:
    """Tests for augment_audio function."""

    def test_augments_audio(self):
        """Test augmenting audio data."""
        config = AugmentationConfig(add_noise=True)
        pipeline = create_augmentation_pipeline(config)

        # Create test audio
        sample_rate = 16000
        duration = 1.0
        audio = np.sin(2 * np.pi * 440 * np.arange(int(sample_rate * duration)) / sample_rate)
        audio = audio.astype(np.float32)

        augmented = augment_audio(audio, sample_rate, pipeline)

        assert isinstance(augmented, np.ndarray)
        assert len(augmented) > 0

    def test_converts_dtype(self):
        """Test that audio is converted to float32."""
        config = AugmentationConfig(gain=True)
        pipeline = create_augmentation_pipeline(config)

        # Create audio as float64
        audio = np.zeros(16000, dtype=np.float64)

        augmented = augment_audio(audio, 16000, pipeline)

        # Should work without error
        assert isinstance(augmented, np.ndarray)


class TestAugmentFile:
    """Tests for augment_file function."""

    def test_augments_file(self, temp_dir):
        """Test augmenting a single file."""
        input_file = temp_dir / "input.wav"
        output_file = temp_dir / "output.wav"
        _create_mock_wav(input_file)

        config = AugmentationConfig(add_noise=True)

        result = augment_file(str(input_file), str(output_file), config)

        assert result == str(output_file)
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_raises_on_missing_file(self, temp_dir):
        """Test that FileNotFoundError is raised for missing file."""
        config = AugmentationConfig(add_noise=True)

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            augment_file(
                "/nonexistent/audio.wav",
                str(temp_dir / "output.wav"),
                config,
            )

    def test_raises_on_no_augmentations(self, temp_dir):
        """Test that ValueError is raised when no augmentations configured."""
        input_file = temp_dir / "input.wav"
        output_file = temp_dir / "output.wav"
        _create_mock_wav(input_file)

        config = AugmentationConfig()  # No augmentations enabled

        with pytest.raises(ValueError, match="No augmentations configured"):
            augment_file(str(input_file), str(output_file), config)

    def test_creates_output_directory(self, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        input_file = temp_dir / "input.wav"
        output_file = temp_dir / "nested" / "dir" / "output.wav"
        _create_mock_wav(input_file)

        config = AugmentationConfig(add_noise=True)

        result = augment_file(str(input_file), str(output_file), config)

        assert output_file.exists()


class TestBatchAugment:
    """Tests for batch_augment function."""

    def test_processes_directory(self, temp_dir):
        """Test batch processing of a directory."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        # Create multiple audio files
        for i in range(3):
            _create_mock_wav(input_dir / f"audio_{i}.wav")

        config = AugmentationConfig(add_noise=True, multiply=1)

        result = batch_augment(
            str(input_dir),
            str(output_dir),
            config,
            verbose=False,
        )

        assert result["files_processed"] == 3
        assert result["files_created"] == 3
        assert result["files_failed"] == 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.wav"))) == 3

    def test_multiply_creates_multiple_copies(self, temp_dir):
        """Test that multiply option creates multiple augmented copies."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "test.wav")

        config = AugmentationConfig(add_noise=True, multiply=5)

        result = batch_augment(
            str(input_dir),
            str(output_dir),
            config,
            verbose=False,
        )

        assert result["files_processed"] == 1
        assert result["files_created"] == 5
        assert len(list(output_dir.glob("*.wav"))) == 5

    def test_handles_empty_directory(self, temp_dir):
        """Test handling of empty directory."""
        input_dir = temp_dir / "empty"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        config = AugmentationConfig(add_noise=True)

        result = batch_augment(
            str(input_dir),
            str(output_dir),
            config,
            verbose=False,
        )

        assert result["files_processed"] == 0
        assert result["files_created"] == 0

    def test_raises_on_missing_directory(self, temp_dir):
        """Test that FileNotFoundError is raised for missing directory."""
        config = AugmentationConfig(add_noise=True)

        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            batch_augment(
                "/nonexistent/dir",
                str(temp_dir / "output"),
                config,
            )

    def test_raises_on_no_augmentations(self, temp_dir):
        """Test that ValueError is raised when no augmentations configured."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        _create_mock_wav(input_dir / "test.wav")

        config = AugmentationConfig()  # No augmentations enabled

        with pytest.raises(ValueError, match="No augmentations configured"):
            batch_augment(
                str(input_dir),
                str(temp_dir / "output"),
                config,
            )

    def test_preserves_directory_structure(self, temp_dir):
        """Test that subdirectory structure is preserved."""
        input_dir = temp_dir / "input"
        subdir = input_dir / "subdir"
        subdir.mkdir(parents=True)
        output_dir = temp_dir / "output"

        _create_mock_wav(subdir / "nested_audio.wav")

        config = AugmentationConfig(add_noise=True)

        result = batch_augment(
            str(input_dir),
            str(output_dir),
            config,
            recursive=True,
            verbose=False,
        )

        assert result["files_processed"] == 1
        expected_output = output_dir / "subdir" / "nested_audio_aug.wav"
        assert expected_output.exists()

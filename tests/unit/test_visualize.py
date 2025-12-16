"""
Unit tests for bioamla.core.visualize module.
"""

from pathlib import Path

import pytest

from bioamla.visualize import (
    batch_generate_spectrograms,
    generate_spectrogram,
)


class TestGenerateSpectrogram:
    """Tests for generate_spectrogram function."""

    def test_generates_mel_spectrogram(self, mock_audio_file, temp_dir):
        """Test generating a mel spectrogram."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="mel",
        )

        assert result == str(output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_generates_mfcc(self, mock_audio_file, temp_dir):
        """Test generating MFCC visualization."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="mfcc",
        )

        assert result == str(output_path)
        assert output_path.exists()

    def test_generates_waveform(self, mock_audio_file, temp_dir):
        """Test generating waveform visualization."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="waveform",
        )

        assert result == str(output_path)
        assert output_path.exists()

    def test_raises_on_missing_file(self, temp_dir):
        """Test that FileNotFoundError is raised for missing audio."""
        output_path = temp_dir / "output.png"

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            generate_spectrogram(
                audio_path="/nonexistent/audio.wav",
                output_path=str(output_path),
            )

    def test_raises_on_invalid_viz_type(self, mock_audio_file, temp_dir):
        """Test that ValueError is raised for invalid visualization type."""
        output_path = temp_dir / "output.png"

        with pytest.raises(ValueError, match="Invalid visualization type"):
            generate_spectrogram(
                audio_path=str(mock_audio_file),
                output_path=str(output_path),
                viz_type="invalid",
            )

    def test_creates_output_directory(self, mock_audio_file, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        output_path = temp_dir / "nested" / "dir" / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
        )

        assert output_path.exists()

    def test_custom_sample_rate(self, mock_audio_file, temp_dir):
        """Test generating spectrogram with custom sample rate."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            sample_rate=8000,
        )

        assert output_path.exists()

    def test_custom_figsize(self, mock_audio_file, temp_dir):
        """Test generating spectrogram with custom figure size."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            figsize=(12, 6),
        )

        assert output_path.exists()

    def test_custom_title(self, mock_audio_file, temp_dir):
        """Test generating spectrogram with custom title."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            title="Custom Title",
        )

        assert output_path.exists()


class TestBatchGenerateSpectrograms:
    """Tests for batch_generate_spectrograms function."""

    def test_processes_directory(self, temp_dir):
        """Test batch processing of a directory."""
        # Create input directory with mock audio files
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        # Create mock audio files using the same structure as conftest

        for i in range(3):
            audio_path = input_dir / f"test_{i}.wav"
            _create_mock_wav(audio_path)

        result = batch_generate_spectrograms(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            verbose=False,
        )

        assert result["files_processed"] == 3
        assert result["files_failed"] == 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.png"))) == 3

    def test_handles_empty_directory(self, temp_dir):
        """Test handling of empty directory."""
        input_dir = temp_dir / "empty"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        result = batch_generate_spectrograms(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            verbose=False,
        )

        assert result["files_processed"] == 0
        assert result["files_failed"] == 0

    def test_raises_on_missing_directory(self, temp_dir):
        """Test that FileNotFoundError is raised for missing directory."""
        output_dir = temp_dir / "output"

        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            batch_generate_spectrograms(
                input_dir="/nonexistent/dir",
                output_dir=str(output_dir),
            )

    def test_preserves_directory_structure(self, temp_dir):
        """Test that subdirectory structure is preserved."""
        input_dir = temp_dir / "input"
        subdir = input_dir / "subdir"
        subdir.mkdir(parents=True)
        output_dir = temp_dir / "output"

        # Create audio file in subdirectory
        audio_path = subdir / "nested_audio.wav"
        _create_mock_wav(audio_path)

        result = batch_generate_spectrograms(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            recursive=True,
            verbose=False,
        )

        assert result["files_processed"] == 1
        # Check that output preserves structure
        expected_output = output_dir / "subdir" / "nested_audio.png"
        assert expected_output.exists()

    def test_different_viz_types(self, temp_dir):
        """Test batch processing with different visualization types."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        audio_path = input_dir / "test.wav"
        _create_mock_wav(audio_path)

        for viz_type in ["mel", "mfcc", "waveform"]:
            output_dir = temp_dir / f"output_{viz_type}"

            result = batch_generate_spectrograms(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                viz_type=viz_type,
                verbose=False,
            )

            assert result["files_processed"] == 1

    def test_handles_failed_files(self, temp_dir):
        """Test handling of files that fail to process."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        # Create a valid audio file
        valid_audio = input_dir / "valid.wav"
        _create_mock_wav(valid_audio)

        # Create an invalid "audio" file (not actually audio)
        invalid_audio = input_dir / "invalid.wav"
        invalid_audio.write_bytes(b"not audio data")

        result = batch_generate_spectrograms(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            verbose=False,
        )

        assert result["files_processed"] == 1
        assert result["files_failed"] == 1


def _create_mock_wav(path: Path) -> None:
    """Create a minimal WAV file for testing."""
    import struct

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
        f.write(b"\x00" * data_size)

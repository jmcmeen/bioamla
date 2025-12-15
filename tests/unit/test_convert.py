"""
Unit tests for bioamla.core.datasets audio conversion functions.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from bioamla.core.datasets import (
    batch_convert_audio,
    convert_audio_file,
)


def _create_mock_wav(path: Path, duration: float = 1.0, freq: float = 440.0) -> None:
    """Create a WAV file with a sine wave for testing."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
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
        for i in range(num_samples):
            sample = int(16000 * np.sin(2 * np.pi * freq * i / sample_rate))
            f.write(struct.pack("<h", sample))


class TestConvertAudioFile:
    """Tests for convert_audio_file function."""

    def test_converts_wav_to_mp3(self, temp_dir):
        """Test converting WAV to MP3."""
        input_path = temp_dir / "test.wav"
        output_path = temp_dir / "test.mp3"
        _create_mock_wav(input_path)

        result = convert_audio_file(str(input_path), str(output_path))

        assert result == str(output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_converts_wav_to_flac(self, temp_dir):
        """Test converting WAV to FLAC."""
        input_path = temp_dir / "test.wav"
        output_path = temp_dir / "test.flac"
        _create_mock_wav(input_path)

        result = convert_audio_file(str(input_path), str(output_path))

        assert result == str(output_path)
        assert output_path.exists()

    @pytest.mark.skip(reason="OGG conversion may fail with mono audio in some ffmpeg configs")
    def test_converts_wav_to_ogg(self, temp_dir):
        """Test converting WAV to OGG."""
        input_path = temp_dir / "test.wav"
        output_path = temp_dir / "test.ogg"
        _create_mock_wav(input_path)

        result = convert_audio_file(str(input_path), str(output_path))

        assert result == str(output_path)
        assert output_path.exists()

    def test_infers_format_from_output_path(self, temp_dir):
        """Test that format is inferred from output path."""
        input_path = temp_dir / "test.wav"
        output_path = temp_dir / "test.mp3"
        _create_mock_wav(input_path)

        result = convert_audio_file(str(input_path), str(output_path))

        assert output_path.exists()

    def test_uses_explicit_target_format(self, temp_dir):
        """Test using explicit target_format parameter."""
        input_path = temp_dir / "test.wav"
        output_path = temp_dir / "test.audio"  # Unusual extension
        _create_mock_wav(input_path)

        result = convert_audio_file(str(input_path), str(output_path), target_format="mp3")

        # The file should be created even with unusual extension
        assert Path(result).exists()

    def test_creates_output_directory(self, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        input_path = temp_dir / "test.wav"
        output_path = temp_dir / "nested" / "dir" / "test.mp3"
        _create_mock_wav(input_path)

        result = convert_audio_file(str(input_path), str(output_path))

        assert output_path.exists()

    def test_raises_on_missing_file(self, temp_dir):
        """Test that FileNotFoundError is raised for missing input."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            convert_audio_file("/nonexistent/file.wav", str(temp_dir / "out.mp3"))

    def test_raises_on_invalid_format(self, temp_dir):
        """Test that ValueError is raised for invalid format."""
        input_path = temp_dir / "test.wav"
        _create_mock_wav(input_path)

        with pytest.raises(ValueError, match="Unsupported target format"):
            convert_audio_file(str(input_path), str(temp_dir / "out.xyz"), target_format="xyz")

    def test_same_format_copies_file(self, temp_dir):
        """Test that same format just copies the file."""
        input_path = temp_dir / "test.wav"
        output_path = temp_dir / "copy.wav"
        _create_mock_wav(input_path)

        result = convert_audio_file(str(input_path), str(output_path), target_format="wav")

        assert output_path.exists()
        # Should be same size since it's a copy
        assert input_path.stat().st_size == output_path.stat().st_size


class TestBatchConvertAudio:
    """Tests for batch_convert_audio function."""

    def test_converts_directory(self, temp_dir):
        """Test batch converting a directory."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        for i in range(3):
            _create_mock_wav(input_dir / f"audio_{i}.wav")

        result = batch_convert_audio(
            str(input_dir),
            str(output_dir),
            "mp3",
            verbose=False,
        )

        assert result["files_converted"] == 3
        assert result["files_skipped"] == 0
        assert result["files_failed"] == 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.mp3"))) == 3

    def test_skips_same_format(self, temp_dir):
        """Test that files already in target format are skipped."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "test.wav")

        result = batch_convert_audio(
            str(input_dir),
            str(output_dir),
            "wav",  # Same format
            verbose=False,
        )

        assert result["files_converted"] == 0
        assert result["files_skipped"] == 1

    def test_handles_empty_directory(self, temp_dir):
        """Test handling of empty directory."""
        input_dir = temp_dir / "empty"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        result = batch_convert_audio(
            str(input_dir),
            str(output_dir),
            "mp3",
            verbose=False,
        )

        assert result["files_converted"] == 0
        assert result["files_skipped"] == 0
        assert result["files_failed"] == 0

    def test_raises_on_missing_directory(self, temp_dir):
        """Test that FileNotFoundError is raised for missing directory."""
        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            batch_convert_audio(
                "/nonexistent/dir",
                str(temp_dir / "output"),
                "mp3",
            )

    def test_raises_on_invalid_format(self, temp_dir):
        """Test that ValueError is raised for invalid format."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        with pytest.raises(ValueError, match="Unsupported target format"):
            batch_convert_audio(
                str(input_dir),
                str(temp_dir / "output"),
                "xyz",
            )

    def test_preserves_directory_structure(self, temp_dir):
        """Test that subdirectory structure is preserved."""
        input_dir = temp_dir / "input"
        subdir = input_dir / "subdir"
        subdir.mkdir(parents=True)
        output_dir = temp_dir / "output"

        _create_mock_wav(subdir / "nested.wav")

        result = batch_convert_audio(
            str(input_dir),
            str(output_dir),
            "mp3",
            recursive=True,
            verbose=False,
        )

        assert result["files_converted"] == 1
        expected_output = output_dir / "subdir" / "nested.mp3"
        assert expected_output.exists()

"""
Unit tests for the audio convert CLI command.
"""

import struct
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla.views.cli import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


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


class TestConvertHelp:
    """Tests for convert help and options."""

    def test_convert_help(self, runner):
        """Test convert --help shows all options."""
        result = runner.invoke(cli, ["audio", "convert", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "TARGET_FORMAT" in result.output
        assert "--output" in result.output
        assert "--batch" in result.output
        assert "--dataset" in result.output
        assert "--keep-original" in result.output

    def test_convert_requires_args(self, runner):
        """Test that convert requires path and target_format arguments."""
        result = runner.invoke(cli, ["audio", "convert"])

        assert result.exit_code != 0


class TestConvertSingleFile:
    """Tests for single file conversion."""

    def test_convert_single_file(self, runner, temp_dir):
        """Test converting a single audio file."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.mp3"

        result = runner.invoke(cli, [
            "audio", "convert",
            str(audio_file),
            "mp3",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Converted" in result.output

    def test_convert_default_output(self, runner, temp_dir):
        """Test conversion with default output name."""
        audio_file = temp_dir / "my_audio.wav"
        _create_mock_wav(audio_file)

        result = runner.invoke(cli, [
            "audio", "convert",
            str(audio_file),
            "mp3",
        ])

        assert result.exit_code == 0
        expected_output = temp_dir / "my_audio.mp3"
        assert expected_output.exists()

    def test_convert_missing_file(self, runner, temp_dir):
        """Test error handling for missing file."""
        result = runner.invoke(cli, [
            "audio", "convert",
            "/nonexistent/audio.wav",
            "mp3",
        ])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_convert_invalid_format(self, runner, temp_dir):
        """Test error handling for invalid format."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)

        result = runner.invoke(cli, [
            "audio", "convert",
            str(audio_file),
            "xyz",
        ])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_convert_quiet_mode(self, runner, temp_dir):
        """Test quiet mode suppresses output."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.mp3"

        result = runner.invoke(cli, [
            "audio", "convert",
            str(audio_file),
            "mp3",
            "--output", str(output_file),
            "--quiet",
        ])

        assert result.exit_code == 0
        assert output_file.exists()
        assert result.output == ""


class TestConvertBatch:
    """Tests for batch conversion."""

    def test_batch_convert(self, runner, temp_dir):
        """Test batch converting a directory."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        for i in range(3):
            _create_mock_wav(input_dir / f"audio_{i}.wav")

        result = runner.invoke(cli, [
            "audio", "convert",
            str(input_dir),
            "mp3",
            "--batch",
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.mp3"))) == 3

    def test_batch_default_output(self, runner, temp_dir):
        """Test batch mode with default output directory."""
        input_dir = temp_dir / "audio"
        input_dir.mkdir()
        _create_mock_wav(input_dir / "test.wav")

        result = runner.invoke(cli, [
            "audio", "convert",
            str(input_dir),
            "mp3",
            "--batch",
        ])

        assert result.exit_code == 0
        # Default should be input_dir_mp3
        default_output = temp_dir / "audio_mp3"
        assert default_output.exists()

    def test_batch_missing_directory(self, runner, temp_dir):
        """Test error handling for missing directory."""
        result = runner.invoke(cli, [
            "audio", "convert",
            "/nonexistent/dir",
            "mp3",
            "--batch",
        ])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_batch_quiet_mode(self, runner, temp_dir):
        """Test batch quiet mode output."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        _create_mock_wav(input_dir / "test.wav")

        result = runner.invoke(cli, [
            "audio", "convert",
            str(input_dir),
            "mp3",
            "--batch",
            "--output", str(output_dir),
            "--quiet",
        ])

        assert result.exit_code == 0
        assert "Converted" in result.output


class TestConvertFormats:
    """Tests for different audio format conversions."""

    def test_wav_to_flac(self, runner, temp_dir):
        """Test WAV to FLAC conversion."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.flac"

        result = runner.invoke(cli, [
            "audio", "convert",
            str(audio_file),
            "flac",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    @pytest.mark.skip(reason="OGG conversion may fail with mono audio in some ffmpeg configs")
    def test_wav_to_ogg(self, runner, temp_dir):
        """Test WAV to OGG conversion."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.ogg"

        result = runner.invoke(cli, [
            "audio", "convert",
            str(audio_file),
            "ogg",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()

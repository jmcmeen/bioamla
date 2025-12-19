"""
Unit tests for the augment CLI command.
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


class TestAugmentHelp:
    """Tests for augment help and options."""

    def test_augment_help(self, runner):
        """Test augment --help shows all options."""
        result = runner.invoke(cli, ["dataset", "augment", "--help"])

        assert result.exit_code == 0
        assert "INPUT_DIR" in result.output
        assert "--output" in result.output
        assert "--add-noise" in result.output
        assert "--time-stretch" in result.output
        assert "--pitch-shift" in result.output
        assert "--gain" in result.output
        assert "--multiply" in result.output

    def test_augment_requires_input_dir(self, runner):
        """Test that augment requires input directory argument."""
        result = runner.invoke(cli, ["dataset", "augment", "--output", "out"])

        assert result.exit_code != 0

    def test_augment_requires_output(self, runner, temp_dir):
        """Test that augment requires --output option."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        result = runner.invoke(cli, ["dataset", "augment", str(input_dir)])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "--output" in result.output


class TestAugmentExecution:
    """Tests for augment command execution."""

    def test_augment_with_noise(self, runner, temp_dir):
        """Test augmentation with noise."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "test.wav")

        result = runner.invoke(cli, [
            "dataset", "augment",
            str(input_dir),
            "--output", str(output_dir),
            "--add-noise", "3-30",
        ])

        assert result.exit_code == 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.wav"))) == 1

    def test_augment_with_time_stretch(self, runner, temp_dir):
        """Test augmentation with time stretch."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "test.wav")

        result = runner.invoke(cli, [
            "dataset", "augment",
            str(input_dir),
            "--output", str(output_dir),
            "--time-stretch", "0.8-1.2",
        ])

        assert result.exit_code == 0
        assert output_dir.exists()

    def test_augment_with_pitch_shift(self, runner, temp_dir):
        """Test augmentation with pitch shift."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "test.wav")

        result = runner.invoke(cli, [
            "dataset", "augment",
            str(input_dir),
            "--output", str(output_dir),
            "--pitch-shift", "-2,2",
        ])

        assert result.exit_code == 0
        assert output_dir.exists()

    def test_augment_with_gain(self, runner, temp_dir):
        """Test augmentation with gain."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "test.wav")

        result = runner.invoke(cli, [
            "dataset", "augment",
            str(input_dir),
            "--output", str(output_dir),
            "--gain", "-12,12",
        ])

        assert result.exit_code == 0
        assert output_dir.exists()

    def test_augment_with_multiply(self, runner, temp_dir):
        """Test augmentation with multiply option."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "test.wav")

        result = runner.invoke(cli, [
            "dataset", "augment",
            str(input_dir),
            "--output", str(output_dir),
            "--add-noise", "3-30",
            "--multiply", "5",
        ])

        assert result.exit_code == 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.wav"))) == 5

    def test_augment_with_multiple_augmentations(self, runner, temp_dir):
        """Test augmentation with multiple augmentation types."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "test.wav")

        result = runner.invoke(cli, [
            "dataset", "augment",
            str(input_dir),
            "--output", str(output_dir),
            "--add-noise", "3-30",
            "--time-stretch", "0.8-1.2",
            "--pitch-shift", "-2,2",
            "--gain", "-12,12",
        ])

        assert result.exit_code == 0
        assert output_dir.exists()

    def test_augment_requires_at_least_one_augmentation(self, runner, temp_dir):
        """Test that at least one augmentation must be specified."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "test.wav")

        result = runner.invoke(cli, [
            "dataset", "augment",
            str(input_dir),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 1
        assert "At least one augmentation option must be specified" in result.output

    def test_augment_missing_input_dir(self, runner, temp_dir):
        """Test error handling for missing input directory."""
        result = runner.invoke(cli, [
            "dataset", "augment",
            "/nonexistent/dir",
            "--output", str(temp_dir / "output"),
            "--add-noise", "3-30",
        ])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_augment_quiet_mode(self, runner, temp_dir):
        """Test quiet mode output."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "test.wav")

        result = runner.invoke(cli, [
            "dataset", "augment",
            str(input_dir),
            "--output", str(output_dir),
            "--add-noise", "3-30",
            "--quiet",
        ])

        assert result.exit_code == 0
        assert "Created 1 augmented files" in result.output

    def test_augment_multiple_files(self, runner, temp_dir):
        """Test augmentation of multiple files."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        for i in range(3):
            _create_mock_wav(input_dir / f"audio_{i}.wav")

        result = runner.invoke(cli, [
            "dataset", "augment",
            str(input_dir),
            "--output", str(output_dir),
            "--add-noise", "3-30",
        ])

        assert result.exit_code == 0
        assert len(list(output_dir.glob("*.wav"))) == 3


class TestParseRange:
    """Tests for range parsing in augment command."""

    def test_parse_dash_range(self, runner, temp_dir):
        """Test parsing range with dash separator."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "test.wav")

        # This should work with dash separator
        result = runner.invoke(cli, [
            "dataset", "augment",
            str(input_dir),
            "--output", str(output_dir),
            "--add-noise", "5-25",
        ])

        assert result.exit_code == 0

    def test_parse_comma_range(self, runner, temp_dir):
        """Test parsing range with comma separator."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "test.wav")

        # This should work with comma separator (for negative values)
        result = runner.invoke(cli, [
            "dataset", "augment",
            str(input_dir),
            "--output", str(output_dir),
            "--pitch-shift", "-4,4",
        ])

        assert result.exit_code == 0

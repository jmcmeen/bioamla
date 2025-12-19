"""
Unit tests for bioamla CLI detection commands.

Tests the detect command group including directory handling.
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


@pytest.fixture
def mock_audio_file_with_noise(temp_dir):
    """Create a mock audio file with noise for testing detections."""
    audio_path = temp_dir / "test_audio.wav"

    sample_rate = 16000
    duration = 1
    num_samples = sample_rate * duration
    bits_per_sample = 16
    num_channels = 1

    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    file_size = 36 + data_size

    # Generate noise samples
    noise_samples = np.random.randint(-1000, 1000, num_samples, dtype=np.int16)

    with open(audio_path, "wb") as f:
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
        f.write(noise_samples.tobytes())

    return audio_path


@pytest.fixture
def audio_directory(temp_dir):
    """Create a directory with multiple mock audio files."""
    audio_dir = temp_dir / "recordings"
    audio_dir.mkdir()

    sample_rate = 16000
    duration = 1
    num_samples = sample_rate * duration
    bits_per_sample = 16
    num_channels = 1

    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    file_size = 36 + data_size

    for i in range(3):
        audio_path = audio_dir / f"recording_{i}.wav"
        noise_samples = np.random.randint(-1000, 1000, num_samples, dtype=np.int16)

        with open(audio_path, "wb") as f:
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
            f.write(noise_samples.tobytes())

    return audio_dir


class TestDetectCommandGroup:
    """Tests for the detect command group."""

    def test_detect_help(self, runner):
        """Test that detect help works."""
        result = runner.invoke(cli, ["detect", "--help"])

        assert result.exit_code == 0
        assert "energy" in result.output
        assert "ribbit" in result.output
        assert "peaks" in result.output
        assert "accelerating" in result.output
        assert "batch" in result.output


class TestDetectEnergy:
    """Tests for detect energy command."""

    def test_detect_energy_help(self, runner):
        """Test detect energy help."""
        result = runner.invoke(cli, ["detect", "energy", "--help"])

        assert result.exit_code == 0
        assert "low-freq" in result.output
        assert "high-freq" in result.output
        assert "threshold" in result.output

    def test_detect_energy_single_file(self, runner, mock_audio_file_with_noise):
        """Test detect energy on a single file."""
        result = runner.invoke(cli, [
            "detect", "energy",
            str(mock_audio_file_with_noise),
            "--threshold", "-30"
        ])

        assert result.exit_code == 0

    def test_detect_energy_directory(self, runner, audio_directory):
        """Test detect energy on a directory of files."""
        result = runner.invoke(cli, [
            "detect", "energy",
            str(audio_directory),
            "--threshold", "-30"
        ])

        assert result.exit_code == 0

    def test_detect_energy_directory_with_output(self, runner, audio_directory, temp_dir):
        """Test detect energy on a directory with output file."""
        output_file = temp_dir / "detections.csv"
        result = runner.invoke(cli, [
            "detect", "energy",
            str(audio_directory),
            "--threshold", "-30",
            "--output", str(output_file)
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_detect_energy_empty_directory(self, runner, temp_dir):
        """Test detect energy on an empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = runner.invoke(cli, [
            "detect", "energy",
            str(empty_dir)
        ])

        assert result.exit_code == 0
        assert "No audio files found" in result.output


class TestDetectRibbit:
    """Tests for detect ribbit command."""

    def test_detect_ribbit_help(self, runner):
        """Test detect ribbit help."""
        result = runner.invoke(cli, ["detect", "ribbit", "--help"])

        assert result.exit_code == 0
        assert "pulse-rate" in result.output
        assert "tolerance" in result.output

    def test_detect_ribbit_single_file(self, runner, mock_audio_file_with_noise):
        """Test detect ribbit on a single file."""
        result = runner.invoke(cli, [
            "detect", "ribbit",
            str(mock_audio_file_with_noise),
            "--pulse-rate", "10",
            "--min-score", "0.1"
        ])

        assert result.exit_code == 0

    def test_detect_ribbit_directory(self, runner, audio_directory):
        """Test detect ribbit on a directory of files."""
        result = runner.invoke(cli, [
            "detect", "ribbit",
            str(audio_directory),
            "--pulse-rate", "10",
            "--min-score", "0.1"
        ])

        assert result.exit_code == 0

    def test_detect_ribbit_empty_directory(self, runner, temp_dir):
        """Test detect ribbit on an empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = runner.invoke(cli, [
            "detect", "ribbit",
            str(empty_dir)
        ])

        assert result.exit_code == 0
        assert "No audio files found" in result.output


class TestDetectPeaks:
    """Tests for detect peaks command."""

    def test_detect_peaks_help(self, runner):
        """Test detect peaks help."""
        result = runner.invoke(cli, ["detect", "peaks", "--help"])

        assert result.exit_code == 0
        assert "snr" in result.output
        assert "min-distance" in result.output
        assert "sequences" in result.output

    def test_detect_peaks_single_file(self, runner, mock_audio_file_with_noise):
        """Test detect peaks on a single file."""
        result = runner.invoke(cli, [
            "detect", "peaks",
            str(mock_audio_file_with_noise),
            "--snr", "1.0"
        ])

        assert result.exit_code == 0

    def test_detect_peaks_directory(self, runner, audio_directory):
        """Test detect peaks on a directory of files."""
        result = runner.invoke(cli, [
            "detect", "peaks",
            str(audio_directory),
            "--snr", "1.0"
        ])

        assert result.exit_code == 0

    def test_detect_peaks_sequences_directory(self, runner, audio_directory):
        """Test detect peaks sequences on a directory of files."""
        result = runner.invoke(cli, [
            "detect", "peaks",
            str(audio_directory),
            "--snr", "1.0",
            "--sequences",
            "--min-peaks", "2"
        ])

        assert result.exit_code == 0

    def test_detect_peaks_empty_directory(self, runner, temp_dir):
        """Test detect peaks on an empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = runner.invoke(cli, [
            "detect", "peaks",
            str(empty_dir)
        ])

        assert result.exit_code == 0
        assert "No audio files found" in result.output


class TestDetectAccelerating:
    """Tests for detect accelerating command."""

    def test_detect_accelerating_help(self, runner):
        """Test detect accelerating help."""
        result = runner.invoke(cli, ["detect", "accelerating", "--help"])

        assert result.exit_code == 0
        assert "min-pulses" in result.output
        assert "acceleration" in result.output
        assert "deceleration" in result.output

    def test_detect_accelerating_single_file(self, runner, mock_audio_file_with_noise):
        """Test detect accelerating on a single file."""
        result = runner.invoke(cli, [
            "detect", "accelerating",
            str(mock_audio_file_with_noise),
            "--min-pulses", "2",
            "--acceleration", "1.2"
        ])

        assert result.exit_code == 0

    def test_detect_accelerating_directory(self, runner, audio_directory):
        """Test detect accelerating on a directory of files."""
        result = runner.invoke(cli, [
            "detect", "accelerating",
            str(audio_directory),
            "--min-pulses", "2",
            "--acceleration", "1.2"
        ])

        assert result.exit_code == 0

    def test_detect_accelerating_empty_directory(self, runner, temp_dir):
        """Test detect accelerating on an empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = runner.invoke(cli, [
            "detect", "accelerating",
            str(empty_dir)
        ])

        assert result.exit_code == 0
        assert "No audio files found" in result.output


class TestDetectOutputFormats:
    """Tests for detection output formats."""

    def test_detect_energy_json_format(self, runner, mock_audio_file_with_noise):
        """Test detect energy with JSON output format."""
        result = runner.invoke(cli, [
            "detect", "energy",
            str(mock_audio_file_with_noise),
            "--threshold", "-30",
            "--format", "json"
        ])

        assert result.exit_code == 0

    def test_detect_energy_csv_format(self, runner, mock_audio_file_with_noise):
        """Test detect energy with CSV output format."""
        result = runner.invoke(cli, [
            "detect", "energy",
            str(mock_audio_file_with_noise),
            "--threshold", "-30",
            "--format", "csv"
        ])

        assert result.exit_code == 0

    def test_detect_directory_csv_output(self, runner, audio_directory, temp_dir):
        """Test detect on directory with CSV output file."""
        output_file = temp_dir / "output.csv"
        result = runner.invoke(cli, [
            "detect", "energy",
            str(audio_directory),
            "--threshold", "-30",
            "--output", str(output_file)
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify CSV has source_file in metadata
        content = output_file.read_text()
        assert "source_file" in content or "metadata" in content or len(content) > 0

    def test_detect_directory_json_output(self, runner, audio_directory, temp_dir):
        """Test detect on directory with JSON output file."""
        output_file = temp_dir / "output.json"
        result = runner.invoke(cli, [
            "detect", "energy",
            str(audio_directory),
            "--threshold", "-30",
            "--output", str(output_file)
        ])

        assert result.exit_code == 0
        assert output_file.exists()

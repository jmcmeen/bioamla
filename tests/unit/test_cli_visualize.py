"""
Unit tests for the visualize CLI command.
"""

import struct
from pathlib import Path

import pytest
from click.testing import CliRunner

from bioamla.cli import cli


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
        f.write(b"\x00" * data_size)


class TestVisualizeHelp:
    """Tests for visualize help and options."""

    def test_visualize_help(self, runner):
        """Test visualize --help shows all options."""
        result = runner.invoke(cli, ["visualize", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "--output" in result.output
        assert "--batch" in result.output
        assert "--type" in result.output
        assert "mel" in result.output
        assert "mfcc" in result.output
        assert "waveform" in result.output

    def test_visualize_requires_path(self, runner):
        """Test that visualize requires path argument."""
        result = runner.invoke(cli, ["visualize"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "PATH" in result.output


class TestVisualizeSingleFile:
    """Tests for single file visualization."""

    def test_generates_spectrogram(self, runner, temp_dir):
        """Test generating a spectrogram for a single file."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.png"

        result = runner.invoke(cli, [
            "visualize",
            str(audio_file),
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Generated mel spectrogram" in result.output

    def test_generates_mfcc(self, runner, temp_dir):
        """Test generating MFCC visualization."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.png"

        result = runner.invoke(cli, [
            "visualize",
            str(audio_file),
            "--output", str(output_file),
            "--type", "mfcc",
        ])

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Generated mfcc spectrogram" in result.output

    def test_generates_waveform(self, runner, temp_dir):
        """Test generating waveform visualization."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.png"

        result = runner.invoke(cli, [
            "visualize",
            str(audio_file),
            "--output", str(output_file),
            "--type", "waveform",
        ])

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Generated waveform spectrogram" in result.output

    def test_default_output_name(self, runner, temp_dir):
        """Test default output name based on input file."""
        audio_file = temp_dir / "my_audio.wav"
        _create_mock_wav(audio_file)

        result = runner.invoke(cli, [
            "visualize",
            str(audio_file),
        ])

        assert result.exit_code == 0
        expected_output = temp_dir / "my_audio.png"
        assert expected_output.exists()

    def test_missing_file_error(self, runner, temp_dir):
        """Test error handling for missing audio file."""
        result = runner.invoke(cli, [
            "visualize",
            "/nonexistent/audio.wav",
            "--output", str(temp_dir / "output.png"),
        ])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_quiet_mode(self, runner, temp_dir):
        """Test quiet mode suppresses output."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.png"

        result = runner.invoke(cli, [
            "visualize",
            str(audio_file),
            "--output", str(output_file),
            "--quiet",
        ])

        assert result.exit_code == 0
        assert output_file.exists()
        assert result.output == ""


class TestVisualizeBatch:
    """Tests for batch visualization."""

    def test_batch_mode(self, runner, temp_dir):
        """Test batch processing of a directory."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        # Create multiple audio files
        for i in range(3):
            _create_mock_wav(input_dir / f"audio_{i}.wav")

        result = runner.invoke(cli, [
            "visualize",
            str(input_dir),
            "--batch",
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.png"))) == 3

    def test_batch_default_output_dir(self, runner, temp_dir):
        """Test batch mode with default output directory."""
        input_dir = temp_dir / "audio"
        input_dir.mkdir()
        _create_mock_wav(input_dir / "test.wav")

        result = runner.invoke(cli, [
            "visualize",
            str(input_dir),
            "--batch",
        ])

        assert result.exit_code == 0
        # Default should be input_dir/spectrograms
        default_output = input_dir / "spectrograms"
        assert default_output.exists()

    def test_batch_quiet_mode(self, runner, temp_dir):
        """Test batch quiet mode output."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        _create_mock_wav(input_dir / "test.wav")

        result = runner.invoke(cli, [
            "visualize",
            str(input_dir),
            "--batch",
            "--output", str(output_dir),
            "--quiet",
        ])

        assert result.exit_code == 0
        assert "Generated 1 spectrograms" in result.output

    def test_batch_different_viz_types(self, runner, temp_dir):
        """Test batch mode with different visualization types."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        _create_mock_wav(input_dir / "test.wav")

        for viz_type in ["mel", "mfcc", "waveform"]:
            output_dir = temp_dir / f"output_{viz_type}"

            result = runner.invoke(cli, [
                "visualize",
                str(input_dir),
                "--batch",
                "--output", str(output_dir),
                "--type", viz_type,
            ])

            assert result.exit_code == 0
            assert output_dir.exists()


class TestVisualizeOptions:
    """Tests for visualization options."""

    def test_custom_sample_rate(self, runner, temp_dir):
        """Test custom sample rate option."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.png"

        result = runner.invoke(cli, [
            "visualize",
            str(audio_file),
            "--output", str(output_file),
            "--sample-rate", "8000",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_custom_n_mels(self, runner, temp_dir):
        """Test custom n-mels option."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.png"

        result = runner.invoke(cli, [
            "visualize",
            str(audio_file),
            "--output", str(output_file),
            "--n-mels", "64",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_custom_cmap(self, runner, temp_dir):
        """Test custom colormap option."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.png"

        result = runner.invoke(cli, [
            "visualize",
            str(audio_file),
            "--output", str(output_file),
            "--cmap", "viridis",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

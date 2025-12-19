"""
Unit tests for the audio signal processing CLI commands.
"""

import csv
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


class TestFilterCommand:
    """Tests for audio filter command."""

    def test_filter_help(self, runner):
        """Test filter --help shows all options."""
        result = runner.invoke(cli, ["audio", "filter", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "--bandpass" in result.output
        assert "--lowpass" in result.output
        assert "--highpass" in result.output
        assert "--batch" in result.output

    def test_bandpass_filter(self, runner, temp_dir):
        """Test bandpass filtering a file."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "filter",
            str(audio_file),
            "--output", str(output_file),
            "--bandpass", "500-4000",
        ])

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Saved" in result.output

    def test_lowpass_filter(self, runner, temp_dir):
        """Test lowpass filtering a file."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "filter",
            str(audio_file),
            "--output", str(output_file),
            "--lowpass", "2000",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_highpass_filter(self, runner, temp_dir):
        """Test highpass filtering a file."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "filter",
            str(audio_file),
            "--output", str(output_file),
            "--highpass", "500",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_filter_missing_file(self, runner, temp_dir):
        """Test error handling for missing file."""
        result = runner.invoke(cli, [
            "audio", "filter",
            "/nonexistent/audio.wav",
            "--lowpass", "2000",
        ])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_filter_batch_mode(self, runner, temp_dir):
        """Test batch filtering a directory."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        for i in range(2):
            _create_mock_wav(input_dir / f"audio_{i}.wav")

        result = runner.invoke(cli, [
            "audio", "filter",
            str(input_dir),
            "--batch",
            "--output", str(output_dir),
            "--lowpass", "4000",
        ])

        assert result.exit_code == 0
        assert output_dir.exists()


class TestDenoiseCommand:
    """Tests for audio denoise command."""

    def test_denoise_help(self, runner):
        """Test denoise --help shows all options."""
        result = runner.invoke(cli, ["audio", "denoise", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "--method" in result.output
        assert "--strength" in result.output
        assert "--batch" in result.output

    def test_denoise_single_file(self, runner, temp_dir):
        """Test denoising a single file."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "denoise",
            str(audio_file),
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_denoise_with_strength(self, runner, temp_dir):
        """Test denoising with custom strength."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "denoise",
            str(audio_file),
            "--output", str(output_file),
            "--strength", "0.5",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_denoise_batch_mode(self, runner, temp_dir):
        """Test batch denoising."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "audio.wav")

        result = runner.invoke(cli, [
            "audio", "denoise",
            str(input_dir),
            "--batch",
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0


class TestSegmentCommand:
    """Tests for audio segment command."""

    def test_segment_help(self, runner):
        """Test segment --help shows all options."""
        result = runner.invoke(cli, ["audio", "segment", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "--output" in result.output
        assert "--silence-threshold" in result.output
        assert "--min-silence" in result.output
        assert "--min-segment" in result.output

    def test_segment_single_file(self, runner, temp_dir):
        """Test segmenting a file on silence."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_dir = temp_dir / "segments"

        result = runner.invoke(cli, [
            "audio", "segment",
            str(audio_file),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0
        assert output_dir.exists()

    def test_segment_custom_threshold(self, runner, temp_dir):
        """Test segmenting with custom threshold."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_dir = temp_dir / "segments"

        result = runner.invoke(cli, [
            "audio", "segment",
            str(audio_file),
            "--output", str(output_dir),
            "--silence-threshold", "-50",
        ])

        assert result.exit_code == 0

    def test_segment_missing_file(self, runner, temp_dir):
        """Test error handling for missing file."""
        result = runner.invoke(cli, [
            "audio", "segment",
            "/nonexistent/audio.wav",
            "--output", str(temp_dir / "out"),
        ])

        assert result.exit_code == 1
        assert "Error" in result.output


class TestDetectEventsCommand:
    """Tests for audio detect-events command."""

    def test_detect_events_help(self, runner):
        """Test detect-events --help shows all options."""
        result = runner.invoke(cli, ["audio", "detect-events", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "--output" in result.output

    def test_detect_events_single_file(self, runner, temp_dir):
        """Test detecting events in a file."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "events.csv"

        result = runner.invoke(cli, [
            "audio", "detect-events",
            str(audio_file),
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_detect_events_csv_format(self, runner, temp_dir):
        """Test that output is valid CSV."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "events.csv"

        runner.invoke(cli, [
            "audio", "detect-events",
            str(audio_file),
            "--output", str(output_file),
        ])

        # Verify CSV format
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Should have time and strength columns
            if rows:
                assert "time" in rows[0]
                assert "strength" in rows[0]

    def test_detect_events_missing_file(self, runner, temp_dir):
        """Test error handling for missing file."""
        result = runner.invoke(cli, [
            "audio", "detect-events",
            "/nonexistent/audio.wav",
            "--output", str(temp_dir / "events.csv"),
        ])

        assert result.exit_code == 1
        assert "Error" in result.output


class TestNormalizeCommand:
    """Tests for audio normalize command."""

    def test_normalize_help(self, runner):
        """Test normalize --help shows all options."""
        result = runner.invoke(cli, ["audio", "normalize", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "--target-db" in result.output
        assert "--peak" in result.output
        assert "--batch" in result.output

    def test_normalize_single_file(self, runner, temp_dir):
        """Test normalizing a single file."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "normalize",
            str(audio_file),
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_normalize_custom_target(self, runner, temp_dir):
        """Test normalizing with custom target dB."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "normalize",
            str(audio_file),
            "--output", str(output_file),
            "--target-db", "-16",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_normalize_peak_mode(self, runner, temp_dir):
        """Test peak normalization."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "normalize",
            str(audio_file),
            "--output", str(output_file),
            "--peak",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_normalize_batch_mode(self, runner, temp_dir):
        """Test batch normalization."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        for i in range(2):
            _create_mock_wav(input_dir / f"audio_{i}.wav")

        result = runner.invoke(cli, [
            "audio", "normalize",
            str(input_dir),
            "--batch",
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0


class TestResampleCommand:
    """Tests for audio resample command."""

    def test_resample_help(self, runner):
        """Test resample --help shows all options."""
        result = runner.invoke(cli, ["audio", "resample", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "--rate" in result.output
        assert "--batch" in result.output

    def test_resample_single_file(self, runner, temp_dir):
        """Test resampling a single file."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "resample",
            str(audio_file),
            "--output", str(output_file),
            "--rate", "8000",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_resample_batch_mode(self, runner, temp_dir):
        """Test batch resampling."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "audio.wav")

        result = runner.invoke(cli, [
            "audio", "resample",
            str(input_dir),
            "--batch",
            "--output", str(output_dir),
            "--rate", "22050",
        ])

        assert result.exit_code == 0

    def test_resample_missing_rate(self, runner, temp_dir):
        """Test error when --rate is not provided."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)

        result = runner.invoke(cli, [
            "audio", "resample",
            str(audio_file),
        ])

        assert result.exit_code != 0


class TestTrimCommand:
    """Tests for audio trim command."""

    def test_trim_help(self, runner):
        """Test trim --help shows all options."""
        result = runner.invoke(cli, ["audio", "trim", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "--start" in result.output
        assert "--end" in result.output
        assert "--silence" in result.output
        assert "--batch" in result.output

    def test_trim_by_time(self, runner, temp_dir):
        """Test trimming by time range."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file, duration=2.0)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "trim",
            str(audio_file),
            "--output", str(output_file),
            "--start", "0.5",
            "--end", "1.5",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_trim_start_only(self, runner, temp_dir):
        """Test trimming start only."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file, duration=2.0)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "trim",
            str(audio_file),
            "--output", str(output_file),
            "--start", "0.5",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_trim_silence(self, runner, temp_dir):
        """Test trimming silence."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "trim",
            str(audio_file),
            "--output", str(output_file),
            "--silence",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_trim_silence_custom_threshold(self, runner, temp_dir):
        """Test trimming silence with custom threshold."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "trim",
            str(audio_file),
            "--output", str(output_file),
            "--silence",
            "--threshold", "-50",
        ])

        assert result.exit_code == 0

    def test_trim_requires_args(self, runner, temp_dir):
        """Test error when no trim options provided."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)

        result = runner.invoke(cli, [
            "audio", "trim",
            str(audio_file),
        ])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_trim_batch_mode(self, runner, temp_dir):
        """Test batch trimming."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        _create_mock_wav(input_dir / "audio.wav")

        result = runner.invoke(cli, [
            "audio", "trim",
            str(input_dir),
            "--batch",
            "--output", str(output_dir),
            "--silence",
        ])

        assert result.exit_code == 0


class TestQuietMode:
    """Tests for quiet mode across commands."""

    def test_filter_quiet(self, runner, temp_dir):
        """Test filter quiet mode."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "filter",
            str(audio_file),
            "--output", str(output_file),
            "--lowpass", "4000",
            "--quiet",
        ])

        assert result.exit_code == 0
        assert result.output == ""

    def test_denoise_quiet(self, runner, temp_dir):
        """Test denoise quiet mode."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "denoise",
            str(audio_file),
            "--output", str(output_file),
            "--quiet",
        ])

        assert result.exit_code == 0
        assert result.output == ""

    def test_normalize_quiet(self, runner, temp_dir):
        """Test normalize quiet mode."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)
        output_file = temp_dir / "output.wav"

        result = runner.invoke(cli, [
            "audio", "normalize",
            str(audio_file),
            "--output", str(output_file),
            "--quiet",
        ])

        assert result.exit_code == 0
        assert result.output == ""

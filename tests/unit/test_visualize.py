"""
Unit tests for bioamla.core.visualize module.
"""

from pathlib import Path

import numpy as np
import pytest

from bioamla.core.visualize import (
    batch_generate_spectrograms,
    compute_mel_spectrogram,
    compute_stft,
    generate_spectrogram,
    spectrogram_to_db,
    spectrogram_to_image,
    _get_window_function,
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


class TestSTFTSpectrogram:
    """Tests for STFT spectrogram generation."""

    def test_generates_stft_spectrogram(self, mock_audio_file, temp_dir):
        """Test generating an STFT spectrogram."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="stft",
        )

        assert result == str(output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_stft_with_custom_n_fft(self, mock_audio_file, temp_dir):
        """Test STFT spectrogram with custom FFT size."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="stft",
            n_fft=1024,
        )

        assert output_path.exists()

    def test_stft_with_small_n_fft(self, mock_audio_file, temp_dir):
        """Test STFT spectrogram with small FFT size (256)."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="stft",
            n_fft=256,
            hop_length=128,
        )

        assert output_path.exists()

    def test_stft_with_large_n_fft(self, mock_audio_file, temp_dir):
        """Test STFT spectrogram with large FFT size (4096)."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="stft",
            n_fft=4096,
        )

        assert output_path.exists()


class TestWindowFunctions:
    """Tests for window function selection."""

    @pytest.mark.parametrize("window", ["hann", "hamming", "blackman", "bartlett", "rectangular", "kaiser"])
    def test_all_window_types(self, mock_audio_file, temp_dir, window):
        """Test spectrogram generation with all window types."""
        output_path = temp_dir / f"output_{window}.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="mel",
            window=window,
        )

        assert output_path.exists()

    def test_invalid_window_raises_error(self, mock_audio_file, temp_dir):
        """Test that invalid window type raises ValueError."""
        output_path = temp_dir / "output.png"

        with pytest.raises(ValueError, match="Invalid window type"):
            generate_spectrogram(
                audio_path=str(mock_audio_file),
                output_path=str(output_path),
                window="invalid_window",
            )

    def test_get_window_function_hann(self):
        """Test _get_window_function returns correct Hann window."""
        window = _get_window_function("hann", 256)
        assert window.shape == (256,)
        assert np.isclose(window[0], 0.0, atol=1e-5)  # Hann starts at 0
        assert window[128] > 0.99  # Peak near center (value is ~0.9999)

    def test_get_window_function_hamming(self):
        """Test _get_window_function returns correct Hamming window."""
        window = _get_window_function("hamming", 256)
        assert window.shape == (256,)
        assert window[0] > 0.0  # Hamming doesn't start at 0

    def test_get_window_function_rectangular(self):
        """Test _get_window_function returns rectangular (all ones) window."""
        window = _get_window_function("rectangular", 256)
        assert window.shape == (256,)
        assert np.allclose(window, 1.0)


class TestDBScaling:
    """Tests for dB scaling with min/max limits."""

    def test_db_min_limit(self, mock_audio_file, temp_dir):
        """Test spectrogram with dB minimum limit."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="mel",
            db_min=-60.0,
        )

        assert output_path.exists()

    def test_db_max_limit(self, mock_audio_file, temp_dir):
        """Test spectrogram with dB maximum limit."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="mel",
            db_max=0.0,
        )

        assert output_path.exists()

    def test_db_min_max_limits(self, mock_audio_file, temp_dir):
        """Test spectrogram with both dB min and max limits."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="stft",
            db_min=-80.0,
            db_max=-10.0,
        )

        assert output_path.exists()

    def test_spectrogram_to_db_function(self):
        """Test spectrogram_to_db conversion function."""
        # Create a simple spectrogram with known values
        spec = np.array([[1.0, 0.1, 0.01], [0.5, 0.05, 0.005]])

        db_spec = spectrogram_to_db(spec, ref="max", top_db=80.0)

        # Should be 0 at max, negative elsewhere
        assert db_spec.max() == 0.0
        assert db_spec.min() >= -80.0

    def test_spectrogram_to_db_no_clip(self):
        """Test spectrogram_to_db without top_db clipping."""
        spec = np.array([[1.0, 1e-10]])

        db_spec = spectrogram_to_db(spec, ref="max", top_db=None)

        # Without clipping, should have very negative values
        assert db_spec.min() < -80.0


class TestJPEGExport:
    """Tests for JPEG export support."""

    def test_jpeg_export_with_jpg_extension(self, mock_audio_file, temp_dir):
        """Test exporting spectrogram as JPEG with .jpg extension."""
        output_path = temp_dir / "output.jpg"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="mel",
        )

        assert result == str(output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_jpeg_export_with_jpeg_extension(self, mock_audio_file, temp_dir):
        """Test exporting spectrogram as JPEG with .jpeg extension."""
        output_path = temp_dir / "output.jpeg"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="mel",
        )

        assert result == str(output_path)
        assert output_path.exists()

    def test_jpeg_export_with_format_parameter(self, mock_audio_file, temp_dir):
        """Test exporting as JPEG using format parameter regardless of extension."""
        output_path = temp_dir / "output.png"  # PNG extension but JPEG format

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            viz_type="mel",
            format="jpeg",
        )

        assert output_path.exists()

    def test_batch_jpeg_export(self, temp_dir):
        """Test batch processing with JPEG output format."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        audio_path = input_dir / "test.wav"
        _create_mock_wav(audio_path)

        result = batch_generate_spectrograms(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            format="jpg",
            verbose=False,
        )

        assert result["files_processed"] == 1
        # Check that output has .jpg extension
        jpg_files = list(output_dir.glob("*.jpg"))
        assert len(jpg_files) == 1


class TestComputeSTFT:
    """Tests for compute_stft function."""

    def test_compute_stft_returns_correct_shape(self):
        """Test that compute_stft returns arrays with correct shapes."""
        # Create test audio signal
        sample_rate = 16000
        duration = 1.0
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))

        frequencies, times, stft_mag = compute_stft(audio, sample_rate, n_fft=2048, hop_length=512)

        # Check shapes
        assert frequencies.shape[0] == 1025  # n_fft // 2 + 1
        assert stft_mag.shape[0] == 1025
        assert stft_mag.shape[1] == times.shape[0]

    def test_compute_stft_with_different_windows(self):
        """Test compute_stft with different window functions."""
        sample_rate = 16000
        audio = np.random.randn(sample_rate)

        for window in ["hann", "hamming", "blackman"]:
            frequencies, times, stft_mag = compute_stft(audio, sample_rate, window=window)
            assert stft_mag.shape[0] > 0
            assert stft_mag.shape[1] > 0


class TestComputeMelSpectrogram:
    """Tests for compute_mel_spectrogram function."""

    def test_compute_mel_spectrogram_shape(self):
        """Test that compute_mel_spectrogram returns correct shape."""
        sample_rate = 16000
        audio = np.random.randn(sample_rate)
        n_mels = 128

        times, mel_spec = compute_mel_spectrogram(audio, sample_rate, n_mels=n_mels)

        assert mel_spec.shape[0] == n_mels
        assert mel_spec.shape[1] == times.shape[0]

    def test_compute_mel_spectrogram_with_fmin_fmax(self):
        """Test compute_mel_spectrogram with frequency limits."""
        sample_rate = 16000
        audio = np.random.randn(sample_rate)

        times, mel_spec = compute_mel_spectrogram(
            audio, sample_rate, fmin=100.0, fmax=4000.0
        )

        assert mel_spec.shape[0] == 128  # default n_mels


class TestSpectrogramToImage:
    """Tests for spectrogram_to_image function."""

    def test_spectrogram_to_image_png(self, temp_dir):
        """Test exporting spectrogram array to PNG."""
        spec = np.random.randn(128, 100)
        output_path = temp_dir / "spec.png"

        result = spectrogram_to_image(spec, str(output_path))

        assert result == str(output_path)
        assert output_path.exists()

    def test_spectrogram_to_image_jpeg(self, temp_dir):
        """Test exporting spectrogram array to JPEG."""
        spec = np.random.randn(128, 100)
        output_path = temp_dir / "spec.jpg"

        result = spectrogram_to_image(spec, str(output_path))

        assert output_path.exists()

    def test_spectrogram_to_image_with_options(self, temp_dir):
        """Test spectrogram_to_image with custom options."""
        spec = np.random.randn(128, 100)
        output_path = temp_dir / "spec.png"

        result = spectrogram_to_image(
            spec,
            str(output_path),
            cmap="viridis",
            title="Test Spectrogram",
            xlabel="Time (frames)",
            ylabel="Mel bands",
            colorbar=True,
            colorbar_label="Power (dB)",
            vmin=-80.0,
            vmax=0.0,
        )

        assert output_path.exists()

    def test_spectrogram_to_image_no_colorbar(self, temp_dir):
        """Test spectrogram_to_image without colorbar."""
        spec = np.random.randn(64, 50)
        output_path = temp_dir / "spec.png"

        result = spectrogram_to_image(spec, str(output_path), colorbar=False)

        assert output_path.exists()


class TestDPIOption:
    """Tests for DPI configuration."""

    def test_custom_dpi(self, mock_audio_file, temp_dir):
        """Test generating spectrogram with custom DPI."""
        output_path = temp_dir / "output.png"

        result = generate_spectrogram(
            audio_path=str(mock_audio_file),
            output_path=str(output_path),
            dpi=300,
        )

        assert output_path.exists()
        # Higher DPI should result in larger file size
        assert output_path.stat().st_size > 0


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

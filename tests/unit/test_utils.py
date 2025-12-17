"""
Unit tests for bioamla.utils module.
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from bioamla.utils import (
    SUPPORTED_AUDIO_EXTENSIONS,
    concatenate_audio,
    create_directory,
    create_zip_file,
    directory_exists,
    extract_zip_file,
    file_exists,
    get_audio_files,
    get_files_by_extension,
    get_wav_metadata,
    load_audio,
    loop_audio,
    save_audio,
    to_mono,
    zip_directory,
)


class TestSupportedAudioExtensions:
    """Tests for SUPPORTED_AUDIO_EXTENSIONS constant."""

    def test_common_extensions_included(self):
        """Test that common audio extensions are included."""
        assert ".wav" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".mp3" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".flac" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".ogg" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".m4a" in SUPPORTED_AUDIO_EXTENSIONS

    def test_extensions_are_lowercase(self):
        """Test that all extensions are lowercase."""
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            assert ext == ext.lower()

    def test_extensions_start_with_dot(self):
        """Test that all extensions start with a dot."""
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            assert ext.startswith(".")


class TestGetFilesByExtension:
    """Tests for get_files_by_extension function."""

    def test_finds_files_by_extension(self, temp_dir):
        """Test finding files by extension."""
        # Create test files
        (temp_dir / "file1.txt").touch()
        (temp_dir / "file2.txt").touch()
        (temp_dir / "file3.wav").touch()

        result = get_files_by_extension(str(temp_dir), [".txt"])

        assert len(result) == 2
        assert all(".txt" in f for f in result)

    def test_recursive_search(self, temp_dir):
        """Test recursive file search."""
        # Create nested structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (temp_dir / "file1.wav").touch()
        (subdir / "file2.wav").touch()

        result = get_files_by_extension(str(temp_dir), [".wav"], recursive=True)

        assert len(result) == 2

    def test_non_recursive_search(self, temp_dir):
        """Test non-recursive file search."""
        # Create nested structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (temp_dir / "file1.wav").touch()
        (subdir / "file2.wav").touch()

        result = get_files_by_extension(str(temp_dir), [".wav"], recursive=False)

        assert len(result) == 1

    def test_no_extension_filter_returns_all(self, temp_dir):
        """Test that None extensions returns all files."""
        (temp_dir / "file1.txt").touch()
        (temp_dir / "file2.wav").touch()

        result = get_files_by_extension(str(temp_dir), None)

        assert len(result) == 2

    def test_case_insensitive_extension(self, temp_dir):
        """Test case-insensitive extension matching."""
        (temp_dir / "file1.WAV").touch()
        (temp_dir / "file2.wav").touch()

        result = get_files_by_extension(str(temp_dir), [".wav"])

        assert len(result) == 2

    def test_extension_without_dot(self, temp_dir):
        """Test that extension without dot still works."""
        (temp_dir / "file1.wav").touch()

        result = get_files_by_extension(str(temp_dir), ["wav"])

        assert len(result) == 1

    def test_nonexistent_directory_returns_empty(self):
        """Test that nonexistent directory returns empty list."""
        result = get_files_by_extension("/nonexistent/path", [".txt"])

        assert result == []

    def test_returns_sorted_list(self, temp_dir):
        """Test that results are sorted."""
        (temp_dir / "c.txt").touch()
        (temp_dir / "a.txt").touch()
        (temp_dir / "b.txt").touch()

        result = get_files_by_extension(str(temp_dir), [".txt"])

        # Should be sorted alphabetically
        filenames = [os.path.basename(f) for f in result]
        assert filenames == sorted(filenames)


class TestCreateDirectory:
    """Tests for create_directory function."""

    def test_creates_directory(self, temp_dir):
        """Test that directory is created."""
        new_dir = temp_dir / "new_subdir"
        assert not new_dir.exists()

        result = create_directory(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == str(new_dir)

    def test_creates_nested_directories(self, temp_dir):
        """Test that nested directories are created."""
        nested = temp_dir / "level1" / "level2" / "level3"
        assert not nested.exists()

        create_directory(str(nested))

        assert nested.exists()

    def test_existing_directory_ok(self, temp_dir):
        """Test that existing directory doesn't raise error."""
        existing = temp_dir / "existing"
        existing.mkdir()

        # Should not raise
        result = create_directory(str(existing))
        assert result == str(existing)


class TestFileExists:
    """Tests for file_exists function."""

    def test_existing_file_returns_true(self, temp_dir):
        """Test that existing file returns True."""
        test_file = temp_dir / "test.txt"
        test_file.touch()

        assert file_exists(str(test_file)) is True

    def test_nonexistent_file_returns_false(self, temp_dir):
        """Test that nonexistent file returns False."""
        assert file_exists(str(temp_dir / "nonexistent.txt")) is False

    def test_directory_returns_false(self, temp_dir):
        """Test that directory returns False."""
        assert file_exists(str(temp_dir)) is False


class TestDirectoryExists:
    """Tests for directory_exists function."""

    def test_existing_directory_returns_true(self, temp_dir):
        """Test that existing directory returns True."""
        assert directory_exists(str(temp_dir)) is True

    def test_nonexistent_directory_returns_false(self):
        """Test that nonexistent directory returns False."""
        assert directory_exists("/nonexistent/path") is False

    def test_file_returns_false(self, temp_dir):
        """Test that file returns False."""
        test_file = temp_dir / "test.txt"
        test_file.touch()

        assert directory_exists(str(test_file)) is False


class TestGetAudioFiles:
    """Tests for get_audio_files function."""

    def test_finds_audio_files(self, temp_dir):
        """Test finding audio files."""
        (temp_dir / "file1.wav").touch()
        (temp_dir / "file2.mp3").touch()
        (temp_dir / "file3.txt").touch()

        result = get_audio_files(str(temp_dir))

        assert len(result) == 2
        filenames = [os.path.basename(f) for f in result]
        assert "file1.wav" in filenames
        assert "file2.mp3" in filenames
        assert "file3.txt" not in filenames

    def test_custom_extensions(self, temp_dir):
        """Test custom extensions filter."""
        (temp_dir / "file1.wav").touch()
        (temp_dir / "file2.mp3").touch()

        result = get_audio_files(str(temp_dir), extensions=[".wav"])

        assert len(result) == 1
        assert "wav" in result[0]


class TestGetWavMetadata:
    """Tests for get_wav_metadata function."""

    def test_reads_wav_metadata(self, mock_audio_file):
        """Test reading WAV file metadata."""
        metadata = get_wav_metadata(str(mock_audio_file))

        assert metadata["channels"] == 1
        assert metadata["sample_rate"] == 16000
        assert metadata["sample_width"] == 2  # 16-bit = 2 bytes
        assert metadata["duration"] == 1.0
        assert "num_frames" in metadata
        assert "compression_type" in metadata


class TestZipOperations:
    """Tests for ZIP file operations."""

    def test_create_zip_file(self, temp_dir):
        """Test creating a ZIP file from files."""
        # Create test files
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        zip_path = temp_dir / "output.zip"

        result = create_zip_file([str(file1), str(file2)], str(zip_path))

        assert zip_path.exists()
        assert result == str(zip_path)

        # Verify contents
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            assert "file1.txt" in names
            assert "file2.txt" in names

    def test_zip_directory(self, temp_dir):
        """Test creating a ZIP file from a directory."""
        # Create test directory with files
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("content1")
        subdir = source_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("content2")

        zip_path = temp_dir / "output.zip"

        result = zip_directory(str(source_dir), str(zip_path))

        assert zip_path.exists()
        assert result == str(zip_path)

        # Verify contents
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            assert any("file1.txt" in n for n in names)
            assert any("file2.txt" in n for n in names)

    def test_extract_zip_file(self, temp_dir):
        """Test extracting a ZIP file."""
        # Create a ZIP file first
        source_file = temp_dir / "source.txt"
        source_file.write_text("test content")
        zip_path = temp_dir / "test.zip"

        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(source_file, "source.txt")

        # Extract to new directory
        output_dir = temp_dir / "extracted"

        result = extract_zip_file(str(zip_path), str(output_dir))

        assert output_dir.exists()
        assert (output_dir / "source.txt").exists()
        assert (output_dir / "source.txt").read_text() == "test content"
        assert result == str(output_dir)


class TestToMono:
    """Tests for to_mono function."""

    def test_mono_unchanged(self):
        """Test that mono audio is unchanged."""
        audio = np.array([0.1, 0.2, 0.3, 0.4])
        result = to_mono(audio)

        np.testing.assert_array_equal(result, audio)

    def test_stereo_to_mono_channels_samples(self):
        """Test stereo to mono conversion with (channels, samples) shape."""
        # 2 channels, 4 samples
        left = np.array([0.1, 0.2, 0.3, 0.4])
        right = np.array([0.5, 0.6, 0.7, 0.8])
        stereo = np.vstack([left, right])  # Shape: (2, 4)

        result = to_mono(stereo)

        expected = (left + right) / 2
        np.testing.assert_array_almost_equal(result, expected)

    def test_stereo_to_mono_samples_channels(self):
        """Test stereo to mono conversion with (samples, channels) shape."""
        # 4 samples, 2 channels (will be transposed internally)
        stereo = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]])

        result = to_mono(stereo)

        # Expected: average of each row
        expected = np.array([0.3, 0.4, 0.5, 0.6])
        np.testing.assert_array_almost_equal(result, expected)


class TestConcatenateAudio:
    """Tests for concatenate_audio function."""

    def test_concatenate_single_array(self):
        """Test concatenating a single array returns it unchanged."""
        audio = np.array([0.1, 0.2, 0.3])
        result = concatenate_audio([audio])

        np.testing.assert_array_equal(result, audio)

    def test_concatenate_multiple_arrays(self):
        """Test concatenating multiple arrays."""
        audio1 = np.array([0.1, 0.2])
        audio2 = np.array([0.3, 0.4])
        audio3 = np.array([0.5, 0.6])

        result = concatenate_audio([audio1, audio2, audio3])

        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        np.testing.assert_array_almost_equal(result, expected)

    def test_concatenate_with_crossfade(self):
        """Test concatenating with crossfade."""
        audio1 = np.array([1.0, 1.0, 1.0, 1.0])
        audio2 = np.array([0.0, 0.0, 0.0, 0.0])

        result = concatenate_audio([audio1, audio2], crossfade_samples=2)

        # With 2-sample crossfade:
        # Original audio1: [1.0, 1.0, 1.0, 1.0]
        # After crossfade at end: first 2 unchanged, last 2 faded
        # Fade: audio1[-2:] * [1, 0.5, 0] + audio2[:2] * [0, 0.5, 1]
        assert len(result) == 6  # 4 + 4 - 2 (crossfade overlap)

    def test_concatenate_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            concatenate_audio([])


class TestLoopAudio:
    """Tests for loop_audio function."""

    def test_loop_once_unchanged(self):
        """Test that looping once returns unchanged audio."""
        audio = np.array([0.1, 0.2, 0.3])
        result = loop_audio(audio, num_loops=1)

        np.testing.assert_array_equal(result, audio)

    def test_loop_twice(self):
        """Test looping audio twice."""
        audio = np.array([0.1, 0.2])
        result = loop_audio(audio, num_loops=2)

        expected = np.array([0.1, 0.2, 0.1, 0.2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_loop_three_times(self):
        """Test looping audio three times."""
        audio = np.array([1.0, 2.0])
        result = loop_audio(audio, num_loops=3)

        expected = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_loop_with_crossfade(self):
        """Test looping with crossfade."""
        audio = np.array([1.0, 1.0, 1.0, 1.0])
        result = loop_audio(audio, num_loops=2, crossfade_samples=2)

        # Length should be 8 - 2 = 6 due to crossfade
        assert len(result) == 6

    def test_loop_zero_raises(self):
        """Test that zero loops raises ValueError."""
        audio = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="at least 1"):
            loop_audio(audio, num_loops=0)


class TestLoadAndSaveAudio:
    """Tests for load_audio and save_audio functions."""

    def test_save_and_load_wav(self, temp_dir):
        """Test saving and loading a WAV file."""
        # Create test audio
        sample_rate = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Save
        filepath = str(temp_dir / "test.wav")
        save_audio(filepath, audio, sample_rate)

        assert os.path.exists(filepath)

        # Load
        loaded_audio, loaded_sr = load_audio(filepath)

        assert loaded_sr == sample_rate
        assert len(loaded_audio) == len(audio)
        np.testing.assert_array_almost_equal(loaded_audio, audio, decimal=4)

    def test_load_with_resample(self, temp_dir):
        """Test loading with resampling."""
        # Create test audio at 44100 Hz
        sample_rate = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Save
        filepath = str(temp_dir / "test.wav")
        save_audio(filepath, audio, sample_rate)

        # Load with resampling to 16000 Hz
        loaded_audio, loaded_sr = load_audio(filepath, sample_rate=16000)

        assert loaded_sr == 16000
        # Duration should be preserved (approximately)
        expected_samples = int(16000 * duration)
        assert abs(len(loaded_audio) - expected_samples) < 10

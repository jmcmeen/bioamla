"""
Unit tests for bioamla.utils module.
"""

import os
import zipfile
from pathlib import Path

import pytest

from bioamla.utils import (
    SUPPORTED_AUDIO_EXTENSIONS,
    create_directory,
    create_zip_file,
    directory_exists,
    extract_zip_file,
    file_exists,
    get_audio_files,
    get_files_by_extension,
    get_wav_metadata,
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

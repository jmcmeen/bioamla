"""
Unit tests for bioamla.core.explore module.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from bioamla.core.explore import (
    AudioFileInfo,
    DatasetInfo,
    filter_audio_files,
    get_audio_file_info,
    get_category_summary,
    get_split_summary,
    scan_directory,
    sort_audio_files,
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


class TestAudioFileInfo:
    """Tests for AudioFileInfo dataclass."""

    def test_size_human_bytes(self):
        """Test human-readable size for bytes."""
        info = AudioFileInfo(
            path=Path("/test.wav"),
            filename="test.wav",
            size_bytes=500,
        )
        assert info.size_human == "500.0 B"

    def test_size_human_kilobytes(self):
        """Test human-readable size for kilobytes."""
        info = AudioFileInfo(
            path=Path("/test.wav"),
            filename="test.wav",
            size_bytes=2048,
        )
        assert info.size_human == "2.0 KB"

    def test_size_human_megabytes(self):
        """Test human-readable size for megabytes."""
        info = AudioFileInfo(
            path=Path("/test.wav"),
            filename="test.wav",
            size_bytes=5 * 1024 * 1024,
        )
        assert info.size_human == "5.0 MB"

    def test_duration_human_seconds(self):
        """Test human-readable duration for seconds."""
        info = AudioFileInfo(
            path=Path("/test.wav"),
            filename="test.wav",
            size_bytes=1000,
            duration_seconds=5.5,
        )
        assert info.duration_human == "5.5s"

    def test_duration_human_minutes(self):
        """Test human-readable duration for minutes."""
        info = AudioFileInfo(
            path=Path("/test.wav"),
            filename="test.wav",
            size_bytes=1000,
            duration_seconds=125.0,
        )
        assert info.duration_human == "2m 5.0s"

    def test_duration_human_unknown(self):
        """Test human-readable duration when unknown."""
        info = AudioFileInfo(
            path=Path("/test.wav"),
            filename="test.wav",
            size_bytes=1000,
        )
        assert info.duration_human == "Unknown"


class TestDatasetInfo:
    """Tests for DatasetInfo dataclass."""

    def test_total_size_human(self):
        """Test human-readable total size."""
        info = DatasetInfo(
            path=Path("/test"),
            name="test",
            total_size_bytes=10 * 1024 * 1024,
        )
        assert info.total_size_human == "10.0 MB"


class TestGetAudioFileInfo:
    """Tests for get_audio_file_info function."""

    def test_gets_basic_info(self, temp_dir):
        """Test getting basic file info."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file, duration=1.0)

        info = get_audio_file_info(str(audio_file), include_metadata=False)

        assert info.filename == "test.wav"
        assert info.size_bytes > 0
        assert info.format == "wav"

    def test_gets_audio_metadata(self, temp_dir):
        """Test getting audio metadata."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file, duration=2.0)

        info = get_audio_file_info(str(audio_file), include_metadata=True)

        assert info.sample_rate == 16000
        assert info.num_channels == 1
        assert info.duration_seconds is not None
        assert 1.9 < info.duration_seconds < 2.1

    def test_raises_on_missing_file(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            get_audio_file_info("/nonexistent/file.wav")


class TestScanDirectory:
    """Tests for scan_directory function."""

    def test_scans_directory(self, temp_dir):
        """Test scanning a directory for audio files."""
        for i in range(3):
            _create_mock_wav(temp_dir / f"audio_{i}.wav")

        files, info = scan_directory(str(temp_dir), load_audio_metadata=False)

        assert len(files) == 3
        assert info.total_files == 3
        assert info.name == temp_dir.name

    def test_scans_recursively(self, temp_dir):
        """Test recursive directory scanning."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        _create_mock_wav(temp_dir / "root.wav")
        _create_mock_wav(subdir / "nested.wav")

        files, info = scan_directory(str(temp_dir), recursive=True, load_audio_metadata=False)

        assert len(files) == 2
        assert info.total_files == 2

    def test_non_recursive_scan(self, temp_dir):
        """Test non-recursive directory scanning."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        _create_mock_wav(temp_dir / "root.wav")
        _create_mock_wav(subdir / "nested.wav")

        files, info = scan_directory(str(temp_dir), recursive=False, load_audio_metadata=False)

        assert len(files) == 1
        assert files[0].filename == "root.wav"

    def test_loads_metadata_from_csv(self, temp_dir):
        """Test loading metadata from CSV."""
        _create_mock_wav(temp_dir / "test.wav")
        metadata_csv = temp_dir / "metadata.csv"
        metadata_csv.write_text(
            "file_name,split,target,category,attr_id,attr_lic,attr_url,attr_note\n"
            "test.wav,train,0,species_a,user1,CC-BY,http://example.com,note\n"
        )

        files, info = scan_directory(str(temp_dir), load_audio_metadata=False)

        assert len(files) == 1
        assert files[0].category == "species_a"
        assert files[0].split == "train"
        assert files[0].target == 0
        assert info.has_metadata is True

    def test_aggregates_statistics(self, temp_dir):
        """Test that statistics are aggregated correctly."""
        _create_mock_wav(temp_dir / "a.wav")
        _create_mock_wav(temp_dir / "b.wav")
        metadata_csv = temp_dir / "metadata.csv"
        metadata_csv.write_text(
            "file_name,split,target,category,attr_id,attr_lic,attr_url,attr_note\n"
            "a.wav,train,0,cat_a,user1,CC-BY,http://example.com,\n"
            "b.wav,test,1,cat_b,user1,CC-BY,http://example.com,\n"
        )

        files, info = scan_directory(str(temp_dir), load_audio_metadata=False)

        assert info.categories == {"cat_a": 1, "cat_b": 1}
        assert info.splits == {"train": 1, "test": 1}
        assert info.formats == {"WAV": 2}

    def test_raises_on_missing_directory(self):
        """Test error on missing directory."""
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            scan_directory("/nonexistent/directory")

    def test_raises_on_file_path(self, temp_dir):
        """Test error when path is a file."""
        audio_file = temp_dir / "test.wav"
        _create_mock_wav(audio_file)

        with pytest.raises(ValueError, match="not a directory"):
            scan_directory(str(audio_file))


class TestGetCategorySummary:
    """Tests for get_category_summary function."""

    def test_groups_by_category(self):
        """Test grouping files by category."""
        files = [
            AudioFileInfo(
                path=Path("/a.wav"), filename="a.wav", size_bytes=1000,
                category="cat_a", duration_seconds=1.0
            ),
            AudioFileInfo(
                path=Path("/b.wav"), filename="b.wav", size_bytes=2000,
                category="cat_a", duration_seconds=2.0
            ),
            AudioFileInfo(
                path=Path("/c.wav"), filename="c.wav", size_bytes=3000,
                category="cat_b", duration_seconds=3.0
            ),
        ]

        summary = get_category_summary(files)

        assert len(summary) == 2
        assert summary["cat_a"]["count"] == 2
        assert summary["cat_a"]["total_size"] == 3000
        assert summary["cat_a"]["total_duration"] == 3.0
        assert summary["cat_b"]["count"] == 1

    def test_uncategorized_files(self):
        """Test handling of files without category."""
        files = [
            AudioFileInfo(path=Path("/a.wav"), filename="a.wav", size_bytes=1000),
        ]

        summary = get_category_summary(files)

        assert "Uncategorized" in summary
        assert summary["Uncategorized"]["count"] == 1


class TestGetSplitSummary:
    """Tests for get_split_summary function."""

    def test_groups_by_split(self):
        """Test grouping files by split."""
        files = [
            AudioFileInfo(
                path=Path("/a.wav"), filename="a.wav", size_bytes=1000, split="train"
            ),
            AudioFileInfo(
                path=Path("/b.wav"), filename="b.wav", size_bytes=2000, split="train"
            ),
            AudioFileInfo(
                path=Path("/c.wav"), filename="c.wav", size_bytes=3000, split="test"
            ),
        ]

        summary = get_split_summary(files)

        assert len(summary) == 2
        assert summary["train"]["count"] == 2
        assert summary["test"]["count"] == 1


class TestFilterAudioFiles:
    """Tests for filter_audio_files function."""

    @pytest.fixture
    def sample_files(self):
        """Create sample files for filtering tests."""
        return [
            AudioFileInfo(
                path=Path("/a.wav"), filename="alpha.wav", size_bytes=1000,
                category="cat_a", split="train", format="wav", duration_seconds=1.0
            ),
            AudioFileInfo(
                path=Path("/b.mp3"), filename="beta.mp3", size_bytes=2000,
                category="cat_b", split="test", format="mp3", duration_seconds=5.0
            ),
            AudioFileInfo(
                path=Path("/c.wav"), filename="gamma.wav", size_bytes=3000,
                category="cat_a", split="train", format="wav", duration_seconds=10.0
            ),
        ]

    def test_filter_by_category(self, sample_files):
        """Test filtering by category."""
        result = filter_audio_files(sample_files, category="cat_a")
        assert len(result) == 2

    def test_filter_by_split(self, sample_files):
        """Test filtering by split."""
        result = filter_audio_files(sample_files, split="train")
        assert len(result) == 2

    def test_filter_by_format(self, sample_files):
        """Test filtering by format."""
        result = filter_audio_files(sample_files, format="mp3")
        assert len(result) == 1
        assert result[0].filename == "beta.mp3"

    def test_filter_by_min_duration(self, sample_files):
        """Test filtering by minimum duration."""
        result = filter_audio_files(sample_files, min_duration=3.0)
        assert len(result) == 2

    def test_filter_by_max_duration(self, sample_files):
        """Test filtering by maximum duration."""
        result = filter_audio_files(sample_files, max_duration=3.0)
        assert len(result) == 1

    def test_filter_by_search_term(self, sample_files):
        """Test filtering by search term."""
        result = filter_audio_files(sample_files, search_term="alpha")
        assert len(result) == 1
        assert result[0].filename == "alpha.wav"

    def test_filter_combined(self, sample_files):
        """Test combining multiple filters."""
        result = filter_audio_files(
            sample_files, category="cat_a", format="wav", min_duration=5.0
        )
        assert len(result) == 1
        assert result[0].filename == "gamma.wav"


class TestSortAudioFiles:
    """Tests for sort_audio_files function."""

    @pytest.fixture
    def sample_files(self):
        """Create sample files for sorting tests."""
        return [
            AudioFileInfo(
                path=Path("/c.wav"), filename="charlie.wav", size_bytes=3000,
                category="cat_b", format="wav", duration_seconds=3.0
            ),
            AudioFileInfo(
                path=Path("/a.wav"), filename="alpha.wav", size_bytes=1000,
                category="cat_a", format="mp3", duration_seconds=1.0
            ),
            AudioFileInfo(
                path=Path("/b.wav"), filename="bravo.wav", size_bytes=2000,
                category="cat_c", format="flac", duration_seconds=2.0
            ),
        ]

    def test_sort_by_name(self, sample_files):
        """Test sorting by name."""
        result = sort_audio_files(sample_files, sort_by="name")
        assert result[0].filename == "alpha.wav"
        assert result[1].filename == "bravo.wav"
        assert result[2].filename == "charlie.wav"

    def test_sort_by_size(self, sample_files):
        """Test sorting by size."""
        result = sort_audio_files(sample_files, sort_by="size")
        assert result[0].size_bytes == 1000
        assert result[2].size_bytes == 3000

    def test_sort_by_duration(self, sample_files):
        """Test sorting by duration."""
        result = sort_audio_files(sample_files, sort_by="duration")
        assert result[0].duration_seconds == 1.0
        assert result[2].duration_seconds == 3.0

    def test_sort_by_category(self, sample_files):
        """Test sorting by category."""
        result = sort_audio_files(sample_files, sort_by="category")
        assert result[0].category == "cat_a"
        assert result[2].category == "cat_c"

    def test_sort_by_format(self, sample_files):
        """Test sorting by format."""
        result = sort_audio_files(sample_files, sort_by="format")
        assert result[0].format == "flac"
        assert result[2].format == "wav"

    def test_sort_reverse(self, sample_files):
        """Test reverse sorting."""
        result = sort_audio_files(sample_files, sort_by="name", reverse=True)
        assert result[0].filename == "charlie.wav"
        assert result[2].filename == "alpha.wav"

"""
Unit tests for bioamla.core.datasets module.
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bioamla.core.datasets import convert_filetype
from bioamla.core.metadata import read_metadata_csv


def write_test_metadata(csv_path: Path, rows: list):
    """Helper to write test metadata CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestConvertFiletype:
    """Tests for convert_filetype function."""

    def test_no_converter_removes_row_and_deletes_source(self, temp_dir):
        """No converter: row removed from metadata, source file deleted."""
        # Create dataset structure
        audio_dir = temp_dir / "species_a"
        audio_dir.mkdir()

        # Create source file with unsupported extension for conversion to flac
        source_file = audio_dir / "test_audio.xyz"
        source_file.write_bytes(b"dummy-audio-data")

        # Create metadata
        metadata_path = temp_dir / "metadata.csv"
        rows = [
            {
                "file_name": "species_a/test_audio.xyz",
                "split": "train",
                "target": "1",
                "label": "species_a",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/1",
                "attr_note": ""
            }
        ]
        write_test_metadata(metadata_path, rows)

        # Mock _get_converter to return None (no converter available)
        with patch("bioamla.core.datasets._get_converter", return_value=None):
            stats = convert_filetype(
                dataset_path=str(temp_dir),
                target_format="wav",
                verbose=False
            )

        # Verify row was removed from metadata
        updated_rows, _ = read_metadata_csv(metadata_path)
        assert len(updated_rows) == 0

        # Verify source file was deleted
        assert not source_file.exists()

        # Verify stats
        assert stats["files_failed"] == 1
        assert stats["files_converted"] == 0

    def test_missing_source_removes_row_no_new_file(self, temp_dir):
        """Missing source: row removed, no new file created."""
        # Create metadata pointing to non-existent file
        metadata_path = temp_dir / "metadata.csv"
        rows = [
            {
                "file_name": "missing_dir/missing_file.mp3",
                "split": "train",
                "target": "1",
                "label": "species_a",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/1",
                "attr_note": ""
            }
        ]
        write_test_metadata(metadata_path, rows)

        # Mock _get_converter to return a valid converter
        mock_converter = MagicMock()
        with patch("bioamla.core.datasets._get_converter", return_value=mock_converter):
            stats = convert_filetype(
                dataset_path=str(temp_dir),
                target_format="wav",
                verbose=False
            )

        # Verify row was removed from metadata
        updated_rows, _ = read_metadata_csv(metadata_path)
        assert len(updated_rows) == 0

        # Verify no converted file was created
        converted_path = temp_dir / "missing_dir" / "missing_file.wav"
        assert not converted_path.exists()

        # Verify converter was never called
        mock_converter.assert_not_called()

        # Verify stats
        assert stats["files_failed"] == 1
        assert stats["files_converted"] == 0

    def test_conversion_error_removes_row_and_deletes_source(self, temp_dir):
        """Conversion error: row removed from metadata, source file deleted."""
        # Create dataset structure
        audio_dir = temp_dir / "species_a"
        audio_dir.mkdir()

        # Create source file
        source_file = audio_dir / "error_file.mp3"
        source_file.write_bytes(b"dummy-audio-data")

        # Create metadata
        metadata_path = temp_dir / "metadata.csv"
        rows = [
            {
                "file_name": "species_a/error_file.mp3",
                "split": "train",
                "target": "1",
                "label": "species_a",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/1",
                "attr_note": ""
            }
        ]
        write_test_metadata(metadata_path, rows)

        # Mock _get_converter to return a failing converter
        def failing_converter(src, dst):
            raise RuntimeError("Conversion failed!")

        with patch("bioamla.core.datasets._get_converter", return_value=failing_converter):
            stats = convert_filetype(
                dataset_path=str(temp_dir),
                target_format="wav",
                verbose=False
            )

        # Verify row was removed from metadata
        updated_rows, _ = read_metadata_csv(metadata_path)
        assert len(updated_rows) == 0

        # Verify source file was deleted
        assert not source_file.exists()

        # Verify converted file was not created
        converted_path = audio_dir / "error_file.wav"
        assert not converted_path.exists()

        # Verify stats
        assert stats["files_failed"] == 1
        assert stats["files_converted"] == 0

    def test_successful_conversion_updates_row_and_deletes_old(self, temp_dir):
        """Successful conversion: row updated and old file removed when keep_original=False."""
        # Create dataset structure
        audio_dir = temp_dir / "species_a"
        audio_dir.mkdir()

        # Create source file
        source_file = audio_dir / "success_file.mp3"
        source_file.write_bytes(b"dummy-audio-data")

        # Create metadata
        metadata_path = temp_dir / "metadata.csv"
        rows = [
            {
                "file_name": "species_a/success_file.mp3",
                "split": "train",
                "target": "1",
                "label": "species_a",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/1",
                "attr_note": ""
            }
        ]
        write_test_metadata(metadata_path, rows)

        # Mock _get_converter to return a successful converter
        def successful_converter(src, dst):
            Path(dst).write_bytes(b"converted-audio-data")

        with patch("bioamla.core.datasets._get_converter", return_value=successful_converter):
            stats = convert_filetype(
                dataset_path=str(temp_dir),
                target_format="wav",
                keep_original=False,
                verbose=False
            )

        # Verify row was updated in metadata
        updated_rows, _ = read_metadata_csv(metadata_path)
        assert len(updated_rows) == 1
        assert updated_rows[0]["file_name"] == "species_a/success_file.wav"
        assert updated_rows[0]["attr_note"] == "modified clip from original source"

        # Verify converted file was created
        converted_path = audio_dir / "success_file.wav"
        assert converted_path.exists()

        # Verify old source file was deleted
        assert not source_file.exists()

        # Verify stats
        assert stats["files_converted"] == 1
        assert stats["files_failed"] == 0

    def test_successful_conversion_keeps_original_when_requested(self, temp_dir):
        """Successful conversion with keep_original=True preserves source file."""
        # Create dataset structure
        audio_dir = temp_dir / "species_a"
        audio_dir.mkdir()

        # Create source file
        source_file = audio_dir / "keep_file.mp3"
        source_file.write_bytes(b"dummy-audio-data")

        # Create metadata
        metadata_path = temp_dir / "metadata.csv"
        rows = [
            {
                "file_name": "species_a/keep_file.mp3",
                "split": "train",
                "target": "1",
                "label": "species_a",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/1",
                "attr_note": ""
            }
        ]
        write_test_metadata(metadata_path, rows)

        # Mock _get_converter to return a successful converter
        def successful_converter(src, dst):
            Path(dst).write_bytes(b"converted-audio-data")

        with patch("bioamla.core.datasets._get_converter", return_value=successful_converter):
            stats = convert_filetype(
                dataset_path=str(temp_dir),
                target_format="wav",
                keep_original=True,
                verbose=False
            )

        # Verify converted file was created
        converted_path = audio_dir / "keep_file.wav"
        assert converted_path.exists()

        # Verify source file was kept
        assert source_file.exists()

        # Verify stats
        assert stats["files_converted"] == 1

    def test_skips_files_already_in_target_format(self, temp_dir):
        """Files already in target format are skipped without modification."""
        # Create dataset structure
        audio_dir = temp_dir / "species_a"
        audio_dir.mkdir()

        # Create source file already in target format
        source_file = audio_dir / "already_wav.wav"
        source_file.write_bytes(b"wav-audio-data")

        # Create metadata
        metadata_path = temp_dir / "metadata.csv"
        rows = [
            {
                "file_name": "species_a/already_wav.wav",
                "split": "train",
                "target": "1",
                "label": "species_a",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/1",
                "attr_note": "original note"
            }
        ]
        write_test_metadata(metadata_path, rows)

        mock_converter = MagicMock()
        with patch("bioamla.core.datasets._get_converter", return_value=mock_converter):
            stats = convert_filetype(
                dataset_path=str(temp_dir),
                target_format="wav",
                verbose=False
            )

        # Verify row was preserved unchanged
        updated_rows, _ = read_metadata_csv(metadata_path)
        assert len(updated_rows) == 1
        assert updated_rows[0]["file_name"] == "species_a/already_wav.wav"
        assert updated_rows[0]["attr_note"] == "original note"

        # Verify source file still exists
        assert source_file.exists()

        # Verify converter was never called
        mock_converter.assert_not_called()

        # Verify stats
        assert stats["files_skipped"] == 1
        assert stats["files_converted"] == 0

    def test_handles_mixed_success_and_failure(self, temp_dir):
        """Test handling of mixed successful and failed conversions."""
        # Create dataset structure
        audio_dir = temp_dir / "species_a"
        audio_dir.mkdir()

        # Create source files
        success_file = audio_dir / "success.mp3"
        success_file.write_bytes(b"success-audio-data")

        fail_file = audio_dir / "fail.mp3"
        fail_file.write_bytes(b"fail-audio-data")

        skip_file = audio_dir / "skip.wav"
        skip_file.write_bytes(b"skip-audio-data")

        # Create metadata
        metadata_path = temp_dir / "metadata.csv"
        rows = [
            {
                "file_name": "species_a/success.mp3",
                "split": "train",
                "target": "1",
                "label": "species_a",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/1",
                "attr_note": ""
            },
            {
                "file_name": "species_a/fail.mp3",
                "split": "train",
                "target": "1",
                "label": "species_a",
                "attr_id": "user2",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/2",
                "attr_note": ""
            },
            {
                "file_name": "species_a/skip.wav",
                "split": "train",
                "target": "1",
                "label": "species_a",
                "attr_id": "user3",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/3",
                "attr_note": ""
            }
        ]
        write_test_metadata(metadata_path, rows)

        # Track which files the converter was called with
        call_count = [0]

        def mixed_converter(src, dst):
            call_count[0] += 1
            if "fail" in src:
                raise RuntimeError("Intentional failure")
            Path(dst).write_bytes(b"converted-audio-data")

        with patch("bioamla.core.datasets._get_converter", return_value=mixed_converter):
            stats = convert_filetype(
                dataset_path=str(temp_dir),
                target_format="wav",
                verbose=False
            )

        # Verify metadata has correct rows (success + skip, not fail)
        updated_rows, _ = read_metadata_csv(metadata_path)
        assert len(updated_rows) == 2

        file_names = [row["file_name"] for row in updated_rows]
        assert "species_a/success.wav" in file_names
        assert "species_a/skip.wav" in file_names
        assert "species_a/fail.mp3" not in file_names
        assert "species_a/fail.wav" not in file_names

        # Verify stats
        assert stats["files_converted"] == 1
        assert stats["files_failed"] == 1
        assert stats["files_skipped"] == 1

    def test_invalid_target_format_raises_error(self, temp_dir):
        """Test that invalid target format raises ValueError."""
        metadata_path = temp_dir / "metadata.csv"
        write_test_metadata(metadata_path, [{"file_name": "test.mp3"}])

        with pytest.raises(ValueError, match="Unsupported target format"):
            convert_filetype(
                dataset_path=str(temp_dir),
                target_format="invalid_format",
                verbose=False
            )

    def test_missing_dataset_path_raises_error(self):
        """Test that non-existent dataset path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            convert_filetype(
                dataset_path="/nonexistent/path",
                target_format="wav",
                verbose=False
            )

    def test_missing_metadata_raises_error(self, temp_dir):
        """Test that missing metadata file raises ValueError."""
        with pytest.raises(ValueError, match="Metadata file not found"):
            convert_filetype(
                dataset_path=str(temp_dir),
                target_format="wav",
                verbose=False
            )

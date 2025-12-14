"""
Unit tests for bioamla.core.metadata module.
"""

import csv

from bioamla.core.metadata import (
    OPTIONAL_INAT_FIELDS,
    REQUIRED_FIELDS,
    get_existing_observation_ids,
    read_metadata_csv,
    validate_required_fields,
    write_metadata_csv,
)


class TestConstants:
    """Tests for metadata constants."""

    def test_required_fields_defined(self):
        """Test that required fields are properly defined."""
        assert isinstance(REQUIRED_FIELDS, list)
        assert len(REQUIRED_FIELDS) == 8
        assert "filename" in REQUIRED_FIELDS
        assert "split" in REQUIRED_FIELDS
        assert "target" in REQUIRED_FIELDS
        assert "category" in REQUIRED_FIELDS

    def test_optional_inat_fields_defined(self):
        """Test that optional iNaturalist fields are properly defined."""
        assert isinstance(OPTIONAL_INAT_FIELDS, list)
        assert "observation_id" in OPTIONAL_INAT_FIELDS
        assert "sound_id" in OPTIONAL_INAT_FIELDS
        assert "taxon_id" in OPTIONAL_INAT_FIELDS


class TestReadMetadataCsv:
    """Tests for read_metadata_csv function."""

    def test_read_existing_file(self, metadata_csv_file, sample_metadata_rows):
        """Test reading an existing CSV file."""
        rows, fieldnames = read_metadata_csv(metadata_csv_file)

        assert len(rows) == len(sample_metadata_rows)
        assert "filename" in fieldnames
        assert rows[0]["filename"] == "audio1.wav"

    def test_read_nonexistent_file(self, temp_dir):
        """Test reading a nonexistent file returns empty results."""
        nonexistent = temp_dir / "nonexistent.csv"
        rows, fieldnames = read_metadata_csv(nonexistent)

        assert rows == []
        assert fieldnames == set()

    def test_read_preserves_all_fields(self, metadata_csv_file):
        """Test that all fields are preserved when reading."""
        rows, fieldnames = read_metadata_csv(metadata_csv_file)

        for field in REQUIRED_FIELDS:
            assert field in fieldnames


class TestWriteMetadataCsv:
    """Tests for write_metadata_csv function."""

    def test_write_new_file(self, temp_dir, sample_metadata_rows):
        """Test writing to a new CSV file."""
        csv_path = temp_dir / "new_metadata.csv"

        count = write_metadata_csv(csv_path, sample_metadata_rows, merge_existing=False)

        assert count == len(sample_metadata_rows)
        assert csv_path.exists()

        # Verify content
        rows, fieldnames = read_metadata_csv(csv_path)
        assert len(rows) == len(sample_metadata_rows)

    def test_write_empty_rows(self, temp_dir):
        """Test writing empty rows returns 0."""
        csv_path = temp_dir / "empty.csv"
        count = write_metadata_csv(csv_path, [])
        assert count == 0

    def test_merge_existing_data(self, temp_dir, sample_metadata_rows):
        """Test merging with existing data."""
        csv_path = temp_dir / "merge_test.csv"

        # Write initial data
        initial_rows = sample_metadata_rows[:2]
        write_metadata_csv(csv_path, initial_rows, merge_existing=False)

        # Merge new data
        new_rows = [
            {
                "filename": "audio4.wav",
                "split": "train",
                "target": "3",
                "category": "species_c",
                "attr_id": "user4",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/4",
                "attr_note": ""
            }
        ]
        count = write_metadata_csv(csv_path, new_rows, merge_existing=True)

        assert count == 3  # 2 initial + 1 new

    def test_deduplicate_on_merge(self, temp_dir, sample_metadata_rows):
        """Test that duplicate filenames are skipped during merge."""
        csv_path = temp_dir / "dedup_test.csv"

        # Write initial data
        write_metadata_csv(csv_path, sample_metadata_rows, merge_existing=False)

        # Try to add duplicate
        duplicate_rows = [sample_metadata_rows[0].copy()]
        write_metadata_csv(csv_path, duplicate_rows, merge_existing=True)

        # Should still have only 3 rows (no duplicate added)
        rows, _ = read_metadata_csv(csv_path)
        assert len(rows) == 3


class TestGetExistingObservationIds:
    """Tests for get_existing_observation_ids function."""

    def test_extract_ids_from_filenames(self, temp_dir):
        """Test extraction of observation and sound IDs from filenames."""
        csv_path = temp_dir / "inat_metadata.csv"

        rows = [
            {"filename": "species_a/inat_123_sound_456.mp3", "split": "train",
             "target": "1", "category": "species_a", "attr_id": "", "attr_lic": "",
             "attr_url": "", "attr_note": ""},
            {"filename": "species_b/inat_789_sound_101.wav", "split": "train",
             "target": "2", "category": "species_b", "attr_id": "", "attr_lic": "",
             "attr_url": "", "attr_note": ""},
        ]

        # Write the file
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        existing = get_existing_observation_ids(csv_path)

        assert (123, 456) in existing
        assert (789, 101) in existing
        assert len(existing) == 2

    def test_nonexistent_file_returns_empty(self, temp_dir):
        """Test that nonexistent file returns empty set."""
        nonexistent = temp_dir / "nonexistent.csv"
        existing = get_existing_observation_ids(nonexistent)
        assert existing == set()

    def test_malformed_filenames_skipped(self, temp_dir):
        """Test that malformed filenames are skipped."""
        csv_path = temp_dir / "malformed.csv"

        rows = [
            {"filename": "regular_file.mp3", "split": "train", "target": "1",
             "category": "x", "attr_id": "", "attr_lic": "", "attr_url": "", "attr_note": ""},
            {"filename": "inat_abc_sound_def.mp3", "split": "train", "target": "1",
             "category": "x", "attr_id": "", "attr_lic": "", "attr_url": "", "attr_note": ""},
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        existing = get_existing_observation_ids(csv_path)
        assert len(existing) == 0


class TestValidateRequiredFields:
    """Tests for validate_required_fields function."""

    def test_valid_rows_pass(self, sample_metadata_rows):
        """Test that valid rows pass validation."""
        is_valid, errors = validate_required_fields(sample_metadata_rows)
        assert is_valid
        assert errors == []

    def test_missing_fields_detected(self):
        """Test that missing required fields are detected."""
        invalid_rows = [
            {"filename": "test.wav", "split": "train"}  # Missing several fields
        ]
        is_valid, errors = validate_required_fields(invalid_rows)
        assert not is_valid
        assert len(errors) > 0

    def test_empty_rows_valid(self):
        """Test that empty list is valid."""
        is_valid, errors = validate_required_fields([])
        assert is_valid
        assert errors == []

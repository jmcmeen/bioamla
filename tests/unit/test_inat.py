"""
Unit tests for bioamla.core.inat module.
"""

import csv

import pytest

from bioamla.core.inat import load_taxon_ids_from_csv


class TestLoadTaxonIdsFromCsv:
    """Tests for load_taxon_ids_from_csv function."""

    def test_loads_valid_taxon_ids(self, temp_dir):
        """Test loading valid taxon IDs from CSV."""
        csv_path = temp_dir / "taxa.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["taxon_id"])
            writer.writerow(["12345"])
            writer.writerow(["67890"])
            writer.writerow(["11111"])

        result = load_taxon_ids_from_csv(csv_path)

        assert result == [12345, 67890, 11111]

    def test_loads_from_string_path(self, temp_dir):
        """Test loading from a string path."""
        csv_path = temp_dir / "taxa.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["taxon_id"])
            writer.writerow(["999"])

        result = load_taxon_ids_from_csv(str(csv_path))

        assert result == [999]

    def test_handles_additional_columns(self, temp_dir):
        """Test that additional columns are ignored."""
        csv_path = temp_dir / "taxa.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["taxon_id", "name", "common_name", "observation_count"])
            writer.writerow(["12345", "Lithobates catesbeianus", "American Bullfrog", "150"])
            writer.writerow(["67890", "Anaxyrus americanus", "American Toad", "200"])

        result = load_taxon_ids_from_csv(csv_path)

        assert result == [12345, 67890]

    def test_strips_whitespace(self, temp_dir):
        """Test that whitespace is stripped from taxon IDs."""
        csv_path = temp_dir / "taxa.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["taxon_id"])
            writer.writerow(["  12345  "])
            writer.writerow(["67890"])

        result = load_taxon_ids_from_csv(csv_path)

        assert result == [12345, 67890]

    def test_skips_empty_rows(self, temp_dir):
        """Test that empty taxon_id values are skipped."""
        csv_path = temp_dir / "taxa.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["taxon_id"])
            writer.writerow(["12345"])
            writer.writerow([""])
            writer.writerow(["67890"])
            writer.writerow(["  "])

        result = load_taxon_ids_from_csv(csv_path)

        assert result == [12345, 67890]

    def test_skips_invalid_taxon_ids(self, temp_dir):
        """Test that invalid (non-integer) taxon IDs are skipped with warning."""
        csv_path = temp_dir / "taxa.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["taxon_id"])
            writer.writerow(["12345"])
            writer.writerow(["not_a_number"])
            writer.writerow(["67890"])
            writer.writerow(["12.34"])

        result = load_taxon_ids_from_csv(csv_path)

        assert result == [12345, 67890]

    def test_raises_on_missing_file(self, temp_dir):
        """Test that FileNotFoundError is raised for missing file."""
        nonexistent = temp_dir / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="Taxon CSV file not found"):
            load_taxon_ids_from_csv(nonexistent)

    def test_raises_on_missing_column(self, temp_dir):
        """Test that ValueError is raised when taxon_id column is missing."""
        csv_path = temp_dir / "taxa.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name"])
            writer.writerow(["12345", "Test Species"])

        with pytest.raises(ValueError, match="must have a 'taxon_id' column"):
            load_taxon_ids_from_csv(csv_path)

    def test_raises_on_empty_csv(self, temp_dir):
        """Test that ValueError is raised when CSV has no valid taxon IDs."""
        csv_path = temp_dir / "taxa.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["taxon_id"])
            # No data rows

        with pytest.raises(ValueError, match="No valid taxon IDs found"):
            load_taxon_ids_from_csv(csv_path)

    def test_raises_on_all_invalid_ids(self, temp_dir):
        """Test that ValueError is raised when all taxon IDs are invalid."""
        csv_path = temp_dir / "taxa.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["taxon_id"])
            writer.writerow(["invalid"])
            writer.writerow(["also_invalid"])

        with pytest.raises(ValueError, match="No valid taxon IDs found"):
            load_taxon_ids_from_csv(csv_path)

    def test_returns_list_of_integers(self, temp_dir):
        """Test that returned values are integers, not strings."""
        csv_path = temp_dir / "taxa.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["taxon_id"])
            writer.writerow(["12345"])

        result = load_taxon_ids_from_csv(csv_path)

        assert all(isinstance(tid, int) for tid in result)

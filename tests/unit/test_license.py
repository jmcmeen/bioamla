"""
Unit tests for bioamla.core.license module.
"""

import csv
from pathlib import Path

import pytest

from bioamla.license import (
    format_attribution,
    generate_license_file,
    generate_license_for_dataset,
    generate_licenses_for_directory,
    parse_csv_file,
    read_template_file,
    validate_csv_structure,
    find_datasets,
)


def write_test_metadata(csv_path: Path, rows: list):
    """Helper to write test metadata CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestReadTemplateFile:
    """Tests for read_template_file function."""

    def test_reads_template_content(self, temp_dir):
        """Test that template content is read correctly."""
        template_path = temp_dir / "template.txt"
        template_path.write_text("This is a template\nwith multiple lines")

        result = read_template_file(template_path)

        assert result == "This is a template\nwith multiple lines"

    def test_raises_on_missing_file(self, temp_dir):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            read_template_file(temp_dir / "nonexistent.txt")


class TestParseCsvFile:
    """Tests for parse_csv_file function."""

    def test_parses_valid_csv(self, temp_dir):
        """Test parsing a valid CSV file."""
        csv_path = temp_dir / "metadata.csv"
        rows = [
            {
                "file_name": "audio1.wav",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/1",
                "attr_note": "Test note"
            }
        ]
        write_test_metadata(csv_path, rows)

        result = parse_csv_file(csv_path)

        assert len(result) == 1
        assert result[0]["file_name"] == "audio1.wav"
        assert result[0]["attr_id"] == "user1"
        assert result[0]["attr_lic"] == "CC-BY"

    def test_skips_rows_without_filename(self, temp_dir):
        """Test that rows without file_name are skipped."""
        csv_path = temp_dir / "metadata.csv"
        rows = [
            {
                "file_name": "audio1.wav",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/1",
                "attr_note": ""
            },
            {
                "file_name": "",
                "attr_id": "user2",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com/2",
                "attr_note": ""
            }
        ]
        write_test_metadata(csv_path, rows)

        result = parse_csv_file(csv_path)

        assert len(result) == 1
        assert result[0]["file_name"] == "audio1.wav"

    def test_raises_on_missing_fields(self, temp_dir):
        """Test that ValueError is raised for missing required fields."""
        csv_path = temp_dir / "metadata.csv"
        rows = [{"file_name": "audio1.wav", "other_field": "value"}]
        write_test_metadata(csv_path, rows)

        with pytest.raises(ValueError, match="Missing required fields"):
            parse_csv_file(csv_path)

    def test_raises_on_missing_file(self, temp_dir):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_csv_file(temp_dir / "nonexistent.csv")


class TestFormatAttribution:
    """Tests for format_attribution function."""

    def test_formats_full_attribution(self):
        """Test formatting with all fields populated."""
        attribution = {
            "file_name": "test.wav",
            "attr_id": "author123",
            "attr_lic": "CC-BY-4.0",
            "attr_url": "https://example.com",
            "attr_note": "Modified version"
        }

        result = format_attribution(attribution)

        assert "File: test.wav" in result
        assert "Attribution ID: author123" in result
        assert "License: CC-BY-4.0" in result
        assert "Source URL: https://example.com" in result
        assert "Notes: Modified version" in result

    def test_formats_minimal_attribution(self):
        """Test formatting with only required fields."""
        attribution = {
            "file_name": "test.wav",
            "attr_id": "",
            "attr_lic": "",
            "attr_url": "",
            "attr_note": ""
        }

        result = format_attribution(attribution)

        assert "File: test.wav" in result
        assert "Attribution ID:" not in result
        assert "License:" not in result


class TestGenerateLicenseFile:
    """Tests for generate_license_file function."""

    def test_generates_license_without_template(self, temp_dir):
        """Test generating license file without a template."""
        output_path = temp_dir / "LICENSE"
        attributions = [
            {
                "file_name": "audio1.wav",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com",
                "attr_note": ""
            }
        ]

        result = generate_license_file(attributions, output_path)

        assert output_path.exists()
        assert result["attributions_count"] == 1
        assert result["file_size"] > 0

        content = output_path.read_text()
        assert "LICENSE AND ATTRIBUTION FILE" in content
        assert "audio1.wav" in content

    def test_generates_license_with_template(self, temp_dir):
        """Test generating license file with a template."""
        output_path = temp_dir / "LICENSE"
        template_content = "Custom License Header\n\nAll rights reserved."
        attributions = [
            {
                "file_name": "audio1.wav",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com",
                "attr_note": ""
            }
        ]

        result = generate_license_file(attributions, output_path, template_content)

        content = output_path.read_text()
        assert "Custom License Header" in content
        assert "FILE ATTRIBUTIONS" in content
        assert "audio1.wav" in content


class TestValidateCsvStructure:
    """Tests for validate_csv_structure function."""

    def test_validates_valid_csv(self, temp_dir):
        """Test validation of a valid CSV file."""
        csv_path = temp_dir / "metadata.csv"
        rows = [
            {
                "file_name": "audio1.wav",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com",
                "attr_note": ""
            }
        ]
        write_test_metadata(csv_path, rows)

        result = validate_csv_structure(csv_path)

        assert result["is_valid"] is True
        assert result["row_count"] == 1
        assert len(result["missing_fields"]) == 0

    def test_detects_missing_fields(self, temp_dir):
        """Test that missing fields are detected."""
        csv_path = temp_dir / "metadata.csv"
        rows = [{"file_name": "audio1.wav", "attr_id": "user1"}]
        write_test_metadata(csv_path, rows)

        result = validate_csv_structure(csv_path)

        assert result["is_valid"] is False
        assert "attr_lic" in result["missing_fields"]


class TestFindDatasets:
    """Tests for find_datasets function."""

    def test_finds_datasets_with_metadata(self, temp_dir):
        """Test finding datasets that have metadata.csv."""
        # Create dataset directories
        dataset1 = temp_dir / "dataset1"
        dataset1.mkdir()
        (dataset1 / "metadata.csv").write_text("file_name,attr_id,attr_lic,attr_url,attr_note\n")

        dataset2 = temp_dir / "dataset2"
        dataset2.mkdir()
        (dataset2 / "metadata.csv").write_text("file_name,attr_id,attr_lic,attr_url,attr_note\n")

        # Create directory without metadata
        no_metadata = temp_dir / "no_metadata"
        no_metadata.mkdir()

        result = find_datasets(temp_dir)

        assert len(result) == 2
        dataset_names = [r[0] for r in result]
        assert "dataset1" in dataset_names
        assert "dataset2" in dataset_names

    def test_raises_on_missing_directory(self, temp_dir):
        """Test that FileNotFoundError is raised for missing directory."""
        with pytest.raises(FileNotFoundError):
            find_datasets(temp_dir / "nonexistent")


class TestGenerateLicenseForDataset:
    """Tests for generate_license_for_dataset function."""

    def test_generates_license_for_dataset(self, temp_dir):
        """Test generating license for a single dataset."""
        dataset_path = temp_dir / "my_dataset"
        dataset_path.mkdir()

        rows = [
            {
                "file_name": "audio1.wav",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com",
                "attr_note": ""
            }
        ]
        write_test_metadata(dataset_path / "metadata.csv", rows)

        result = generate_license_for_dataset(dataset_path)

        assert (dataset_path / "LICENSE").exists()
        assert result["attributions_count"] == 1

    def test_uses_custom_output_filename(self, temp_dir):
        """Test using custom output filename."""
        dataset_path = temp_dir / "my_dataset"
        dataset_path.mkdir()

        rows = [
            {
                "file_name": "audio1.wav",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com",
                "attr_note": ""
            }
        ]
        write_test_metadata(dataset_path / "metadata.csv", rows)

        result = generate_license_for_dataset(dataset_path, output_filename="ATTRIBUTION.txt")

        assert (dataset_path / "ATTRIBUTION.txt").exists()
        assert not (dataset_path / "LICENSE").exists()

    def test_raises_on_missing_metadata(self, temp_dir):
        """Test that FileNotFoundError is raised for missing metadata."""
        dataset_path = temp_dir / "my_dataset"
        dataset_path.mkdir()

        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            generate_license_for_dataset(dataset_path)


class TestGenerateLicensesForDirectory:
    """Tests for generate_licenses_for_directory function."""

    def test_generates_licenses_for_multiple_datasets(self, temp_dir):
        """Test generating licenses for multiple datasets."""
        # Create dataset directories
        for name in ["dataset1", "dataset2"]:
            dataset_path = temp_dir / name
            dataset_path.mkdir()
            rows = [
                {
                    "file_name": f"{name}_audio.wav",
                    "attr_id": "user1",
                    "attr_lic": "CC-BY",
                    "attr_url": "https://example.com",
                    "attr_note": ""
                }
            ]
            write_test_metadata(dataset_path / "metadata.csv", rows)

        result = generate_licenses_for_directory(temp_dir)

        assert result["datasets_found"] == 2
        assert result["datasets_processed"] == 2
        assert result["datasets_failed"] == 0
        assert (temp_dir / "dataset1" / "LICENSE").exists()
        assert (temp_dir / "dataset2" / "LICENSE").exists()

    def test_reports_failed_datasets(self, temp_dir):
        """Test that failed datasets are reported."""
        # Create valid dataset
        dataset1 = temp_dir / "dataset1"
        dataset1.mkdir()
        rows = [
            {
                "file_name": "audio.wav",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com",
                "attr_note": ""
            }
        ]
        write_test_metadata(dataset1 / "metadata.csv", rows)

        # Create invalid dataset (missing required fields)
        dataset2 = temp_dir / "dataset2"
        dataset2.mkdir()
        (dataset2 / "metadata.csv").write_text("file_name,other\naudio.wav,value\n")

        result = generate_licenses_for_directory(temp_dir)

        assert result["datasets_found"] == 2
        assert result["datasets_processed"] == 1
        assert result["datasets_failed"] == 1

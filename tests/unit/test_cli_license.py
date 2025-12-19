"""
Unit tests for the dataset license CLI command.
"""

import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from bioamla.views.cli import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


def write_test_metadata(csv_path: Path, rows: list):
    """Helper to write test metadata CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestDatasetLicenseHelp:
    """Tests for dataset license help."""

    def test_license_help(self, runner):
        """Test dataset license --help shows all options."""
        result = runner.invoke(cli, ["dataset", "license", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "--template" in result.output
        assert "--output" in result.output
        assert "--metadata-filename" in result.output
        assert "--batch" in result.output
        assert "--quiet" in result.output

    def test_license_requires_path(self, runner):
        """Test that dataset license requires path argument."""
        result = runner.invoke(cli, ["dataset", "license"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "PATH" in result.output


class TestDatasetLicenseSingleDataset:
    """Tests for single dataset license generation."""

    def test_generates_license_file(self, runner, temp_dir):
        """Test generating license file for a single dataset."""
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

        result = runner.invoke(cli, ["dataset", "license", str(dataset_path)])

        assert result.exit_code == 0
        assert (dataset_path / "LICENSE").exists()
        assert "1 attributions" in result.output or "Attributions: 1" in result.output

    def test_uses_custom_output_filename(self, runner, temp_dir):
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

        result = runner.invoke(cli, [
            "dataset", "license", str(dataset_path),
            "--output", "ATTRIBUTION.txt"
        ])

        assert result.exit_code == 0
        assert (dataset_path / "ATTRIBUTION.txt").exists()

    def test_uses_template_file(self, runner, temp_dir):
        """Test using a template file."""
        dataset_path = temp_dir / "my_dataset"
        dataset_path.mkdir()

        template_path = temp_dir / "template.txt"
        template_path.write_text("Custom License Header\n\nAll rights reserved.")

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

        result = runner.invoke(cli, [
            "dataset", "license", str(dataset_path),
            "--template", str(template_path)
        ])

        assert result.exit_code == 0
        license_content = (dataset_path / "LICENSE").read_text()
        assert "Custom License Header" in license_content

    def test_fails_on_missing_metadata(self, runner, temp_dir):
        """Test failure when metadata file is missing."""
        dataset_path = temp_dir / "my_dataset"
        dataset_path.mkdir()

        result = runner.invoke(cli, ["dataset", "license", str(dataset_path)])

        assert result.exit_code != 0
        assert "not found" in result.output

    def test_fails_on_nonexistent_path(self, runner):
        """Test failure when path doesn't exist."""
        result = runner.invoke(cli, ["dataset", "license", "/nonexistent/path"])

        assert result.exit_code != 0
        assert "not a directory" in result.output

    def test_fails_on_missing_template(self, runner, temp_dir):
        """Test failure when template file is missing."""
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

        result = runner.invoke(cli, [
            "dataset", "license", str(dataset_path),
            "--template", "/nonexistent/template.txt"
        ])

        assert result.exit_code != 0
        assert "Template file" in result.output and "not found" in result.output

    def test_quiet_mode(self, runner, temp_dir):
        """Test quiet mode produces minimal output."""
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

        result = runner.invoke(cli, ["dataset", "license", str(dataset_path), "--quiet"])

        assert result.exit_code == 0
        assert "Generated LICENSE with" in result.output


class TestDatasetLicenseBatch:
    """Tests for batch license generation."""

    def test_batch_generates_licenses_for_multiple_datasets(self, runner, temp_dir):
        """Test batch mode generates licenses for all datasets."""
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

        result = runner.invoke(cli, ["dataset", "license", str(temp_dir), "--batch"])

        assert result.exit_code == 0
        assert (temp_dir / "dataset1" / "LICENSE").exists()
        assert (temp_dir / "dataset2" / "LICENSE").exists()
        assert "Successful: 2" in result.output

    def test_batch_with_template(self, runner, temp_dir):
        """Test batch mode with template file."""
        template_path = temp_dir / "template.txt"
        template_path.write_text("Batch Template Header")

        dataset_path = temp_dir / "dataset1"
        dataset_path.mkdir()
        rows = [
            {
                "file_name": "audio.wav",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com",
                "attr_note": ""
            }
        ]
        write_test_metadata(dataset_path / "metadata.csv", rows)

        result = runner.invoke(cli, [
            "dataset", "license", str(temp_dir),
            "--batch", "--template", str(template_path)
        ])

        assert result.exit_code == 0
        license_content = (dataset_path / "LICENSE").read_text()
        assert "Batch Template Header" in license_content

    def test_batch_no_datasets_found(self, runner, temp_dir):
        """Test batch mode when no datasets are found."""
        result = runner.invoke(cli, ["dataset", "license", str(temp_dir), "--batch"])

        assert result.exit_code != 0
        assert "No datasets found" in result.output

    def test_batch_quiet_mode(self, runner, temp_dir):
        """Test batch mode with quiet output."""
        dataset_path = temp_dir / "dataset1"
        dataset_path.mkdir()
        rows = [
            {
                "file_name": "audio.wav",
                "attr_id": "user1",
                "attr_lic": "CC-BY",
                "attr_url": "https://example.com",
                "attr_note": ""
            }
        ]
        write_test_metadata(dataset_path / "metadata.csv", rows)

        result = runner.invoke(cli, [
            "dataset", "license", str(temp_dir), "--batch", "--quiet"
        ])

        assert result.exit_code == 0
        assert "Generated 1 license files" in result.output

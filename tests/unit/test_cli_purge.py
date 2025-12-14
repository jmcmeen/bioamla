"""
Unit tests for the purge CLI command.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bioamla.cli import cli, _format_size


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestFormatSize:
    """Tests for the _format_size helper function."""

    def test_format_bytes(self):
        assert _format_size(500) == "500.0 B"

    def test_format_kilobytes(self):
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(2048) == "2.0 KB"

    def test_format_megabytes(self):
        assert _format_size(1024 * 1024) == "1.0 MB"
        assert _format_size(1024 * 1024 * 5) == "5.0 MB"

    def test_format_gigabytes(self):
        assert _format_size(1024 * 1024 * 1024) == "1.0 GB"

    def test_format_terabytes(self):
        assert _format_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"


class TestPurgeCommand:
    """Tests for the purge CLI command."""

    def test_purge_no_options_shows_help(self, runner):
        """Test that purge without options shows help message."""
        result = runner.invoke(cli, ["purge"])

        assert result.exit_code == 0
        assert "Please specify what to purge" in result.output
        assert "--models" in result.output

    def test_purge_help(self, runner):
        """Test purge --help shows usage information."""
        result = runner.invoke(cli, ["purge", "--help"])

        assert result.exit_code == 0
        assert "--models" in result.output
        assert "--datasets" in result.output
        assert "--all" in result.output
        assert "--yes" in result.output

    @patch("huggingface_hub.scan_cache_dir")
    def test_purge_empty_cache(self, mock_scan, runner):
        """Test purge with empty cache."""
        mock_cache_info = MagicMock()
        mock_cache_info.repos = []
        mock_scan.return_value = mock_cache_info

        result = runner.invoke(cli, ["purge", "--all"])

        assert result.exit_code == 0
        assert "No cached data found to purge" in result.output

    @patch("huggingface_hub.scan_cache_dir")
    def test_purge_models_only(self, mock_scan, runner):
        """Test purge with --models flag shows only models."""
        mock_model_repo = MagicMock()
        mock_model_repo.repo_type = "model"
        mock_model_repo.repo_id = "test/model"
        mock_model_repo.size_on_disk = 1024 * 1024 * 100  # 100 MB

        mock_dataset_repo = MagicMock()
        mock_dataset_repo.repo_type = "dataset"
        mock_dataset_repo.repo_id = "test/dataset"
        mock_dataset_repo.size_on_disk = 1024 * 1024 * 50  # 50 MB

        mock_cache_info = MagicMock()
        mock_cache_info.repos = [mock_model_repo, mock_dataset_repo]
        mock_scan.return_value = mock_cache_info

        result = runner.invoke(cli, ["purge", "--models"], input="n\n")

        assert result.exit_code == 0
        assert "test/model" in result.output
        assert "test/dataset" not in result.output
        assert "100.0 MB" in result.output

    @patch("huggingface_hub.scan_cache_dir")
    def test_purge_datasets_only(self, mock_scan, runner):
        """Test purge with --datasets flag shows only datasets."""
        mock_model_repo = MagicMock()
        mock_model_repo.repo_type = "model"
        mock_model_repo.repo_id = "test/model"
        mock_model_repo.size_on_disk = 1024 * 1024 * 100  # 100 MB

        mock_dataset_repo = MagicMock()
        mock_dataset_repo.repo_type = "dataset"
        mock_dataset_repo.repo_id = "test/dataset"
        mock_dataset_repo.size_on_disk = 1024 * 1024 * 50  # 50 MB

        mock_cache_info = MagicMock()
        mock_cache_info.repos = [mock_model_repo, mock_dataset_repo]
        mock_scan.return_value = mock_cache_info

        result = runner.invoke(cli, ["purge", "--datasets"], input="n\n")

        assert result.exit_code == 0
        assert "test/model" not in result.output
        assert "test/dataset" in result.output
        assert "50.0 MB" in result.output

    @patch("huggingface_hub.scan_cache_dir")
    def test_purge_all(self, mock_scan, runner):
        """Test purge with --all flag shows both models and datasets."""
        mock_model_repo = MagicMock()
        mock_model_repo.repo_type = "model"
        mock_model_repo.repo_id = "test/model"
        mock_model_repo.size_on_disk = 1024 * 1024 * 100  # 100 MB

        mock_dataset_repo = MagicMock()
        mock_dataset_repo.repo_type = "dataset"
        mock_dataset_repo.repo_id = "test/dataset"
        mock_dataset_repo.size_on_disk = 1024 * 1024 * 50  # 50 MB

        mock_cache_info = MagicMock()
        mock_cache_info.repos = [mock_model_repo, mock_dataset_repo]
        mock_scan.return_value = mock_cache_info

        result = runner.invoke(cli, ["purge", "--all"], input="n\n")

        assert result.exit_code == 0
        assert "test/model" in result.output
        assert "test/dataset" in result.output
        assert "150.0 MB" in result.output  # Total size

    @patch("huggingface_hub.scan_cache_dir")
    def test_purge_abort_on_no_confirmation(self, mock_scan, runner):
        """Test that purge aborts when user says no to confirmation."""
        mock_repo = MagicMock()
        mock_repo.repo_type = "model"
        mock_repo.repo_id = "test/model"
        mock_repo.size_on_disk = 1024 * 1024 * 100

        mock_cache_info = MagicMock()
        mock_cache_info.repos = [mock_repo]
        mock_scan.return_value = mock_cache_info

        result = runner.invoke(cli, ["purge", "--models"], input="n\n")

        assert result.exit_code == 0
        assert "Aborted" in result.output

    @patch("shutil.rmtree")
    @patch("huggingface_hub.scan_cache_dir")
    def test_purge_with_yes_flag_skips_confirmation(self, mock_scan, mock_rmtree, runner):
        """Test that -y flag skips confirmation prompt."""
        mock_revision = MagicMock()
        mock_revision.snapshot_path = "/tmp/test_cache/models--test--model/snapshot"

        mock_repo = MagicMock()
        mock_repo.repo_type = "model"
        mock_repo.repo_id = "test/model"
        mock_repo.size_on_disk = 1024 * 1024 * 100
        mock_repo.revisions = [mock_revision]

        mock_cache_info = MagicMock()
        mock_cache_info.repos = [mock_repo]
        mock_cache_info.cache_dir = "/tmp/test_cache"
        mock_scan.return_value = mock_cache_info

        result = runner.invoke(cli, ["purge", "--models", "-y"])

        assert result.exit_code == 0
        assert "Successfully purged" in result.output
        assert "Are you sure" not in result.output

    @patch("shutil.rmtree")
    @patch("huggingface_hub.scan_cache_dir")
    def test_purge_confirms_deletion(self, mock_scan, mock_rmtree, runner):
        """Test that purge confirms and performs deletion when user says yes."""
        mock_revision = MagicMock()
        mock_revision.snapshot_path = "/tmp/test_cache/models--test--model/snapshot"

        mock_repo = MagicMock()
        mock_repo.repo_type = "model"
        mock_repo.repo_id = "test/model"
        mock_repo.size_on_disk = 1024 * 1024 * 100
        mock_repo.revisions = [mock_revision]

        mock_cache_info = MagicMock()
        mock_cache_info.repos = [mock_repo]
        mock_cache_info.cache_dir = "/tmp/test_cache"
        mock_scan.return_value = mock_cache_info

        result = runner.invoke(cli, ["purge", "--models"], input="y\n")

        assert result.exit_code == 0
        assert "Successfully purged" in result.output
        assert "1 items" in result.output
        mock_rmtree.assert_called()

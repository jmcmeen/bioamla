"""
Integration tests for bioamla CLI.
"""

import pytest
from click.testing import CliRunner

from bioamla.cli import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestCLIBasic:
    """Basic CLI tests."""

    def test_cli_help(self, runner):
        """Test that CLI help works."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "bioamla" in result.output.lower() or "Usage" in result.output

    def test_version_command(self, runner):
        """Test version command if it exists."""
        result = runner.invoke(cli, ["version"])

        # Command may or may not exist, so we just check it doesn't crash badly
        assert result.exit_code in [0, 2]  # 2 = command not found

    def test_devices_command(self, runner):
        """Test devices command."""
        result = runner.invoke(cli, ["devices"])

        # Should complete without error
        assert result.exit_code == 0


class TestCLICommandGroups:
    """Tests for CLI command groups."""

    def test_ast_group_exists(self, runner):
        """Test that ast command group exists."""
        result = runner.invoke(cli, ["ast", "--help"])

        assert result.exit_code == 0
        assert "train" in result.output
        assert "predict" in result.output
        assert "evaluate" in result.output

    def test_audio_group_exists(self, runner):
        """Test that audio command group exists."""
        result = runner.invoke(cli, ["audio", "--help"])

        assert result.exit_code == 0
        assert "list" in result.output
        assert "info" in result.output
        assert "convert" in result.output

    def test_inat_group_exists(self, runner):
        """Test that inat command group exists."""
        result = runner.invoke(cli, ["inat", "--help"])

        assert result.exit_code == 0
        assert "download" in result.output
        assert "search" in result.output
        assert "stats" in result.output

    def test_dataset_group_exists(self, runner):
        """Test that dataset command group exists."""
        result = runner.invoke(cli, ["dataset", "--help"])

        assert result.exit_code == 0
        assert "merge" in result.output


@pytest.mark.integration
class TestCLIDatasets:
    """Integration tests for dataset commands."""

    def test_validate_nonexistent_path(self, runner, temp_dir):
        """Test validate command with nonexistent path."""
        nonexistent = str(temp_dir / "nonexistent")
        result = runner.invoke(cli, ["validate", nonexistent])

        # Should fail gracefully
        assert result.exit_code != 0 or "error" in result.output.lower()


@pytest.mark.integration
class TestCLIInat:
    """Integration tests for iNaturalist commands."""

    @pytest.mark.slow
    def test_inat_search_help(self, runner):
        """Test inat search command help."""
        result = runner.invoke(cli, ["inat", "search", "--help"])

        assert result.exit_code == 0
        assert "taxa" in result.output.lower() or "Usage" in result.output


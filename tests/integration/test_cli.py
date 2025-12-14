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
    def test_inat_taxa_search_help(self, runner):
        """Test inat-taxa-search command help."""
        result = runner.invoke(cli, ["inat-taxa-search", "--help"])

        assert result.exit_code == 0
        assert "taxa" in result.output.lower() or "Usage" in result.output

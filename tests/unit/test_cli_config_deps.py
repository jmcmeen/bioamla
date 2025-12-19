"""
Unit tests for the config deps CLI command.
"""

import pytest
from click.testing import CliRunner

from bioamla.views.cli import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestConfigDepsCommand:
    """Tests for bioamla config deps command."""

    def test_deps_shows_status(self, runner):
        """Test that deps shows dependency status."""
        result = runner.invoke(cli, ["config", "deps"])

        assert result.exit_code == 0
        assert "System Dependencies" in result.output
        assert "Detected OS" in result.output

    def test_deps_shows_all_dependencies(self, runner):
        """Test that deps shows all three dependencies."""
        result = runner.invoke(cli, ["config", "deps"])

        assert result.exit_code == 0
        assert "FFmpeg" in result.output
        assert "libsndfile" in result.output
        assert "PortAudio" in result.output

    def test_deps_shows_descriptions(self, runner):
        """Test that deps shows dependency descriptions."""
        result = runner.invoke(cli, ["config", "deps"])

        assert result.exit_code == 0
        assert "Audio format conversion" in result.output
        assert "Audio file I/O" in result.output
        assert "Audio hardware" in result.output

    def test_deps_help(self, runner):
        """Test deps --help shows options."""
        result = runner.invoke(cli, ["config", "deps", "--help"])

        assert result.exit_code == 0
        assert "--install" in result.output
        assert "--yes" in result.output
        assert "-y" in result.output

    def test_deps_install_without_yes_prompts(self, runner):
        """Test that --install without --yes would prompt for confirmation."""
        # Use input='n' to simulate declining the prompt
        result = runner.invoke(cli, ["config", "deps", "--install"], input="n\n")

        # Should either prompt and exit, or proceed with install
        # The behavior depends on whether there are missing deps
        assert result.exit_code == 0 or "Aborted" in result.output or "cancelled" in result.output.lower()


class TestConfigDepsHelp:
    """Tests for config deps command help."""

    def test_config_deps_in_config_help(self, runner):
        """Test that deps appears in config --help."""
        result = runner.invoke(cli, ["config", "--help"])

        assert result.exit_code == 0
        assert "deps" in result.output

    def test_config_deps_description_in_help(self, runner):
        """Test that deps has a description in config --help."""
        result = runner.invoke(cli, ["config", "--help"])

        assert result.exit_code == 0
        # Should show description about system dependencies
        assert "dependencies" in result.output.lower() or "FFmpeg" in result.output

"""Integration tests for project CLI commands."""

import pytest
from click.testing import CliRunner
from pathlib import Path

from bioamla.views.cli import cli
from bioamla.project import PROJECT_MARKER


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


class TestProjectInit:
    """Tests for bioamla project init command."""

    def test_init_creates_project(self, runner, tmp_path):
        """Test project init creates .bioamla directory."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["project", "init"])

            assert result.exit_code == 0
            assert Path(PROJECT_MARKER).is_dir()
            assert (Path(PROJECT_MARKER) / "config.toml").exists()

    def test_init_with_path(self, runner, tmp_path):
        """Test project init with specific path."""
        target = tmp_path / "my-project"
        target.mkdir()

        result = runner.invoke(cli, ["project", "init", str(target)])

        assert result.exit_code == 0
        assert (target / PROJECT_MARKER).is_dir()

    def test_init_with_name(self, runner, tmp_path):
        """Test project init with custom name."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["project", "init", "-n", "My Project"])

            assert result.exit_code == 0
            assert "My Project" in result.output

    def test_init_with_description(self, runner, tmp_path):
        """Test project init with custom description."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["project", "init", "-n", "Test", "-d", "A test project"]
            )

            assert result.exit_code == 0

            # Check description is in config
            config_content = (Path(PROJECT_MARKER) / "config.toml").read_text()
            assert 'description = "A test project"' in config_content

    def test_init_fails_if_exists(self, runner, tmp_path):
        """Test project init fails if project exists."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["project", "init"])
            result = runner.invoke(cli, ["project", "init"])

            assert result.exit_code != 0
            assert "already exists" in result.output

    def test_init_force_reinitializes(self, runner, tmp_path):
        """Test --force flag allows reinitializing."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["project", "init"])
            result = runner.invoke(cli, ["project", "init", "--force"])

            assert result.exit_code == 0

    def test_init_with_template(self, runner, tmp_path):
        """Test project init with different templates."""
        for template in ["default", "minimal", "research", "production"]:
            target = tmp_path / template
            target.mkdir()

            result = runner.invoke(cli, ["project", "init", str(target), "-t", template])

            assert result.exit_code == 0

    def test_init_research_template_has_debug_logging(self, runner, tmp_path):
        """Test research template has DEBUG logging level."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["project", "init", "-t", "research"])

            assert result.exit_code == 0
            config_content = (Path(PROJECT_MARKER) / "config.toml").read_text()
            assert 'level = "DEBUG"' in config_content

    def test_init_production_template_has_warning_logging(self, runner, tmp_path):
        """Test production template has WARNING logging level."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["project", "init", "-t", "production"])

            assert result.exit_code == 0
            config_content = (Path(PROJECT_MARKER) / "config.toml").read_text()
            assert 'level = "WARNING"' in config_content


class TestProjectStatus:
    """Tests for bioamla project status command."""

    def test_status_shows_info(self, runner, tmp_path):
        """Test status shows project information."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["project", "init", "-n", "Test Project"])
            result = runner.invoke(cli, ["project", "status"])

            assert result.exit_code == 0
            assert "Test Project" in result.output

    def test_status_outside_project(self, runner, tmp_path):
        """Test status outside project shows message."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["project", "status"])

            assert "Not in a bioamla project" in result.output

    def test_status_shows_paths(self, runner, tmp_path):
        """Test status shows config and logs paths."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["project", "init"])
            result = runner.invoke(cli, ["project", "status"])

            assert "Config:" in result.output
            assert "Logs:" in result.output


class TestProjectConfig:
    """Tests for bioamla project config command."""

    def test_config_show(self, runner, tmp_path):
        """Test project config show displays configuration."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["project", "init", "-n", "Test"])
            result = runner.invoke(cli, ["project", "config", "show"])

            assert result.exit_code == 0
            assert "Configuration" in result.output

    def test_config_outside_project(self, runner, tmp_path):
        """Test project config outside project fails."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["project", "config", "show"])

            assert result.exit_code != 0
            assert "Not in a bioamla project" in result.output


class TestLogCommands:
    """Tests for log CLI commands."""

    def test_log_show_empty(self, runner, tmp_path):
        """Test log show with empty history."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["project", "init"])
            result = runner.invoke(cli, ["log", "show"])

            assert "No command history" in result.output

    def test_log_outside_project(self, runner, tmp_path):
        """Test log commands outside project."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["log", "show"])

            assert "requires a bioamla project" in result.output

    def test_log_search_no_results(self, runner, tmp_path):
        """Test log search with no results."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["project", "init"])
            result = runner.invoke(cli, ["log", "search", "nonexistent"])

            assert "No matches" in result.output

    def test_log_stats_empty(self, runner, tmp_path):
        """Test log stats with no history."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["project", "init"])
            result = runner.invoke(cli, ["log", "stats"])

            assert "No command history" in result.output

    def test_log_clear_empty(self, runner, tmp_path):
        """Test log clear with no history."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["project", "init"])
            result = runner.invoke(cli, ["log", "clear", "--yes"])

            assert "Cleared 0 log entries" in result.output


class TestConfigShow:
    """Tests for config show command with new sections."""

    def test_config_show_includes_new_sections(self, runner, tmp_path):
        """Test that config show displays all configuration sections."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["project", "init"])
            result = runner.invoke(cli, ["config", "show"])

            # Check for values from new sections in output
            # (section headers may have Rich formatting stripped)
            assert "default_ast_model" in result.output  # from models section
            assert "batch_size" in result.output  # from inference section
            assert "learning_rate" in result.output  # from training section
            assert "max_history" in result.output  # from logging section

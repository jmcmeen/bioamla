"""Unit tests for project management."""

import pytest
from datetime import datetime, timezone
from pathlib import Path

from bioamla.core.project import (
    find_project_root,
    is_in_project,
    create_project,
    load_project,
    get_project_config_path,
    PROJECT_MARKER,
    CONFIG_FILENAME,
    LOGS_DIR,
    ProjectInfo,
)


class TestFindProjectRoot:
    """Tests for find_project_root function."""

    def test_finds_root_in_current_dir(self, tmp_path):
        """Test finding project root when in project directory."""
        (tmp_path / PROJECT_MARKER).mkdir()

        result = find_project_root(tmp_path)

        assert result == tmp_path

    def test_finds_root_in_parent_dir(self, tmp_path):
        """Test finding project root from subdirectory."""
        (tmp_path / PROJECT_MARKER).mkdir()
        subdir = tmp_path / "audio" / "2024"
        subdir.mkdir(parents=True)

        result = find_project_root(subdir)

        assert result == tmp_path

    def test_returns_none_when_not_in_project(self, tmp_path):
        """Test returns None when not in a project."""
        result = find_project_root(tmp_path)

        assert result is None

    def test_deep_nesting(self, tmp_path):
        """Test finding root from deeply nested directory."""
        (tmp_path / PROJECT_MARKER).mkdir()
        deep = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep.mkdir(parents=True)

        result = find_project_root(deep)

        assert result == tmp_path

    def test_stops_at_first_marker(self, tmp_path):
        """Test that search stops at first .bioamla marker found."""
        # Create nested projects
        outer = tmp_path / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)

        (outer / PROJECT_MARKER).mkdir()
        (inner / PROJECT_MARKER).mkdir()

        # From inner, should find inner's marker
        result = find_project_root(inner)
        assert result == inner

        # From a subdir of inner, should still find inner
        subdir = inner / "data"
        subdir.mkdir()
        result = find_project_root(subdir)
        assert result == inner


class TestIsInProject:
    """Tests for is_in_project function."""

    def test_returns_true_in_project(self, tmp_path):
        """Test returns True when in a project."""
        (tmp_path / PROJECT_MARKER).mkdir()

        assert is_in_project(tmp_path) is True

    def test_returns_false_outside_project(self, tmp_path):
        """Test returns False when not in a project."""
        assert is_in_project(tmp_path) is False

    def test_returns_true_from_subdirectory(self, tmp_path):
        """Test returns True from project subdirectory."""
        (tmp_path / PROJECT_MARKER).mkdir()
        subdir = tmp_path / "models"
        subdir.mkdir()

        assert is_in_project(subdir) is True


class TestCreateProject:
    """Tests for create_project function."""

    def test_creates_project_directory(self, tmp_path):
        """Test that .bioamla directory is created."""
        info = create_project(tmp_path)

        assert (tmp_path / PROJECT_MARKER).is_dir()
        assert info.root == tmp_path

    def test_creates_config_file(self, tmp_path):
        """Test that config file is created."""
        info = create_project(tmp_path)

        assert info.config_path.exists()

    def test_creates_logs_directory(self, tmp_path):
        """Test that logs directory is created."""
        info = create_project(tmp_path)

        assert info.logs_path.is_dir()

    def test_uses_directory_name_as_default_name(self, tmp_path):
        """Test project name defaults to directory name."""
        project_dir = tmp_path / "my-cool-project"
        project_dir.mkdir()

        info = create_project(project_dir)

        assert info.name == "my-cool-project"

    def test_custom_name(self, tmp_path):
        """Test custom project name."""
        info = create_project(tmp_path, name="Custom Name")

        assert info.name == "Custom Name"

    def test_custom_description(self, tmp_path):
        """Test custom project description."""
        info = create_project(tmp_path, description="A test project")

        assert info.description == "A test project"

    def test_config_contains_project_name(self, tmp_path):
        """Test that config file contains project name."""
        create_project(tmp_path, name="Test Project")

        config_content = (tmp_path / PROJECT_MARKER / CONFIG_FILENAME).read_text()
        assert 'name = "Test Project"' in config_content

    def test_config_contains_timestamp(self, tmp_path):
        """Test that config file contains created timestamp."""
        create_project(tmp_path)

        config_content = (tmp_path / PROJECT_MARKER / CONFIG_FILENAME).read_text()
        assert "created = " in config_content
        assert config_content.count('created = ""') == 0  # Should be filled in

    def test_creates_in_nonexistent_directory(self, tmp_path):
        """Test creating project in a directory that doesn't exist yet."""
        project_dir = tmp_path / "new-project"

        info = create_project(project_dir)

        assert project_dir.exists()
        assert (project_dir / PROJECT_MARKER).is_dir()
        assert info.root == project_dir

    def test_template_default(self, tmp_path):
        """Test default template creates standard config."""
        create_project(tmp_path, template="default")

        config_content = (tmp_path / PROJECT_MARKER / CONFIG_FILENAME).read_text()
        # Default template should have INFO logging
        assert 'level = "INFO"' in config_content

    def test_template_research(self, tmp_path):
        """Test research template creates research-focused config."""
        create_project(tmp_path, template="research")

        config_content = (tmp_path / PROJECT_MARKER / CONFIG_FILENAME).read_text()
        # Research template should have DEBUG logging
        assert 'level = "DEBUG"' in config_content

    def test_template_production(self, tmp_path):
        """Test production template creates production config."""
        create_project(tmp_path, template="production")

        config_content = (tmp_path / PROJECT_MARKER / CONFIG_FILENAME).read_text()
        # Production template should have WARNING logging
        assert 'level = "WARNING"' in config_content

    def test_template_minimal(self, tmp_path):
        """Test minimal template creates sparse config."""
        create_project(tmp_path, template="minimal")

        config_content = (tmp_path / PROJECT_MARKER / CONFIG_FILENAME).read_text()
        # Minimal template should be shorter
        assert "[project]" in config_content
        # Should not have actual sections (only [project]), though may have comments
        # Check that no actual section headers exist besides [project]
        import re
        actual_sections = re.findall(r'^\s*\[([^\]]+)\]', config_content, re.MULTILINE)
        assert actual_sections == ["project"], f"Expected only [project], got {actual_sections}"

    def test_invalid_template_raises(self, tmp_path):
        """Test that invalid template name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown template"):
            create_project(tmp_path, template="nonexistent")

    def test_custom_config_file(self, tmp_path):
        """Test using custom config file."""
        # Create custom config
        custom_config = tmp_path / "custom.toml"
        custom_config.write_text('[project]\nname = "From Custom"\n')

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        create_project(project_dir, config_file=custom_config)

        config_content = (project_dir / PROJECT_MARKER / CONFIG_FILENAME).read_text()
        assert 'name = "From Custom"' in config_content


class TestLoadProject:
    """Tests for load_project function."""

    def test_loads_project_info(self, tmp_path):
        """Test loading project information."""
        create_project(tmp_path, name="Test Project", description="A test")

        info = load_project(tmp_path)

        assert info is not None
        assert info.name == "Test Project"
        assert info.description == "A test"
        assert info.root == tmp_path

    def test_returns_none_outside_project(self, tmp_path):
        """Test returns None when not in a project."""
        info = load_project(tmp_path)

        assert info is None

    def test_loads_from_subdirectory(self, tmp_path):
        """Test loading project from subdirectory."""
        create_project(tmp_path, name="Test Project")
        subdir = tmp_path / "audio"
        subdir.mkdir()

        info = load_project(subdir)

        assert info is not None
        assert info.name == "Test Project"
        assert info.root == tmp_path

    def test_handles_missing_config(self, tmp_path):
        """Test handles project marker without config file."""
        (tmp_path / PROJECT_MARKER).mkdir()
        # No config file created

        info = load_project(tmp_path)

        assert info is not None
        assert info.name == tmp_path.name
        assert info.root == tmp_path

    def test_handles_malformed_config(self, tmp_path):
        """Test handles malformed config file gracefully."""
        (tmp_path / PROJECT_MARKER).mkdir()
        config_path = tmp_path / PROJECT_MARKER / CONFIG_FILENAME
        config_path.write_text("this is not valid toml [[[")

        info = load_project(tmp_path)

        assert info is not None
        assert info.root == tmp_path

    def test_loads_version(self, tmp_path):
        """Test loading version from config."""
        create_project(tmp_path)

        info = load_project(tmp_path)

        assert info.version == "1.0.0"

    def test_config_path_is_correct(self, tmp_path):
        """Test that config_path property is set correctly."""
        create_project(tmp_path)

        info = load_project(tmp_path)

        expected_path = tmp_path / PROJECT_MARKER / CONFIG_FILENAME
        assert info.config_path == expected_path

    def test_logs_path_is_correct(self, tmp_path):
        """Test that logs_path property is set correctly."""
        create_project(tmp_path)

        info = load_project(tmp_path)

        expected_path = tmp_path / PROJECT_MARKER / LOGS_DIR
        assert info.logs_path == expected_path


class TestGetProjectConfigPath:
    """Tests for get_project_config_path function."""

    def test_returns_config_path_in_project(self, tmp_path):
        """Test returns config path when in project."""
        create_project(tmp_path)

        result = get_project_config_path(tmp_path)

        assert result == tmp_path / PROJECT_MARKER / CONFIG_FILENAME

    def test_returns_none_outside_project(self, tmp_path):
        """Test returns None when not in a project."""
        result = get_project_config_path(tmp_path)

        assert result is None

    def test_returns_path_from_subdirectory(self, tmp_path):
        """Test returns config path from project subdirectory."""
        create_project(tmp_path)
        subdir = tmp_path / "data"
        subdir.mkdir()

        result = get_project_config_path(subdir)

        assert result == tmp_path / PROJECT_MARKER / CONFIG_FILENAME


class TestProjectInfo:
    """Tests for ProjectInfo dataclass."""

    def test_post_init_sets_paths(self, tmp_path):
        """Test that __post_init__ sets config_path and logs_path."""
        info = ProjectInfo(
            root=tmp_path,
            name="Test",
            version="1.0.0",
            created=datetime.now(timezone.utc),
        )

        assert info.config_path == tmp_path / PROJECT_MARKER / CONFIG_FILENAME
        assert info.logs_path == tmp_path / PROJECT_MARKER / LOGS_DIR

    def test_description_defaults_to_empty(self, tmp_path):
        """Test that description defaults to empty string."""
        info = ProjectInfo(
            root=tmp_path,
            name="Test",
            version="1.0.0",
            created=datetime.now(timezone.utc),
        )

        assert info.description == ""

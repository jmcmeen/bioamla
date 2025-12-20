# tests/controllers/test_project.py
"""
Tests for ProjectController.
"""

from unittest.mock import MagicMock

import pytest

from bioamla.controllers.project import (
    ProjectController,
    ProjectSummary,
)


class TestProjectController:
    """Tests for ProjectController."""

    @pytest.fixture
    def controller(self):
        return ProjectController()

    def test_create_project_success(self, controller, tmp_path, mocker):
        """Test that creating a project succeeds."""
        # Mock the core create_project function
        mock_create = mocker.patch("bioamla.core.project.create_project")
        mock_info = MagicMock()
        mock_info.name = "test-project"
        mock_info.root = tmp_path
        mock_info.version = "1.0.0"
        mock_info.created.isoformat.return_value = "2024-01-01T00:00:00"
        mock_info.description = "Test project"
        mock_info.config_path = tmp_path / ".bioamla" / "config.toml"
        mock_info.models_path = tmp_path / ".bioamla" / "models.toml"
        mock_info.datasets_path = tmp_path / ".bioamla" / "datasets.toml"
        mock_info.logs_path = tmp_path / ".bioamla" / "logs"
        mock_info.runs_path = tmp_path / ".bioamla" / "runs"
        mock_info.cache_path = tmp_path / ".bioamla" / "cache"
        mock_create.return_value = mock_info

        result = controller.create(str(tmp_path), name="test-project")

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, ProjectSummary)
        assert result.data.name == "test-project"

    def test_create_existing_project_fails(self, controller, tmp_path, mocker):
        """Test that creating over existing project fails without force."""
        # Simulate existing project marker
        marker = tmp_path / ".bioamla"
        marker.mkdir()

        result = controller.create(str(tmp_path), name="test-project")

        assert result.success is False
        assert "already exists" in result.error

    def test_load_project_not_found_fails(self, controller, tmp_path, mocker):
        """Test that loading non-existent project fails."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_load.return_value = None

        result = controller.load(str(tmp_path))

        assert result.success is False
        assert "Not in a bioamla project" in result.error


class TestProjectTemplates:
    """Tests for project template methods."""

    @pytest.fixture
    def controller(self):
        return ProjectController()

    def test_list_templates_returns_list(self, controller):
        """Test that listing templates returns available templates."""
        result = controller.list_templates()

        assert result.success is True
        assert result.data is not None
        assert "default" in result.data
        assert "minimal" in result.data

    def test_describe_template_success(self, controller):
        """Test that describing a template returns description."""
        result = controller.describe_template("default")

        assert result.success is True
        assert result.data is not None
        assert "description" in result.data

    def test_describe_unknown_template_fails(self, controller):
        """Test that describing unknown template fails."""
        result = controller.describe_template("nonexistent_template")

        assert result.success is False
        assert "Unknown template" in result.error


class TestProjectStatistics:
    """Tests for project statistics methods."""

    @pytest.fixture
    def controller(self):
        return ProjectController()

    def test_get_stats_not_in_project_fails(self, controller, tmp_path, mocker):
        """Test that getting stats outside project fails."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_load.return_value = None

        result = controller.get_stats(str(tmp_path))

        assert result.success is False
        assert "Not in a bioamla project" in result.error

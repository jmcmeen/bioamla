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

    def test_get_stats_success(self, controller, tmp_path, mocker):
        """Test that getting stats succeeds for a valid project."""
        from pathlib import Path

        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.name = "test-project"
        mock_info.root = Path(tmp_path)
        mock_load.return_value = mock_info

        mock_logger = mocker.patch("bioamla.core.log.CommandLogger")
        mock_logger_instance = MagicMock()
        mock_logger_instance.get_stats.return_value = {"total_commands": 5}
        mock_logger_instance.get_history.return_value = []
        mock_logger.return_value = mock_logger_instance

        result = controller.get_stats(str(tmp_path))

        assert result.success is True
        assert result.data.name == "test-project"


class TestGetConfig:
    """Tests for get_config method."""

    @pytest.fixture
    def controller(self):
        return ProjectController()

    def test_get_config_not_in_project_fails(self, controller, tmp_path, mocker):
        """Test that get_config fails outside project."""
        mock_get_path = mocker.patch("bioamla.core.project.get_project_config_path")
        mock_get_path.return_value = None

        result = controller.get_config(str(tmp_path))

        assert result.success is False
        assert "Not in a bioamla project" in result.error

    def test_get_config_success(self, controller, tmp_path, mocker):
        """Test that get_config returns config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[project]\nname = "test"')

        mock_get_path = mocker.patch("bioamla.core.project.get_project_config_path")
        mock_get_path.return_value = config_path

        mock_load = mocker.patch("bioamla.core.config.load_toml")
        mock_load.return_value = {"project": {"name": "test"}}

        result = controller.get_config(str(tmp_path))

        assert result.success is True
        assert "project" in result.data.sections


class TestUpdateConfig:
    """Tests for update_config method."""

    @pytest.fixture
    def controller(self):
        return ProjectController()

    def test_update_config_success(self, controller, tmp_path, mocker):
        """Test that update_config succeeds."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[project]\nname = "test"')

        mock_get_path = mocker.patch("bioamla.core.project.get_project_config_path")
        mock_get_path.return_value = config_path

        mock_load = mocker.patch("bioamla.core.config.load_toml")
        mock_load.return_value = {"project": {"name": "test"}}

        mock_save = mocker.patch("bioamla.core.config.save_toml")

        result = controller.update_config({"project": {"version": "1.0.0"}}, str(tmp_path))

        assert result.success is True
        mock_save.assert_called_once()


class TestModelRegistry:
    """Tests for model registry methods."""

    @pytest.fixture
    def controller(self):
        return ProjectController()

    def test_list_models_success(self, controller, tmp_path, mocker):
        """Test that list_models returns models."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.models_path = tmp_path / "models.toml"
        mock_info.models_path.write_text('[models]\n')
        mock_load.return_value = mock_info

        mock_load_toml = mocker.patch("bioamla.core.config.load_toml")
        mock_load_toml.return_value = {
            "models": {
                "test-model": {"id": "hf/model", "type": "ast", "description": "Test"}
            }
        }

        result = controller.list_models(str(tmp_path))

        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0].name == "test-model"

    def test_register_model_success(self, controller, tmp_path, mocker):
        """Test that register_model succeeds."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.models_path = tmp_path / "models.toml"
        mock_load.return_value = mock_info

        mock_load_toml = mocker.patch("bioamla.core.config.load_toml")
        mock_load_toml.return_value = {}

        mock_save_toml = mocker.patch("bioamla.core.config.save_toml")

        result = controller.register_model(
            name="my-model",
            model_id="hf/test-model",
            model_type="ast",
            path=str(tmp_path),
        )

        assert result.success is True
        assert result.data.name == "my-model"


class TestDatasetRegistry:
    """Tests for dataset registry methods."""

    @pytest.fixture
    def controller(self):
        return ProjectController()

    def test_list_datasets_success(self, controller, tmp_path, mocker):
        """Test that list_datasets returns datasets."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.datasets_path = tmp_path / "datasets.toml"
        mock_info.datasets_path.write_text('[datasets]\n')
        mock_load.return_value = mock_info

        mock_load_toml = mocker.patch("bioamla.core.config.load_toml")
        mock_load_toml.return_value = {
            "datasets": {
                "test-dataset": {"path": "/data", "source": "local"}
            }
        }

        result = controller.list_datasets(str(tmp_path))

        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0].name == "test-dataset"

    def test_register_dataset_success(self, controller, tmp_path, mocker):
        """Test that register_dataset succeeds."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.datasets_path = tmp_path / "datasets.toml"
        mock_load.return_value = mock_info

        mock_load_toml = mocker.patch("bioamla.core.config.load_toml")
        mock_load_toml.return_value = {}

        mock_save_toml = mocker.patch("bioamla.core.config.save_toml")

        result = controller.register_dataset(
            name="my-dataset",
            dataset_path="/data/audio",
            source="local",
            path=str(tmp_path),
        )

        assert result.success is True
        assert result.data.name == "my-dataset"


class TestRunManagement:
    """Tests for run management methods."""

    @pytest.fixture
    def controller(self):
        return ProjectController()

    def test_create_run_success(self, controller, tmp_path, mocker):
        """Test that create_run succeeds."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.runs_path = tmp_path / "runs"
        mock_load.return_value = mock_info

        result = controller.create_run(
            name="Test Run",
            action="predict",
            input_path="/input",
            output_path="/output",
            path=str(tmp_path),
        )

        assert result.success is True
        assert result.data.name == "Test Run"
        assert result.data.action == "predict"

    def test_list_runs_success(self, controller, tmp_path, mocker):
        """Test that list_runs returns runs."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.runs_path = tmp_path / "runs"
        mock_info.runs_path.mkdir(parents=True)
        mock_load.return_value = mock_info

        # Create a mock run
        run_dir = mock_info.runs_path / "test_run_123"
        run_dir.mkdir()
        run_file = run_dir / "run.json"
        import json
        run_file.write_text(json.dumps({
            "run_id": "test_run_123",
            "name": "Test Run",
            "started": "2024-01-01T00:00:00",
            "status": "completed",
        }))

        result = controller.list_runs(path=str(tmp_path))

        assert result.success is True
        assert len(result.data) == 1

    def test_get_run_success(self, controller, tmp_path, mocker):
        """Test that get_run returns run info."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.runs_path = tmp_path / "runs"
        mock_info.runs_path.mkdir(parents=True)
        mock_load.return_value = mock_info

        # Create a mock run
        run_dir = mock_info.runs_path / "test_run_123"
        run_dir.mkdir()
        run_file = run_dir / "run.json"
        import json
        run_file.write_text(json.dumps({
            "run_id": "test_run_123",
            "name": "Test Run",
            "started": "2024-01-01T00:00:00",
            "status": "completed",
        }))

        result = controller.get_run("test_run_123", path=str(tmp_path))

        assert result.success is True
        assert result.data.run_id == "test_run_123"

    def test_get_run_not_found_fails(self, controller, tmp_path, mocker):
        """Test that get_run fails for nonexistent run."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.runs_path = tmp_path / "runs"
        mock_load.return_value = mock_info

        result = controller.get_run("nonexistent", path=str(tmp_path))

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_complete_run_success(self, controller, tmp_path, mocker):
        """Test that complete_run succeeds."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.runs_path = tmp_path / "runs"
        mock_info.runs_path.mkdir(parents=True)
        mock_load.return_value = mock_info

        # Create a mock run
        run_dir = mock_info.runs_path / "test_run_123"
        run_dir.mkdir()
        run_file = run_dir / "run.json"
        import json
        run_file.write_text(json.dumps({
            "run_id": "test_run_123",
            "name": "Test Run",
            "started": "2024-01-01T00:00:00",
            "status": "running",
        }))

        result = controller.complete_run(
            "test_run_123",
            status="completed",
            results={"count": 10},
            path=str(tmp_path),
        )

        assert result.success is True
        assert result.data.status == "completed"


class TestCacheManagement:
    """Tests for cache management methods."""

    @pytest.fixture
    def controller(self):
        return ProjectController()

    def test_get_cache_stats_success(self, controller, tmp_path, mocker):
        """Test that get_cache_stats returns cache statistics."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.embeddings_cache = tmp_path / "cache" / "embeddings"
        mock_info.models_cache = tmp_path / "cache" / "models"
        mock_info.temp_cache = tmp_path / "cache" / "temp"
        mock_load.return_value = mock_info

        # Create cache directories
        mock_info.embeddings_cache.mkdir(parents=True)
        mock_info.models_cache.mkdir(parents=True)
        mock_info.temp_cache.mkdir(parents=True)

        result = controller.get_cache_stats(str(tmp_path))

        assert result.success is True
        assert "total_size_mb" in result.data
        assert "embeddings" in result.data

    def test_clear_cache_success(self, controller, tmp_path, mocker):
        """Test that clear_cache clears cache files."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.embeddings_cache = tmp_path / "cache" / "embeddings"
        mock_info.models_cache = tmp_path / "cache" / "models"
        mock_info.temp_cache = tmp_path / "cache" / "temp"
        mock_load.return_value = mock_info

        # Create cache directories with files
        mock_info.temp_cache.mkdir(parents=True)
        (mock_info.temp_cache / "temp_file.txt").write_text("test")

        result = controller.clear_cache(cache_type="temp", path=str(tmp_path))

        assert result.success is True
        assert result.data["temp"] == 1


class TestProjectExistsAndFindRoot:
    """Tests for exists and find_root methods."""

    @pytest.fixture
    def controller(self):
        return ProjectController()

    def test_exists_in_project(self, controller, tmp_path, mocker):
        """Test exists returns True when in project."""
        mock_is_in = mocker.patch("bioamla.core.project.is_in_project")
        mock_is_in.return_value = True

        result = controller.exists(str(tmp_path))

        assert result.success is True
        assert result.data is True

    def test_exists_not_in_project(self, controller, tmp_path, mocker):
        """Test exists returns False when not in project."""
        mock_is_in = mocker.patch("bioamla.core.project.is_in_project")
        mock_is_in.return_value = False

        result = controller.exists(str(tmp_path))

        assert result.success is True
        assert result.data is False

    def test_find_root_success(self, controller, tmp_path, mocker):
        """Test find_root returns project root."""
        mock_find = mocker.patch("bioamla.core.project.find_project_root")
        mock_find.return_value = tmp_path

        result = controller.find_root(str(tmp_path))

        assert result.success is True
        assert result.data == str(tmp_path)

    def test_find_root_not_in_project(self, controller, tmp_path, mocker):
        """Test find_root fails when not in project."""
        mock_find = mocker.patch("bioamla.core.project.find_project_root")
        mock_find.return_value = None

        result = controller.find_root(str(tmp_path))

        assert result.success is False
        assert "Not in a bioamla project" in result.error


class TestResetConfig:
    """Tests for reset_config method."""

    @pytest.fixture
    def controller(self):
        return ProjectController()

    def test_reset_config_success(self, controller, tmp_path, mocker):
        """Test that reset_config resets to template."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.name = "test-project"
        mock_info.description = "Test"
        mock_info.config_path = tmp_path / "config.toml"
        mock_load.return_value = mock_info

        mock_get_template = mocker.patch("bioamla.core.project._get_template_content")
        mock_get_template.return_value = "[project]\nname = '{{ name }}'"

        mock_customize = mocker.patch("bioamla.core.project._customize_template")
        mock_customize.return_value = "[project]\nname = 'test-project'"

        mock_get_config = mocker.patch.object(controller, "get_config")
        mock_get_config.return_value = MagicMock(success=True)

        result = controller.reset_config(template="default", path=str(tmp_path))

        assert result.success is True

    def test_reset_config_invalid_template_fails(self, controller, tmp_path, mocker):
        """Test that reset_config fails with invalid template."""
        mock_load = mocker.patch("bioamla.core.project.load_project")
        mock_info = MagicMock()
        mock_info.name = "test-project"
        mock_load.return_value = mock_info

        mock_get_template = mocker.patch("bioamla.core.project._get_template_content")
        mock_get_template.side_effect = ValueError("Unknown template")

        result = controller.reset_config(template="nonexistent", path=str(tmp_path))

        assert result.success is False
        assert "Invalid template" in result.error


class TestProjectSummary:
    """Tests for ProjectSummary dataclass."""

    def test_project_summary_to_dict(self):
        """Test ProjectSummary to_dict conversion."""
        summary = ProjectSummary(
            name="test",
            root="/test",
            version="1.0.0",
            created="2024-01-01T00:00:00",
            description="Test project",
            config_path="/test/.bioamla/config.toml",
            models_path="/test/.bioamla/models.toml",
            datasets_path="/test/.bioamla/datasets.toml",
            logs_path="/test/.bioamla/logs",
            runs_path="/test/.bioamla/runs",
            cache_path="/test/.bioamla/cache",
        )

        d = summary.to_dict()

        assert d["name"] == "test"
        assert d["version"] == "1.0.0"


class TestProjectStatisticsDataclass:
    """Tests for ProjectStatistics dataclass."""

    def test_project_statistics_fields(self):
        """Test ProjectStatistics has all fields."""
        from bioamla.controllers.project import ProjectStatistics

        stats = ProjectStatistics(
            name="test",
            root="/test",
            audio_files=100,
            total_size_mb=50.5,
            datasets=["dataset1", "dataset2"],
            registered_models=3,
            registered_datasets=2,
            run_count=5,
            cache_size_mb=10.2,
            has_metadata=True,
            command_count=20,
            last_command="predict",
        )

        assert stats.audio_files == 100
        assert stats.total_size_mb == 50.5
        assert len(stats.datasets) == 2

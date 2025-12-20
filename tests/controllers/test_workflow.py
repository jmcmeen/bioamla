# tests/controllers/test_workflow.py
"""
Tests for WorkflowController.
"""

from unittest.mock import MagicMock, Mock

import pytest

from bioamla.controllers.workflow import (
    ValidationSummary,
    WorkflowController,
    WorkflowSummary,
)


class TestWorkflowController:
    """Tests for WorkflowController."""

    @pytest.fixture
    def controller(self):
        return WorkflowController()

    @pytest.fixture
    def sample_workflow_toml(self, tmp_path):
        """Create a sample workflow TOML file."""
        workflow_content = """[workflow]
name = "test_workflow"
description = "A test workflow"
version = "1.0.0"

[variables]
input_dir = "./input"
output_dir = "./output"

[[steps]]
name = "step1"
action = "audio.resample"

[steps.params]
input = "{{ input_dir }}"
output = "{{ output_dir }}"
"""
        workflow_file = tmp_path / "workflow.toml"
        workflow_file.write_text(workflow_content)
        return str(workflow_file)

    def test_parse_valid_workflow_success(self, controller, sample_workflow_toml, mocker):
        """Test that parsing a valid workflow succeeds."""
        mock_parse = mocker.patch("bioamla.core.workflow.parser.parse_workflow")
        mock_workflow = MagicMock()
        mock_workflow.name = "test_workflow"
        mock_workflow.description = "A test workflow"
        mock_workflow.version = "1.0.0"
        mock_workflow.steps = [MagicMock(name="step1")]
        mock_workflow.steps[0].name = "step1"
        mock_workflow.variables = {"input_dir": "./input"}
        mock_workflow.get_execution_order.return_value = ["step1"]
        mock_parse.return_value = mock_workflow

        result = controller.parse(sample_workflow_toml)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, WorkflowSummary)
        assert result.data.name == "test_workflow"

    def test_parse_nonexistent_file_fails(self, controller):
        """Test that parsing nonexistent file fails."""
        result = controller.parse("/nonexistent/workflow.toml")

        assert result.success is False
        assert "does not exist" in result.error


class TestWorkflowValidation:
    """Tests for workflow validation methods."""

    @pytest.fixture
    def controller(self):
        return WorkflowController()

    def test_validate_valid_workflow_success(self, controller, tmp_path, mocker):
        """Test that validating a valid workflow succeeds."""
        workflow_file = tmp_path / "workflow.toml"
        workflow_file.write_text(
            '[workflow]\nname = "test"\n[[steps]]\nname = "s1"\naction = "audio.resample"'
        )

        mock_parse = mocker.patch("bioamla.core.workflow.parser.parse_workflow")
        mock_workflow = MagicMock()
        mock_parse.return_value = mock_workflow

        mock_validate = mocker.patch("bioamla.core.workflow.validator.validate_workflow")
        mock_result = MagicMock()
        mock_result.valid = True
        mock_result.errors = []
        mock_result.warnings = []
        mock_validate.return_value = mock_result

        result = controller.validate(str(workflow_file))

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, ValidationSummary)
        assert result.data.valid is True

    def test_validate_invalid_workflow_returns_errors(self, controller, tmp_path, mocker):
        """Test that validating invalid workflow returns errors."""
        workflow_file = tmp_path / "workflow.toml"
        workflow_file.write_text('[workflow]\nname = "test"')

        mock_parse = mocker.patch("bioamla.core.workflow.parser.parse_workflow")
        mock_workflow = MagicMock()
        mock_parse.return_value = mock_workflow

        mock_validate = mocker.patch("bioamla.core.workflow.validator.validate_workflow")
        mock_result = MagicMock()
        mock_result.valid = False
        mock_result.errors = [MagicMock(__str__=lambda x: "Missing steps")]
        mock_result.warnings = []
        mock_validate.return_value = mock_result

        result = controller.validate(str(workflow_file))

        assert result.success is True
        assert result.data is not None
        assert result.data.valid is False
        assert result.data.num_errors > 0


class TestWorkflowExecution:
    """Tests for workflow execution methods."""

    @pytest.fixture
    def controller(self):
        return WorkflowController()

    def test_list_actions_returns_list(self, controller, mocker):
        """Test that listing actions returns action names."""
        mock_engine = MagicMock()
        mock_engine._actions = {
            "audio.resample": Mock(),
            "analysis.indices": Mock(),
        }
        mocker.patch.object(controller, "_get_engine", return_value=mock_engine)

        result = controller.list_actions()

        assert result.success is True
        assert result.data is not None
        assert "audio.resample" in result.data

    def test_get_example_workflow_returns_toml(self, controller):
        """Test that get_example_workflow returns TOML content."""
        result = controller.get_example_workflow()

        assert result.success is True
        assert result.data is not None
        assert "[workflow]" in result.data
        assert "[[steps]]" in result.data

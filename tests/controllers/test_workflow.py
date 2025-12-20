# tests/controllers/test_workflow.py
"""
Tests for WorkflowController.
"""

from unittest.mock import MagicMock, Mock

import pytest

from bioamla.controllers.workflow import (
    ExecutionSummary,
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


class TestWorkflowExecuteFull:
    """Tests for full workflow execution."""

    @pytest.fixture
    def controller(self):
        return WorkflowController()

    @pytest.fixture
    def sample_workflow_file(self, tmp_path):
        """Create a sample workflow file."""
        workflow_content = """[workflow]
name = "test_execution"
description = "Test execution workflow"
version = "1.0.0"

[[steps]]
name = "step1"
action = "audio.resample"
"""
        workflow_file = tmp_path / "workflow.toml"
        workflow_file.write_text(workflow_content)
        return str(workflow_file)

    def test_execute_success(self, controller, sample_workflow_file, mocker):
        """Test that execute succeeds."""
        # Mock parser
        mock_parse = mocker.patch("bioamla.core.workflow.parser.parse_workflow")
        mock_workflow = MagicMock()
        mock_workflow.name = "test_execution"
        mock_workflow.steps = []
        mock_parse.return_value = mock_workflow

        # Mock validator
        mock_validate_fn = mocker.patch("bioamla.core.workflow.validator.validate_workflow")
        mock_val_result = MagicMock()
        mock_val_result.valid = True
        mock_val_result.errors = []
        mock_val_result.warnings = []
        mock_validate_fn.return_value = mock_val_result

        # Mock engine
        mock_engine = MagicMock()
        mock_exec_result = MagicMock()
        mock_exec_result.workflow_name = "test_execution"
        mock_exec_result.execution_id = "exec_123"
        mock_exec_result.status = MagicMock(value="completed")
        mock_exec_result.total_duration_seconds = 1.5
        mock_exec_result.steps_completed = 1
        mock_exec_result.steps_failed = 0
        mock_exec_result.step_results = []
        mock_exec_result.success = True
        mock_exec_result.error = None
        mock_exec_result.outputs = {}
        mock_engine.execute.return_value = mock_exec_result
        mocker.patch.object(controller, "_get_engine", return_value=mock_engine)

        result = controller.execute(sample_workflow_file)

        assert result.success is True
        assert result.data.workflow_name == "test_execution"
        assert result.data.status == "completed"

    def test_execute_validation_failure(self, controller, sample_workflow_file, mocker):
        """Test that execute fails when validation fails."""
        mock_parse = mocker.patch("bioamla.core.workflow.parser.parse_workflow")
        mock_workflow = MagicMock()
        mock_parse.return_value = mock_workflow

        mock_validate_fn = mocker.patch("bioamla.core.workflow.validator.validate_workflow")
        mock_val_result = MagicMock()
        mock_val_result.valid = False
        mock_val_result.errors = [MagicMock(__str__=lambda x: "Missing required field")]
        mock_val_result.warnings = []
        mock_validate_fn.return_value = mock_val_result

        result = controller.execute(sample_workflow_file, validate_first=True)

        assert result.success is False
        assert "validation failed" in result.error.lower()


class TestExportToToml:
    """Tests for export_to_toml method."""

    @pytest.fixture
    def controller(self):
        return WorkflowController()

    def test_export_to_toml_success(self, controller, mocker):
        """Test that export_to_toml succeeds."""
        mock_workflow = MagicMock()
        mock_workflow.name = "test_workflow"

        mock_to_toml = mocker.patch("bioamla.core.workflow.parser.workflow_to_toml")
        mock_to_toml.return_value = '[workflow]\nname = "test_workflow"'

        result = controller.export_to_toml(mock_workflow)

        assert result.success is True
        assert "[workflow]" in result.data

    def test_export_to_toml_saves_file(self, controller, tmp_path, mocker):
        """Test that export_to_toml saves to file."""
        mock_workflow = MagicMock()
        mock_workflow.name = "test_workflow"

        mock_to_toml = mocker.patch("bioamla.core.workflow.parser.workflow_to_toml")
        mock_to_toml.return_value = '[workflow]\nname = "test_workflow"'

        output_path = tmp_path / "exported.toml"
        result = controller.export_to_toml(mock_workflow, output_path=str(output_path))

        assert result.success is True
        assert output_path.exists()


class TestExportToShell:
    """Tests for export_to_shell method."""

    @pytest.fixture
    def controller(self):
        return WorkflowController()

    def test_export_to_shell_success(self, controller, tmp_path, mocker):
        """Test that export_to_shell succeeds."""
        workflow_file = tmp_path / "workflow.toml"
        workflow_file.write_text('[workflow]\nname = "test"')

        mock_parse = mocker.patch("bioamla.core.workflow.parser.parse_workflow")
        mock_workflow = MagicMock()
        mock_parse.return_value = mock_workflow

        mock_engine = MagicMock()
        mock_engine.export_to_shell.return_value = "#!/bin/bash\necho 'hello'"
        mocker.patch.object(controller, "_get_engine", return_value=mock_engine)

        result = controller.export_to_shell(str(workflow_file))

        assert result.success is True
        assert "#!/bin/bash" in result.data


class TestCreateWorkflow:
    """Tests for create_workflow method."""

    @pytest.fixture
    def controller(self):
        return WorkflowController()

    def test_create_workflow_success(self, controller, mocker):
        """Test that create_workflow succeeds."""
        mock_workflow_class = mocker.patch("bioamla.core.workflow.parser.Workflow")
        mock_step_class = mocker.patch("bioamla.core.workflow.parser.WorkflowStep")

        steps = [
            {"name": "step1", "action": "audio.resample", "params": {"input": "./in"}},
            {"name": "step2", "action": "analysis.indices", "depends_on": ["step1"]},
        ]

        result = controller.create_workflow(
            name="my_workflow",
            steps=steps,
            description="A test workflow",
        )

        assert result.success is True
        mock_step_class.assert_called()
        mock_workflow_class.assert_called()


class TestParseString:
    """Tests for parse_string method."""

    @pytest.fixture
    def controller(self):
        return WorkflowController()

    def test_parse_string_success(self, controller, mocker):
        """Test that parse_string succeeds."""
        toml_content = """[workflow]
name = "from_string"
[[steps]]
name = "s1"
action = "audio.resample"
"""
        mock_parse = mocker.patch("bioamla.core.workflow.parser.parse_workflow_string")
        mock_workflow = MagicMock()
        mock_workflow.name = "from_string"
        mock_workflow.description = ""
        mock_workflow.version = "1.0.0"
        mock_workflow.steps = [MagicMock(name="s1")]
        mock_workflow.steps[0].name = "s1"
        mock_workflow.variables = {}
        mock_workflow.get_execution_order.return_value = ["s1"]
        mock_parse.return_value = mock_workflow

        result = controller.parse_string(toml_content)

        assert result.success is True
        assert result.data.name == "from_string"


class TestRegisterAction:
    """Tests for register_action method."""

    @pytest.fixture
    def controller(self):
        return WorkflowController()

    def test_register_action_success(self, controller, mocker):
        """Test that register_action succeeds."""
        mock_engine = MagicMock()
        mocker.patch.object(controller, "_get_engine", return_value=mock_engine)

        def my_handler(params):
            return {"result": "ok"}

        result = controller.register_action("custom.my_action", my_handler)

        assert result.success is True
        mock_engine.register_action.assert_called_once_with("custom.my_action", my_handler)


class TestExecutionProgressCallback:
    """Tests for execution progress callback."""

    @pytest.fixture
    def controller(self):
        return WorkflowController()

    def test_set_execution_progress_callback(self, controller, mocker):
        """Test that progress callback is set."""
        callback_calls = []

        def callback(step_name, current, total, status):
            callback_calls.append((step_name, current, total, status))

        controller.set_execution_progress_callback(callback)

        assert controller._progress_callback is not None

    def test_cancel_execution(self, controller, mocker):
        """Test that cancel calls engine cancel."""
        mock_engine = MagicMock()
        controller._engine = mock_engine

        controller.cancel()

        mock_engine.cancel.assert_called_once()


class TestWorkflowSummary:
    """Tests for WorkflowSummary dataclass."""

    def test_workflow_summary_to_dict(self):
        """Test that WorkflowSummary converts to dict."""
        summary = WorkflowSummary(
            name="test",
            description="desc",
            version="1.0.0",
            num_steps=3,
            step_names=["s1", "s2", "s3"],
            variables={"var1": "value1"},
            execution_order=["s1", "s2", "s3"],
        )

        d = summary.to_dict()

        assert d["name"] == "test"
        assert d["num_steps"] == 3
        assert len(d["step_names"]) == 3


class TestValidationSummary:
    """Tests for ValidationSummary dataclass."""

    def test_validation_summary_fields(self):
        """Test that ValidationSummary has all fields."""
        summary = ValidationSummary(
            valid=False,
            num_errors=2,
            num_warnings=1,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
        )

        assert summary.valid is False
        assert summary.num_errors == 2
        assert summary.num_warnings == 1


class TestExecutionSummary:
    """Tests for ExecutionSummary dataclass."""

    def test_execution_summary_fields(self):
        """Test that ExecutionSummary has all fields."""
        summary = ExecutionSummary(
            workflow_name="test",
            execution_id="exec_123",
            status="completed",
            total_duration=5.5,
            steps_completed=3,
            steps_failed=0,
            steps_skipped=1,
            errors=[],
        )

        assert summary.workflow_name == "test"
        assert summary.status == "completed"
        assert summary.total_duration == 5.5

# tests/controllers/test_pipeline.py
"""
Tests for PipelineController.

Tests the TOML-based pipeline execution controller.
"""

import tempfile
from pathlib import Path

import pytest

from bioamla.controllers.pipeline import (
    ExecutionSummary,
    PipelineController,
    PipelineSummary,
    ValidationSummary,
)


@pytest.fixture
def controller():
    """Create a PipelineController instance."""
    return PipelineController()


@pytest.fixture
def sample_pipeline_toml():
    """Create a sample pipeline TOML content."""
    return """
[pipeline]
name = "test_pipeline"
description = "A test pipeline"
version = "1.0.0"

[variables]
input_dir = "./input"
output_dir = "./output"

[[steps]]
name = "step1"
action = "util.log"
description = "First step"
params = { message = "Step 1 executed" }

[[steps]]
name = "step2"
action = "util.log"
description = "Second step"
depends_on = ["step1"]
params = { message = "Step 2 executed" }
"""


@pytest.fixture
def pipeline_file(sample_pipeline_toml, tmp_path):
    """Create a temporary pipeline file."""
    path = tmp_path / "test_pipeline.toml"
    path.write_text(sample_pipeline_toml)
    return path


class TestPipelineControllerParsing:
    """Tests for pipeline parsing."""

    def test_parse_valid_file(self, controller, pipeline_file):
        """Test parsing a valid pipeline file."""
        result = controller.parse(str(pipeline_file))

        assert result.success is True
        assert isinstance(result.data, PipelineSummary)
        assert result.data.name == "test_pipeline"
        assert result.data.num_steps == 2
        assert "step1" in result.data.step_names
        assert "step2" in result.data.step_names

    def test_parse_nonexistent_file(self, controller):
        """Test parsing a nonexistent file fails."""
        result = controller.parse("/nonexistent/path.toml")

        assert result.success is False
        assert "not found" in result.error.lower() or "does not exist" in result.error.lower()

    def test_parse_string(self, controller, sample_pipeline_toml):
        """Test parsing pipeline from string."""
        result = controller.parse_string(sample_pipeline_toml)

        assert result.success is True
        assert result.data.name == "test_pipeline"
        assert result.data.num_steps == 2

    def test_parse_with_variables(self, controller, pipeline_file):
        """Test that variables are captured during parsing."""
        result = controller.parse(str(pipeline_file))

        assert result.success is True
        assert "input_dir" in result.data.variables
        assert result.data.variables["input_dir"] == "./input"


class TestPipelineControllerValidation:
    """Tests for pipeline validation."""

    def test_validate_valid_pipeline(self, controller, pipeline_file):
        """Test validating a valid pipeline."""
        result = controller.validate(str(pipeline_file))

        assert result.success is True
        assert isinstance(result.data, ValidationSummary)
        assert result.data.valid is True
        assert result.data.num_errors == 0

    def test_validate_strict_mode(self, controller, pipeline_file):
        """Test validation in strict mode."""
        result = controller.validate(str(pipeline_file), strict=True)

        assert result.success is True
        # If there are warnings in strict mode, it should fail
        # Our sample has no issues, so it should pass

    def test_validate_invalid_file(self, controller, tmp_path):
        """Test validating an invalid pipeline file."""
        invalid_file = tmp_path / "invalid.toml"
        invalid_file.write_text("this is not valid toml [[[")

        result = controller.validate(str(invalid_file))

        assert result.success is False


class TestPipelineControllerExecution:
    """Tests for pipeline execution."""

    def test_execute_simple_pipeline(self, controller, pipeline_file):
        """Test executing a simple pipeline."""
        result = controller.execute(str(pipeline_file))

        assert result.success is True
        assert isinstance(result.data, ExecutionSummary)
        assert result.data.pipeline_name == "test_pipeline"
        assert result.data.steps_completed == 2

    def test_execute_with_variable_override(self, controller, pipeline_file):
        """Test executing with variable overrides."""
        result = controller.execute(
            str(pipeline_file),
            variables={"input_dir": "/custom/input"},
        )

        assert result.success is True

    def test_execute_without_validation(self, controller, pipeline_file):
        """Test executing without pre-validation."""
        result = controller.execute(
            str(pipeline_file),
            validate_first=False,
        )

        assert result.success is True


class TestPipelineControllerExport:
    """Tests for pipeline export functionality."""

    def test_export_to_shell(self, controller, pipeline_file, tmp_path):
        """Test exporting pipeline to shell script."""
        output_path = tmp_path / "run.sh"
        result = controller.export_to_shell(
            str(pipeline_file),
            output_path=str(output_path),
        )

        assert result.success is True
        assert output_path.exists()
        content = output_path.read_text()
        assert "#!/bin/bash" in content or "bioamla" in content


class TestPipelineControllerActions:
    """Tests for action management."""

    def test_list_actions(self, controller):
        """Test listing available actions."""
        result = controller.list_actions()

        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) > 0
        # Check some expected actions exist
        assert any("util.log" in action for action in result.data)

    def test_register_custom_action(self, controller):
        """Test registering a custom action."""

        def custom_handler(context, params):
            return {"result": "custom action executed"}

        result = controller.register_action("custom.test", custom_handler)

        assert result.success is True

        # Verify it's in the list
        actions = controller.list_actions()
        assert "custom.test" in actions.data


class TestPipelineControllerHelpers:
    """Tests for helper methods."""

    def test_get_example_pipeline(self, controller):
        """Test getting example pipeline."""
        result = controller.get_example_pipeline()

        assert result.success is True
        assert "[pipeline]" in result.data
        assert "[[steps]]" in result.data

    def test_create_pipeline_programmatically(self, controller):
        """Test creating a pipeline programmatically."""
        steps = [
            {
                "name": "step1",
                "action": "util.log",
                "params": {"message": "Hello"},
            },
            {
                "name": "step2",
                "action": "util.log",
                "params": {"message": "World"},
                "depends_on": ["step1"],
            },
        ]

        result = controller.create_pipeline(
            name="my_pipeline",
            steps=steps,
            description="A test pipeline",
        )

        assert result.success is True
        assert result.data.name == "my_pipeline"
        assert len(result.data.steps) == 2


class TestPipelineControllerProgressCallback:
    """Tests for progress callback functionality."""

    def test_set_progress_callback(self, controller, pipeline_file):
        """Test that progress callback is called during execution."""
        progress_updates = []

        def callback(step_name, current, total, status):
            progress_updates.append({
                "step": step_name,
                "current": current,
                "total": total,
                "status": status,
            })

        controller.set_execution_progress_callback(callback)
        result = controller.execute(str(pipeline_file))

        assert result.success is True
        assert len(progress_updates) > 0


class TestExecutionSummaryDataclass:
    """Tests for ExecutionSummary dataclass."""

    def test_execution_summary_fields(self):
        """Test ExecutionSummary has expected fields."""
        summary = ExecutionSummary(
            pipeline_name="test",
            execution_id="123",
            status="completed",
            total_duration=1.5,
            steps_completed=2,
            steps_failed=0,
            steps_skipped=0,
        )

        assert summary.pipeline_name == "test"
        assert summary.total_duration == 1.5
        assert summary.steps_completed == 2


class TestValidationSummaryDataclass:
    """Tests for ValidationSummary dataclass."""

    def test_validation_summary_valid(self):
        """Test ValidationSummary for valid pipeline."""
        summary = ValidationSummary(
            valid=True,
            num_errors=0,
            num_warnings=0,
        )

        assert summary.valid is True
        assert summary.num_errors == 0

    def test_validation_summary_invalid(self):
        """Test ValidationSummary for invalid pipeline."""
        summary = ValidationSummary(
            valid=False,
            num_errors=2,
            num_warnings=1,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
        )

        assert summary.valid is False
        assert len(summary.errors) == 2


class TestPipelineSummaryDataclass:
    """Tests for PipelineSummary dataclass."""

    def test_pipeline_summary_fields(self):
        """Test PipelineSummary has expected fields."""
        summary = PipelineSummary(
            name="test_pipeline",
            description="A test",
            version="1.0.0",
            num_steps=3,
            step_names=["step1", "step2", "step3"],
            variables={"key": "value"},
            execution_order=["step1", "step2", "step3"],
        )

        assert summary.name == "test_pipeline"
        assert summary.num_steps == 3
        assert len(summary.step_names) == 3

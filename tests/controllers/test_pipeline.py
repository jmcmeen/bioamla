# tests/controllers/test_pipeline.py
"""
Tests for PipelineController.
"""


import pytest

from bioamla.controllers.pipeline import (
    PipelineController,
    PipelineState,
    PipelineStatus,
)
from bioamla.core.workflow.step import PipelineStep, StepResult


def create_mock_step(name: str = "mock_step", should_fail: bool = False):
    """Create a mock pipeline step for testing."""

    class MockStep(PipelineStep):
        def execute(self, input_data=None):
            if self._should_fail:
                return StepResult.fail("Mock step failed")
            return StepResult.ok(data=input_data, message="Mock step succeeded")

    # Set class attributes
    MockStep.name = name
    MockStep.description = f"Mock step: {name}"

    # Create instance
    step = MockStep()
    step._should_fail = should_fail
    return step


class TestPipelineController:
    """Tests for PipelineController."""

    @pytest.fixture
    def controller(self):
        return PipelineController(name="test_pipeline")

    def test_execute_pipeline_success(self, controller, mocker):
        """Test that executing a pipeline succeeds."""
        # Mock run tracking
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_complete_run")

        # Add steps to the pipeline
        step1 = create_mock_step(name="step1")
        step2 = create_mock_step(name="step2")
        controller.add_step(step1)
        controller.add_step(step2)

        result = controller.execute(input_data="test_input", track_undo=False)

        assert result.success is True

    def test_execute_empty_pipeline_fails(self, controller):
        """Test that executing an empty pipeline fails."""
        result = controller.execute()

        assert result.success is False
        assert "no steps" in result.error.lower()

    def test_add_step_success(self, controller):
        """Test that adding a step succeeds."""
        step = create_mock_step(name="test_step")

        controller.add_step(step)

        assert len(controller.steps) == 1
        assert controller.steps[0].name == "test_step"


class TestPipelineState:
    """Tests for pipeline state methods."""

    @pytest.fixture
    def controller(self):
        return PipelineController(name="test_pipeline")

    def test_clear_steps_removes_all_steps(self, controller):
        """Test that clear_steps removes all pipeline steps."""
        controller.add_step(create_mock_step(name="step1"))
        controller.add_step(create_mock_step(name="step2"))

        controller.clear_steps()

        assert len(controller.steps) == 0

    def test_get_state_returns_pipeline_state(self, controller):
        """Test that get_state returns the current pipeline state."""
        state = controller.get_state()

        assert isinstance(state, PipelineState)
        assert state.name == "test_pipeline"
        assert state.status == PipelineStatus.IDLE


class TestPipelineUndo:
    """Tests for pipeline undo functionality."""

    @pytest.fixture
    def controller(self):
        return PipelineController(name="test_pipeline")

    def test_undo_without_executed_commands_fails(self, controller):
        """Test that undo without executed commands fails gracefully."""
        result = controller.undo()

        # Should fail when nothing to undo
        assert result.success is False
        assert "nothing to undo" in result.error.lower()

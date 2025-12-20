# tests/controllers/test_pipeline.py
"""
Tests for PipelineController.
"""


import pytest

from bioamla.controllers.pipeline import (
    PipelineController,
    PipelineProgress,
    PipelineResult,
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


class TestPipelineExecution:
    """Tests for pipeline execution."""

    @pytest.fixture
    def controller(self):
        return PipelineController(name="test_pipeline")

    def test_execute_with_steps_tracks_undo(self, controller, mocker):
        """Test that execute with track_undo adds to history."""
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_complete_run")

        step1 = create_mock_step(name="step1")
        controller.add_step(step1)

        result = controller.execute(input_data="input", track_undo=True)

        assert result.success is True
        assert controller.can_undo is True

    def test_execute_step_failure_propagates(self, controller, mocker):
        """Test that step failure propagates to pipeline result."""
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_fail_run")

        step1 = create_mock_step(name="step1")
        step2 = create_mock_step(name="step2", should_fail=True)
        controller.add_step(step1)
        controller.add_step(step2)

        result = controller.execute(track_undo=False)

        assert result.success is False
        assert "step2" in result.error.lower()

    def test_cancel_sets_cancelled_flag(self, controller):
        """Test that cancel sets the cancelled flag."""
        controller.cancel()

        assert controller._cancelled is True


class TestStepManagement:
    """Tests for step management."""

    @pytest.fixture
    def controller(self):
        return PipelineController(name="test_pipeline")

    def test_add_steps_bulk(self, controller):
        """Test adding multiple steps at once."""
        steps = [
            create_mock_step(name="step1"),
            create_mock_step(name="step2"),
            create_mock_step(name="step3"),
        ]

        controller.add_steps(steps)

        assert len(controller.steps) == 3

    def test_remove_step_success(self, controller):
        """Test removing a step."""
        controller.add_step(create_mock_step(name="step1"))
        controller.add_step(create_mock_step(name="step2"))

        result = controller.remove_step("step1")

        assert result is True
        assert len(controller.steps) == 1
        assert controller.steps[0].name == "step2"

    def test_remove_nonexistent_step_fails(self, controller):
        """Test removing nonexistent step fails."""
        result = controller.remove_step("nonexistent")

        assert result is False

    def test_get_step_success(self, controller):
        """Test getting a step by name."""
        step = create_mock_step(name="my_step")
        controller.add_step(step)

        result = controller.get_step("my_step")

        assert result is not None
        assert result.name == "my_step"

    def test_get_nonexistent_step_returns_none(self, controller):
        """Test getting nonexistent step returns None."""
        result = controller.get_step("nonexistent")

        assert result is None

    def test_duplicate_step_name_raises(self, controller):
        """Test that adding duplicate step name raises error."""
        controller.add_step(create_mock_step(name="step1"))

        with pytest.raises(ValueError) as exc_info:
            controller.add_step(create_mock_step(name="step1"))

        assert "already exists" in str(exc_info.value)


class TestPipelineStateSerialization:
    """Tests for pipeline state serialization."""

    @pytest.fixture
    def controller(self):
        return PipelineController(name="test_pipeline")

    def test_save_state_success(self, controller, tmp_path):
        """Test that save_state succeeds."""
        state_path = tmp_path / "pipeline_state.json"
        result = controller.save_state(state_path)

        assert result.success is True
        assert state_path.exists()

    def test_load_state_success(self, controller, tmp_path):
        """Test that load_state succeeds."""
        # First save state
        state_path = tmp_path / "pipeline_state.json"
        controller.save_state(state_path)

        # Then load it
        result = controller.load_state(state_path)

        assert result.success is True
        assert result.data.name == "test_pipeline"

    def test_load_nonexistent_state_fails(self, controller, tmp_path):
        """Test that load_state fails for nonexistent file."""
        result = controller.load_state(tmp_path / "nonexistent.json")

        assert result.success is False


class TestPipelineProgress:
    """Tests for PipelineProgress dataclass."""

    def test_percent_calculation(self):
        """Test percent is calculated correctly."""
        progress = PipelineProgress(total_steps=10, completed_steps=5)

        assert progress.percent == 50.0

    def test_percent_zero_total(self):
        """Test percent with zero total."""
        progress = PipelineProgress(total_steps=0, completed_steps=0)

        assert progress.percent == 0.0

    def test_remaining_calculation(self):
        """Test remaining steps calculation."""
        progress = PipelineProgress(total_steps=10, completed_steps=3)

        assert progress.remaining == 7


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_pipeline_result_ok(self):
        """Test creating success result."""
        result = PipelineResult.ok(data="output", message="Done")

        assert result.success is True
        assert result.data == "output"
        assert result.message == "Done"

    def test_pipeline_result_fail(self):
        """Test creating failure result."""
        result = PipelineResult.fail("Error occurred")

        assert result.success is False
        assert result.error == "Error occurred"


class TestPipelineStateDataclass:
    """Tests for PipelineState dataclass."""

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        from uuid import uuid4
        from datetime import datetime

        original = PipelineState(
            id=uuid4(),
            name="test",
            status=PipelineStatus.COMPLETED,
            current_step_index=2,
            step_results={"step1": {"success": True}},
            started_at=datetime.now(),
            paused_at=None,
            intermediate_data=None,
        )

        dict_data = original.to_dict()
        restored = PipelineState.from_dict(dict_data)

        assert restored.name == original.name
        assert restored.status == original.status
        assert restored.current_step_index == original.current_step_index


class TestPipelineReset:
    """Tests for pipeline reset functionality."""

    @pytest.fixture
    def controller(self):
        return PipelineController(name="test_pipeline")

    def test_reset_clears_state(self, controller, mocker):
        """Test that reset clears execution state."""
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_complete_run")

        controller.add_step(create_mock_step(name="step1"))
        controller.execute(track_undo=False)

        controller.reset()

        assert controller.status == PipelineStatus.IDLE


class TestPipelineSummary:
    """Tests for pipeline summary method."""

    @pytest.fixture
    def controller(self):
        return PipelineController(name="test_pipeline")

    def test_summary_returns_dict(self, controller):
        """Test that summary returns a dictionary."""
        controller.add_step(create_mock_step(name="step1"))
        controller.add_step(create_mock_step(name="step2"))

        summary = controller.summary()

        assert summary["name"] == "test_pipeline"
        assert summary["step_count"] == 2
        assert len(summary["steps"]) == 2


class TestProgressCallback:
    """Tests for progress callback functionality."""

    @pytest.fixture
    def controller(self):
        return PipelineController(name="test_pipeline")

    def test_set_progress_callback(self, controller, mocker):
        """Test that progress callback is called."""
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_complete_run")

        progress_updates = []

        def callback(progress):
            progress_updates.append(progress)

        controller.set_progress_callback(callback)
        controller.add_step(create_mock_step(name="step1"))
        controller.execute(track_undo=False)

        assert len(progress_updates) > 0

    def test_step_history_property(self, controller):
        """Test that step_history returns step info."""
        controller.add_step(create_mock_step(name="step1"))

        history = controller.step_history

        assert len(history) == 1
        assert history[0].name == "step1"


class TestUndoRedo:
    """Tests for undo/redo with history."""

    @pytest.fixture
    def controller(self):
        return PipelineController(name="test_pipeline")

    def test_redo_without_undo_fails(self, controller):
        """Test that redo fails when nothing to redo."""
        result = controller.redo()

        assert result.success is False
        assert "nothing to redo" in result.error.lower()

    def test_clear_history(self, controller, mocker):
        """Test that clear_history removes undo history."""
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_complete_run")

        controller.add_step(create_mock_step(name="step1"))
        controller.execute(track_undo=True)

        controller.clear_history()

        assert controller.can_undo is False

    def test_can_undo_property(self, controller, mocker):
        """Test can_undo property."""
        assert controller.can_undo is False

        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_complete_run")

        controller.add_step(create_mock_step(name="step1"))
        controller.execute(track_undo=True)

        assert controller.can_undo is True

    def test_can_redo_property(self, controller, mocker):
        """Test can_redo property."""
        assert controller.can_redo is False

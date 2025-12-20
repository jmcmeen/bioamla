# controllers/pipeline.py
"""
Pipeline Controller
===================

Controller for orchestrating multi-step processing pipelines.

The PipelineController manages:
- Step registration and dependency resolution
- Sequential and parallel execution
- Progress tracking across stages
- Error handling with partial result recovery
- Pipeline state serialization/resumption
- Integration with UndoManager for pipeline-level undo

Usage:
    from bioamla.controllers import PipelineController
    from bioamla.core.workflow.step import PipelineStep

    # Create controller
    pipeline = PipelineController(name="audio_processing")

    # Add steps
    pipeline.add_step(LoadAudioStep())
    pipeline.add_step(NormalizeStep(target_db=-20))
    pipeline.add_step(FilterStep(low=500, high=8000))
    pipeline.add_step(SaveAudioStep())

    # Execute
    result = pipeline.execute(input_path="recording.wav")

    # Check progress
    for step_info in pipeline.step_history:
        print(f"{step_info.name}: {step_info.status.value}")

    # Undo entire pipeline (if supported)
    pipeline.undo()
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar
from uuid import UUID, uuid4

from bioamla.commands.base import Command, CommandResult, UndoManager
from bioamla.core.workflow.step import PipelineStep, StepInfo, StepResult, StepStatus

from .base import BaseController, ControllerResult

T = TypeVar("T")


class PipelineStatus(Enum):
    """Status of the entire pipeline."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineProgress:
    """Progress information for pipeline execution."""

    total_steps: int
    completed_steps: int = 0
    current_step: Optional[str] = None
    current_step_index: int = 0
    status: PipelineStatus = PipelineStatus.IDLE
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def percent(self) -> float:
        """Get completion percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100

    @property
    def remaining(self) -> int:
        """Get remaining steps."""
        return self.total_steps - self.completed_steps


@dataclass
class PipelineState:
    """Serializable state of a pipeline for resumption."""

    id: UUID
    name: str
    status: PipelineStatus
    current_step_index: int
    step_results: Dict[str, Dict[str, Any]]
    started_at: Optional[datetime]
    paused_at: Optional[datetime]
    intermediate_data: Optional[Any]  # Serialized intermediate result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "step_results": self.step_results,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            status=PipelineStatus(data["status"]),
            current_step_index=data["current_step_index"],
            step_results=data["step_results"],
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            paused_at=datetime.fromisoformat(data["paused_at"]) if data.get("paused_at") else None,
            intermediate_data=None,
        )


@dataclass
class PipelineResult(Generic[T]):
    """Result of a pipeline execution."""

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    message: Optional[str] = None
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    duration_seconds: float = 0.0
    completed_steps: int = 0
    total_steps: int = 0

    @classmethod
    def ok(
        cls,
        data: T = None,
        message: str = None,
        step_results: Dict[str, StepResult] = None,
        **kwargs,
    ) -> "PipelineResult[T]":
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            message=message,
            step_results=step_results or {},
            **kwargs,
        )

    @classmethod
    def fail(
        cls,
        error: str,
        step_results: Dict[str, StepResult] = None,
        **kwargs,
    ) -> "PipelineResult[T]":
        """Create a failed result."""
        return cls(
            success=False,
            error=error,
            step_results=step_results or {},
            **kwargs,
        )


# Type alias for progress callback
PipelineProgressCallback = Callable[[PipelineProgress], None]


class PipelineCommand(Command):
    """Command wrapper for pipeline execution (for undo support)."""

    def __init__(
        self,
        pipeline: "PipelineController",
        input_data: Any,
        cleanup_func: Optional[Callable[[], None]] = None,
    ):
        self._pipeline = pipeline
        self._input_data = input_data
        self._cleanup_func = cleanup_func
        self._result: Optional[PipelineResult] = None
        super().__init__()

    @property
    def name(self) -> str:
        return f"Pipeline: {self._pipeline.name}"

    @property
    def description(self) -> str:
        return f"Execute {len(self._pipeline._steps)} step pipeline"

    def execute(self) -> CommandResult:
        """Execute the pipeline."""
        self._result = self._pipeline._execute_internal(self._input_data)
        if self._result.success:
            return CommandResult.ok(
                data=self._result.data,
                message=self._result.message,
            )
        return CommandResult.fail(self._result.error)

    def undo(self) -> None:
        """Undo the pipeline by calling cleanup."""
        if self._cleanup_func:
            self._cleanup_func()


class PipelineController(BaseController):
    """
    Controller for orchestrating multi-step processing pipelines.

    Manages step registration, dependency resolution, execution,
    progress tracking, and error handling.
    """

    def __init__(
        self,
        name: str = "pipeline",
        max_undo_levels: int = 10,
    ):
        super().__init__()
        self._id = uuid4()
        self._name = name
        self._steps: List[PipelineStep] = []
        self._step_map: Dict[str, PipelineStep] = {}
        self._undo_manager = UndoManager(max_history=max_undo_levels)
        self._progress_callback: Optional[PipelineProgressCallback] = None
        self._status = PipelineStatus.IDLE
        self._current_progress: Optional[PipelineProgress] = None
        self._step_results: Dict[str, StepResult] = {}
        self._started_at: Optional[datetime] = None
        self._cancelled = False

    @property
    def id(self) -> UUID:
        """Get the pipeline ID."""
        return self._id

    @property
    def name(self) -> str:
        """Get the pipeline name."""
        return self._name

    @property
    def status(self) -> PipelineStatus:
        """Get the current pipeline status."""
        return self._status

    @property
    def steps(self) -> List[PipelineStep]:
        """Get the list of steps."""
        return self._steps.copy()

    @property
    def step_history(self) -> List[StepInfo]:
        """Get execution history for all steps."""
        return [step.info for step in self._steps]

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._undo_manager.can_undo

    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._undo_manager.can_redo

    # =========================================================================
    # Step Management
    # =========================================================================

    def add_step(self, step: PipelineStep) -> "PipelineController":
        """
        Add a step to the pipeline.

        Args:
            step: The step to add

        Returns:
            Self for method chaining
        """
        if step.name in self._step_map:
            raise ValueError(f"Step with name '{step.name}' already exists")

        self._steps.append(step)
        self._step_map[step.name] = step
        return self

    def add_steps(self, steps: List[PipelineStep]) -> "PipelineController":
        """Add multiple steps to the pipeline."""
        for step in steps:
            self.add_step(step)
        return self

    def remove_step(self, step_name: str) -> bool:
        """
        Remove a step from the pipeline.

        Args:
            step_name: Name of the step to remove

        Returns:
            True if step was removed, False if not found
        """
        if step_name not in self._step_map:
            return False

        step = self._step_map.pop(step_name)
        self._steps.remove(step)
        return True

    def get_step(self, step_name: str) -> Optional[PipelineStep]:
        """Get a step by name."""
        return self._step_map.get(step_name)

    def clear_steps(self) -> None:
        """Remove all steps from the pipeline."""
        self._steps.clear()
        self._step_map.clear()

    # =========================================================================
    # Dependency Resolution
    # =========================================================================

    def _resolve_execution_order(self) -> List[PipelineStep]:
        """
        Resolve step execution order based on dependencies.

        Uses topological sort to determine order.

        Returns:
            Ordered list of steps to execute

        Raises:
            ValueError: If circular dependencies are detected
        """
        if not any(step.depends_on for step in self._steps):
            # No dependencies, use registration order
            return self._steps.copy()

        # Build dependency graph
        in_degree: Dict[str, int] = {step.name: 0 for step in self._steps}
        graph: Dict[str, Set[str]] = {step.name: set() for step in self._steps}

        for step in self._steps:
            for dep_name in step.depends_on:
                if dep_name not in self._step_map:
                    raise ValueError(
                        f"Step '{step.name}' depends on unknown step '{dep_name}'"
                    )
                graph[dep_name].add(step.name)
                in_degree[step.name] += 1

        # Kahn's algorithm for topological sort
        queue = [name for name, degree in in_degree.items() if degree == 0]
        sorted_names = []

        while queue:
            current = queue.pop(0)
            sorted_names.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_names) != len(self._steps):
            raise ValueError("Circular dependency detected in pipeline steps")

        return [self._step_map[name] for name in sorted_names]

    # =========================================================================
    # Progress Tracking
    # =========================================================================

    def set_progress_callback(
        self,
        callback: PipelineProgressCallback,
    ) -> None:
        """Set a callback for progress updates."""
        self._progress_callback = callback

    def _report_progress(self, progress: PipelineProgress) -> None:
        """Report progress if a callback is set."""
        self._current_progress = progress
        if self._progress_callback:
            self._progress_callback(progress)

    # =========================================================================
    # Execution
    # =========================================================================

    def execute(
        self,
        input_data: Any = None,
        cleanup_func: Optional[Callable[[], None]] = None,
        track_undo: bool = True,
    ) -> ControllerResult[PipelineResult]:
        """
        Execute the pipeline.

        Args:
            input_data: Initial data to pass to the first step
            cleanup_func: Function to call on undo
            track_undo: Whether to track this execution for undo

        Returns:
            ControllerResult containing PipelineResult
        """
        if not self._steps:
            return ControllerResult.fail("Pipeline has no steps")

        if track_undo:
            cmd = PipelineCommand(self, input_data, cleanup_func)
            result = self._undo_manager.execute(cmd)
            if result.success:
                return ControllerResult.ok(
                    data=result.data,
                    message=result.message,
                )
            return ControllerResult.fail(result.error)
        else:
            pipeline_result = self._execute_internal(input_data)
            if pipeline_result.success:
                return ControllerResult.ok(
                    data=pipeline_result,
                    message=pipeline_result.message,
                )
            return ControllerResult.fail(pipeline_result.error)

    def _execute_internal(self, input_data: Any) -> PipelineResult:
        """Internal execution logic."""
        self._status = PipelineStatus.RUNNING
        self._started_at = datetime.utcnow()
        self._step_results.clear()
        self._cancelled = False

        # Resolve execution order
        try:
            ordered_steps = self._resolve_execution_order()
        except ValueError as e:
            self._status = PipelineStatus.FAILED
            return PipelineResult.fail(str(e))

        # Initialize progress
        progress = PipelineProgress(
            total_steps=len(ordered_steps),
            status=PipelineStatus.RUNNING,
        )
        self._report_progress(progress)

        # Execute steps
        current_data = input_data

        for i, step in enumerate(ordered_steps):
            if self._cancelled:
                self._status = PipelineStatus.CANCELLED
                return PipelineResult.fail(
                    "Pipeline cancelled",
                    step_results=self._step_results,
                    completed_steps=progress.completed_steps,
                    total_steps=len(ordered_steps),
                )

            progress.current_step = step.name
            progress.current_step_index = i
            self._report_progress(progress)

            # Execute step
            result = step.run(current_data)
            self._step_results[step.name] = result

            if not result.success:
                self._status = PipelineStatus.FAILED
                progress.status = PipelineStatus.FAILED
                progress.errors.append(f"{step.name}: {result.error}")
                self._report_progress(progress)

                return PipelineResult.fail(
                    f"Step '{step.name}' failed: {result.error}",
                    step_results=self._step_results,
                    completed_steps=progress.completed_steps,
                    total_steps=len(ordered_steps),
                )

            # Collect warnings
            if result.warnings:
                progress.warnings.extend(
                    [f"{step.name}: {w}" for w in result.warnings]
                )

            # Use result data as input for next step (unless skipped with no data)
            if result.data is not None:
                current_data = result.data

            progress.completed_steps += 1
            self._report_progress(progress)

        # Success
        self._status = PipelineStatus.COMPLETED
        progress.status = PipelineStatus.COMPLETED
        self._report_progress(progress)

        duration = (datetime.utcnow() - self._started_at).total_seconds()

        return PipelineResult.ok(
            data=current_data,
            message=f"Pipeline completed: {len(ordered_steps)} steps in {duration:.2f}s",
            step_results=self._step_results,
            duration_seconds=duration,
            completed_steps=len(ordered_steps),
            total_steps=len(ordered_steps),
        )

    def cancel(self) -> None:
        """Cancel the running pipeline."""
        self._cancelled = True

    # =========================================================================
    # State Serialization
    # =========================================================================

    def get_state(self) -> PipelineState:
        """Get the current pipeline state for serialization."""
        step_results_dict = {
            name: {
                "success": result.success,
                "error": result.error,
                "message": result.message,
                "metadata": result.metadata,
            }
            for name, result in self._step_results.items()
        }

        return PipelineState(
            id=self._id,
            name=self._name,
            status=self._status,
            current_step_index=self._current_progress.current_step_index if self._current_progress else 0,
            step_results=step_results_dict,
            started_at=self._started_at,
            paused_at=datetime.utcnow() if self._status == PipelineStatus.PAUSED else None,
            intermediate_data=None,
        )

    def save_state(self, path: Path) -> ControllerResult[str]:
        """
        Save pipeline state to a file.

        Args:
            path: Path to save state file

        Returns:
            ControllerResult with the saved path
        """
        try:
            state = self.get_state()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            return ControllerResult.ok(
                data=str(path),
                message=f"Pipeline state saved to {path}",
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to save state: {e}")

    def load_state(self, path: Path) -> ControllerResult[PipelineState]:
        """
        Load pipeline state from a file.

        Args:
            path: Path to state file

        Returns:
            ControllerResult with loaded PipelineState
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)
            state = PipelineState.from_dict(data)
            return ControllerResult.ok(
                data=state,
                message=f"Pipeline state loaded from {path}",
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to load state: {e}")

    # =========================================================================
    # Undo/Redo
    # =========================================================================

    def undo(self) -> ControllerResult[str]:
        """Undo the last pipeline execution."""
        result = self._undo_manager.undo()
        if result is None:
            return ControllerResult.fail("Nothing to undo")
        if result.success:
            return ControllerResult.ok(message=result.message)
        return ControllerResult.fail(result.error)

    def redo(self) -> ControllerResult[str]:
        """Redo the last undone execution."""
        result = self._undo_manager.redo()
        if result is None:
            return ControllerResult.fail("Nothing to redo")
        if result.success:
            return ControllerResult.ok(message=result.message)
        return ControllerResult.fail(result.error)

    def clear_history(self) -> None:
        """Clear undo/redo history."""
        self._undo_manager.clear()

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def reset(self) -> None:
        """Reset the pipeline to initial state."""
        self._status = PipelineStatus.IDLE
        self._step_results.clear()
        self._current_progress = None
        self._started_at = None
        self._cancelled = False
        for step in self._steps:
            step._info.status = StepStatus.PENDING

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline."""
        return {
            "id": str(self._id),
            "name": self._name,
            "status": self._status.value,
            "step_count": len(self._steps),
            "steps": [
                {
                    "name": step.name,
                    "description": step.description,
                    "status": step.info.status.value,
                    "depends_on": list(step.depends_on),
                }
                for step in self._steps
            ],
            "completed_steps": sum(
                1 for step in self._steps
                if step.info.status == StepStatus.COMPLETED
            ),
            "failed_steps": sum(
                1 for step in self._steps
                if step.info.status == StepStatus.FAILED
            ),
        }

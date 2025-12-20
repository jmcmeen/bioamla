# core/workflow/step.py
"""
Pipeline Step Base Class
========================

Base class for pipeline steps that can be composed into multi-step workflows.

Each step defines:
- Input/output type declarations
- Validation logic
- Execution logic
- Resource cleanup

Steps are executed by the PipelineController, which handles:
- Dependency resolution
- Progress tracking
- Error handling
- State serialization

Usage:
    from bioamla.core.workflow.step import PipelineStep, StepResult

    class NormalizeStep(PipelineStep[AudioData, AudioData]):
        name = "normalize"
        description = "Normalize audio loudness"

        def __init__(self, target_db: float = -20.0):
            super().__init__()
            self.target_db = target_db

        def validate_input(self, data: AudioData) -> Optional[str]:
            if data.samples is None or len(data.samples) == 0:
                return "Input audio is empty"
            return None

        def execute(self, data: AudioData) -> StepResult[AudioData]:
            from bioamla.controllers import AudioTransformController
            ctrl = AudioTransformController()
            result = ctrl.normalize_loudness(data, self.target_db)
            if result.success:
                return StepResult.ok(result.data)
            return StepResult.fail(result.error)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar
from uuid import UUID, uuid4

# Type variables for input/output types
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class StepStatus(Enum):
    """Status of a pipeline step."""

    PENDING = "pending"
    VALIDATING = "validating"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult(Generic[OutputT]):
    """Result of a step execution."""

    success: bool
    data: Optional[OutputT] = None
    error: Optional[str] = None
    message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        data: OutputT = None,
        message: str = None,
        warnings: List[str] = None,
        **metadata,
    ) -> "StepResult[OutputT]":
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            message=message,
            warnings=warnings or [],
            metadata=metadata,
        )

    @classmethod
    def fail(
        cls,
        error: str,
        warnings: List[str] = None,
        **metadata,
    ) -> "StepResult[OutputT]":
        """Create a failed result."""
        return cls(
            success=False,
            error=error,
            warnings=warnings or [],
            metadata=metadata,
        )

    @classmethod
    def skip(cls, reason: str) -> "StepResult[OutputT]":
        """Create a skipped result."""
        return cls(
            success=True,
            message=f"Skipped: {reason}",
            metadata={"skipped": True, "skip_reason": reason},
        )


@dataclass
class StepInfo:
    """Metadata about a step execution."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
        }


class PipelineStep(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for pipeline steps.

    A step represents a single operation in a pipeline that:
    - Takes typed input
    - Produces typed output
    - Can be validated before execution
    - Tracks execution metadata
    - Supports resource cleanup

    Type Parameters:
        InputT: The type of input data this step accepts
        OutputT: The type of output data this step produces

    Class Attributes:
        name: Unique identifier for this step type
        description: Human-readable description
        depends_on: Set of step names this step depends on
    """

    # Subclasses should override these
    name: str = "unnamed_step"
    description: str = "No description"

    def __init__(self):
        self._info = StepInfo(name=self.name, description=self.description)
        self._depends_on: Set[str] = set()
        self._resources: List[Any] = []

    @property
    def info(self) -> StepInfo:
        """Get step execution metadata."""
        return self._info

    @property
    def depends_on(self) -> Set[str]:
        """Get the set of step names this step depends on."""
        return self._depends_on

    def add_dependency(self, step_name: str) -> "PipelineStep":
        """Add a dependency on another step. Returns self for chaining."""
        self._depends_on.add(step_name)
        return self

    def validate_input(self, data: InputT) -> Optional[str]:
        """
        Validate input data before execution.

        Override this method to add input validation logic.

        Args:
            data: The input data to validate

        Returns:
            None if valid, error message string if invalid
        """
        return None

    def validate_output(self, data: OutputT) -> Optional[str]:
        """
        Validate output data after execution.

        Override this method to add output validation logic.

        Args:
            data: The output data to validate

        Returns:
            None if valid, error message string if invalid
        """
        return None

    @abstractmethod
    def execute(self, data: InputT) -> StepResult[OutputT]:
        """
        Execute the step.

        This is the main method that subclasses must implement.

        Args:
            data: The input data

        Returns:
            StepResult containing the output or error
        """
        ...

    def should_skip(self, data: InputT) -> Optional[str]:
        """
        Check if this step should be skipped.

        Override this method to add conditional execution logic.

        Args:
            data: The input data

        Returns:
            None to execute, or skip reason string to skip
        """
        return None

    def on_start(self, data: InputT) -> None:
        """
        Called before execution starts.

        Override this for pre-execution setup.
        """
        pass

    def on_success(self, result: OutputT) -> None:
        """
        Called after successful execution.

        Override this for post-execution actions on success.
        """
        pass

    def on_failure(self, error: str) -> None:
        """
        Called after failed execution.

        Override this for error handling/cleanup.
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up any resources acquired during execution.

        Override this to release resources like temporary files,
        database connections, etc.
        """
        for resource in self._resources:
            try:
                if hasattr(resource, "close"):
                    resource.close()
                elif hasattr(resource, "cleanup"):
                    resource.cleanup()
            except Exception:
                pass  # Best effort cleanup
        self._resources.clear()

    def register_resource(self, resource: Any) -> None:
        """Register a resource for cleanup."""
        self._resources.append(resource)

    def summarize_input(self, data: InputT) -> str:
        """
        Create a summary string for the input data.

        Override this to provide meaningful summaries for logging.
        """
        return f"{type(data).__name__}"

    def summarize_output(self, data: OutputT) -> str:
        """
        Create a summary string for the output data.

        Override this to provide meaningful summaries for logging.
        """
        return f"{type(data).__name__}"

    def run(self, data: InputT) -> StepResult[OutputT]:
        """
        Run the step with full lifecycle management.

        This method handles:
        - Skip checking
        - Input validation
        - Execution timing
        - Output validation
        - Status tracking
        - Cleanup on error

        Args:
            data: The input data

        Returns:
            StepResult containing the output or error
        """
        self._info.started_at = datetime.utcnow()
        self._info.input_summary = self.summarize_input(data)

        try:
            # Check if should skip
            skip_reason = self.should_skip(data)
            if skip_reason:
                self._info.status = StepStatus.SKIPPED
                self._info.completed_at = datetime.utcnow()
                return StepResult.skip(skip_reason)

            # Validate input
            self._info.status = StepStatus.VALIDATING
            validation_error = self.validate_input(data)
            if validation_error:
                self._info.status = StepStatus.FAILED
                self._info.error = f"Input validation failed: {validation_error}"
                self._info.completed_at = datetime.utcnow()
                return StepResult.fail(self._info.error)

            # Execute
            self._info.status = StepStatus.RUNNING
            self.on_start(data)

            result = self.execute(data)

            if not result.success:
                self._info.status = StepStatus.FAILED
                self._info.error = result.error
                self._info.completed_at = datetime.utcnow()
                self._info.duration_seconds = (
                    self._info.completed_at - self._info.started_at
                ).total_seconds()
                self.on_failure(result.error)
                return result

            # Validate output
            if result.data is not None:
                output_error = self.validate_output(result.data)
                if output_error:
                    self._info.status = StepStatus.FAILED
                    self._info.error = f"Output validation failed: {output_error}"
                    self._info.completed_at = datetime.utcnow()
                    self.on_failure(self._info.error)
                    return StepResult.fail(self._info.error)

                self._info.output_summary = self.summarize_output(result.data)

            # Success
            self._info.status = StepStatus.COMPLETED
            self._info.completed_at = datetime.utcnow()
            self._info.duration_seconds = (
                self._info.completed_at - self._info.started_at
            ).total_seconds()
            self.on_success(result.data)

            return result

        except Exception as e:
            self._info.status = StepStatus.FAILED
            self._info.error = str(e)
            self._info.completed_at = datetime.utcnow()
            self._info.duration_seconds = (
                self._info.completed_at - self._info.started_at
            ).total_seconds()
            self.on_failure(str(e))
            return StepResult.fail(str(e))

        finally:
            # Always attempt cleanup
            self.cleanup()


# =============================================================================
# Common Step Types
# =============================================================================


class PassthroughStep(PipelineStep[InputT, InputT]):
    """A step that passes data through unchanged. Useful for conditional logic."""

    name = "passthrough"
    description = "Pass data through unchanged"

    def execute(self, data: InputT) -> StepResult[InputT]:
        return StepResult.ok(data)


class TransformStep(PipelineStep[InputT, OutputT]):
    """
    A step that applies a transformation function.

    Usage:
        step = TransformStep(
            name="double",
            transform=lambda x: x * 2,
        )
    """

    def __init__(
        self,
        name: str,
        transform: callable,
        description: str = "Apply transformation",
    ):
        super().__init__()
        self.name = name
        self.description = description
        self._transform = transform

    def execute(self, data: InputT) -> StepResult[OutputT]:
        try:
            result = self._transform(data)
            return StepResult.ok(result)
        except Exception as e:
            return StepResult.fail(str(e))


class BranchStep(PipelineStep[InputT, OutputT]):
    """
    A step that chooses between multiple sub-steps based on a condition.

    Usage:
        step = BranchStep(
            name="route_by_type",
            condition=lambda x: "stereo" if x.channels > 1 else "mono",
            branches={
                "stereo": StereoProcessStep(),
                "mono": MonoProcessStep(),
            },
        )
    """

    def __init__(
        self,
        name: str,
        condition: callable,
        branches: Dict[str, PipelineStep[InputT, OutputT]],
        default: Optional[str] = None,
        description: str = "Conditional branching",
    ):
        super().__init__()
        self.name = name
        self.description = description
        self._condition = condition
        self._branches = branches
        self._default = default

    def execute(self, data: InputT) -> StepResult[OutputT]:
        try:
            branch_key = self._condition(data)
            step = self._branches.get(branch_key)

            if step is None and self._default:
                step = self._branches.get(self._default)

            if step is None:
                return StepResult.fail(
                    f"No branch found for key '{branch_key}' and no default set"
                )

            return step.run(data)

        except Exception as e:
            return StepResult.fail(str(e))

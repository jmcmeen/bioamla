# controllers/workflow.py
"""
Workflow Controller
===================

Controller for workflow operations.

Orchestrates between CLI/API views and core workflow engine.
Handles parsing, validation, execution, and export of workflows.

Example:
    from bioamla.controllers.workflow import WorkflowController

    controller = WorkflowController()

    # Parse and validate a workflow
    result = controller.parse("workflow.toml")

    # Execute a workflow
    result = controller.execute("workflow.toml", variables={"input": "./audio"})

    # Export to shell script
    result = controller.export_to_shell("workflow.toml", "run_workflow.sh")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import BaseController, ControllerResult, ToDictMixin


@dataclass
class WorkflowSummary(ToDictMixin):
    """Summary of a parsed workflow."""

    name: str
    description: str
    version: str
    num_steps: int
    step_names: List[str]
    variables: Dict[str, Any]
    execution_order: List[str]


@dataclass
class ExecutionSummary(ToDictMixin):
    """Summary of workflow execution."""

    workflow_name: str
    execution_id: str
    status: str
    total_duration: float
    steps_completed: int
    steps_failed: int
    steps_skipped: int
    errors: List[str] = field(default_factory=list)


@dataclass
class ValidationSummary(ToDictMixin):
    """Summary of workflow validation."""

    valid: bool
    num_errors: int
    num_warnings: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class WorkflowController(BaseController):
    """
    Controller for workflow operations.

    Provides high-level methods for:
    - Parsing workflow TOML files
    - Validating workflow definitions
    - Executing workflows with progress tracking
    - Exporting workflows to shell scripts
    """

    def __init__(self):
        """Initialize workflow controller."""
        super().__init__()
        self._engine = None
        self._progress_callback: Optional[Callable[[str, int, int, Optional[str]], None]] = None

    def _get_engine(self):
        """Get or create workflow engine."""
        if self._engine is None:
            from bioamla.core.workflow.engine import WorkflowEngine

            self._engine = WorkflowEngine()

            # Set progress callback if we have one
            if self._progress_callback:
                self._engine.set_progress_callback(self._progress_callback)

        return self._engine

    def set_execution_progress_callback(
        self,
        callback: Callable[[str, int, int, Optional[str]], None],
    ) -> None:
        """
        Set callback for execution progress updates.

        Callback signature: (step_name, current, total, status) -> None
        """
        self._progress_callback = callback
        if self._engine:
            self._engine.set_progress_callback(callback)

    # =========================================================================
    # Parsing
    # =========================================================================

    def parse(
        self,
        filepath: str,
        render_templates: bool = True,
    ) -> ControllerResult[WorkflowSummary]:
        """
        Parse a workflow TOML file.

        Args:
            filepath: Path to workflow TOML file
            render_templates: Whether to render Jinja2 templates

        Returns:
            Result with workflow summary
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.workflow.parser import parse_workflow

            workflow = parse_workflow(filepath, render_templates=render_templates)
            execution_order = workflow.get_execution_order()

            summary = WorkflowSummary(
                name=workflow.name,
                description=workflow.description,
                version=workflow.version,
                num_steps=len(workflow.steps),
                step_names=[step.name for step in workflow.steps],
                variables=workflow.variables,
                execution_order=execution_order,
            )

            return ControllerResult.ok(
                data=summary,
                message=f"Parsed workflow: {workflow.name}",
                workflow=workflow,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to parse workflow: {e}")

    def parse_string(
        self,
        content: str,
        render_templates: bool = True,
    ) -> ControllerResult[WorkflowSummary]:
        """
        Parse workflow from TOML string.

        Args:
            content: TOML workflow content
            render_templates: Whether to render Jinja2 templates

        Returns:
            Result with workflow summary
        """
        try:
            from bioamla.core.workflow.parser import parse_workflow_string

            workflow = parse_workflow_string(content, render_templates=render_templates)
            execution_order = workflow.get_execution_order()

            summary = WorkflowSummary(
                name=workflow.name,
                description=workflow.description,
                version=workflow.version,
                num_steps=len(workflow.steps),
                step_names=[step.name for step in workflow.steps],
                variables=workflow.variables,
                execution_order=execution_order,
            )

            return ControllerResult.ok(
                data=summary,
                message=f"Parsed workflow: {workflow.name}",
                workflow=workflow,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to parse workflow: {e}")

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(
        self,
        filepath: str,
        strict: bool = False,
    ) -> ControllerResult[ValidationSummary]:
        """
        Validate a workflow TOML file.

        Args:
            filepath: Path to workflow TOML file
            strict: If True, treat warnings as errors

        Returns:
            Result with validation summary
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.workflow.parser import parse_workflow
            from bioamla.core.workflow.validator import validate_workflow

            workflow = parse_workflow(filepath, render_templates=False)
            result = validate_workflow(workflow, strict=strict)

            summary = ValidationSummary(
                valid=result.valid,
                num_errors=len(result.errors),
                num_warnings=len(result.warnings),
                errors=[str(e) for e in result.errors],
                warnings=[str(w) for w in result.warnings],
            )

            if result.valid:
                message = "Workflow is valid"
                if result.warnings:
                    message += f" ({len(result.warnings)} warnings)"
            else:
                message = f"Workflow has {len(result.errors)} errors"

            return ControllerResult.ok(
                data=summary,
                message=message,
                warnings=summary.warnings,
                validation_result=result,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to validate workflow: {e}")

    def validate_workflow(
        self,
        workflow,
        strict: bool = False,
    ) -> ControllerResult[ValidationSummary]:
        """
        Validate a parsed Workflow object.

        Args:
            workflow: Workflow object to validate
            strict: If True, treat warnings as errors

        Returns:
            Result with validation summary
        """
        try:
            from bioamla.core.workflow.validator import validate_workflow

            result = validate_workflow(workflow, strict=strict)

            summary = ValidationSummary(
                valid=result.valid,
                num_errors=len(result.errors),
                num_warnings=len(result.warnings),
                errors=[str(e) for e in result.errors],
                warnings=[str(w) for w in result.warnings],
            )

            if result.valid:
                message = "Workflow is valid"
            else:
                message = f"Workflow has {len(result.errors)} errors"

            return ControllerResult.ok(
                data=summary,
                message=message,
                warnings=summary.warnings,
                validation_result=result,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to validate workflow: {e}")

    # =========================================================================
    # Execution
    # =========================================================================

    def execute(
        self,
        filepath: str,
        variables: Optional[Dict[str, Any]] = None,
        validate_first: bool = True,
    ) -> ControllerResult[ExecutionSummary]:
        """
        Execute a workflow from a TOML file.

        Args:
            filepath: Path to workflow TOML file
            variables: Override variables
            validate_first: Whether to validate before execution

        Returns:
            Result with execution summary
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.workflow.parser import parse_workflow
            from bioamla.core.workflow.engine import ExecutionStatus

            workflow = parse_workflow(filepath)

            # Optionally validate first
            if validate_first:
                validation = self.validate_workflow(workflow)
                if validation.data and not validation.data.valid:
                    return ControllerResult.fail(
                        f"Workflow validation failed: {validation.data.errors[0]}"
                    )

            engine = self._get_engine()
            result = engine.execute(workflow, variables=variables)

            # Count skipped steps
            steps_skipped = sum(
                1 for r in result.step_results if r.status == ExecutionStatus.SKIPPED
            )

            # Collect errors
            errors = []
            for step_result in result.step_results:
                if step_result.error:
                    errors.append(f"{step_result.step_name}: {step_result.error}")

            summary = ExecutionSummary(
                workflow_name=result.workflow_name,
                execution_id=str(result.execution_id),
                status=result.status.value,
                total_duration=result.total_duration_seconds,
                steps_completed=result.steps_completed,
                steps_failed=result.steps_failed,
                steps_skipped=steps_skipped,
                errors=errors,
            )

            if result.success:
                message = f"Workflow completed successfully in {result.total_duration_seconds:.2f}s"
            else:
                message = f"Workflow failed: {result.error}"

            return ControllerResult.ok(
                data=summary,
                message=message,
                execution_result=result,
                outputs=result.outputs,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to execute workflow: {e}")

    def execute_workflow(
        self,
        workflow,
        variables: Optional[Dict[str, Any]] = None,
    ) -> ControllerResult[ExecutionSummary]:
        """
        Execute a parsed Workflow object.

        Args:
            workflow: Workflow object to execute
            variables: Override variables

        Returns:
            Result with execution summary
        """
        try:
            from bioamla.core.workflow.engine import ExecutionStatus

            engine = self._get_engine()
            result = engine.execute(workflow, variables=variables)

            steps_skipped = sum(
                1 for r in result.step_results if r.status == ExecutionStatus.SKIPPED
            )

            errors = []
            for step_result in result.step_results:
                if step_result.error:
                    errors.append(f"{step_result.step_name}: {step_result.error}")

            summary = ExecutionSummary(
                workflow_name=result.workflow_name,
                execution_id=str(result.execution_id),
                status=result.status.value,
                total_duration=result.total_duration_seconds,
                steps_completed=result.steps_completed,
                steps_failed=result.steps_failed,
                steps_skipped=steps_skipped,
                errors=errors,
            )

            if result.success:
                message = f"Workflow completed in {result.total_duration_seconds:.2f}s"
            else:
                message = f"Workflow failed: {result.error}"

            return ControllerResult.ok(
                data=summary,
                message=message,
                execution_result=result,
                outputs=result.outputs,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to execute workflow: {e}")

    def cancel(self) -> None:
        """Cancel a running workflow execution."""
        if self._engine:
            self._engine.cancel()

    # =========================================================================
    # Action Registration
    # =========================================================================

    def register_action(
        self,
        name: str,
        handler: Callable,
    ) -> ControllerResult[None]:
        """
        Register a custom action handler.

        Args:
            name: Action name (e.g., "custom.my_action")
            handler: Function to handle the action

        Returns:
            Result indicating success
        """
        try:
            engine = self._get_engine()
            engine.register_action(name, handler)
            return ControllerResult.ok(
                message=f"Registered action: {name}",
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to register action: {e}")

    def list_actions(self) -> ControllerResult[List[str]]:
        """
        List all registered actions.

        Returns:
            Result with list of action names
        """
        try:
            engine = self._get_engine()
            actions = list(engine._actions.keys())
            return ControllerResult.ok(
                data=sorted(actions),
                message=f"Found {len(actions)} registered actions",
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to list actions: {e}")

    # =========================================================================
    # Export
    # =========================================================================

    def export_to_shell(
        self,
        filepath: str,
        output_path: Optional[str] = None,
    ) -> ControllerResult[str]:
        """
        Export a workflow to a shell script.

        Args:
            filepath: Path to workflow TOML file
            output_path: Path to save shell script (optional)

        Returns:
            Result with shell script content
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.workflow.parser import parse_workflow

            workflow = parse_workflow(filepath)
            engine = self._get_engine()
            script = engine.export_to_shell(workflow, output_path=output_path)

            message = "Generated shell script"
            if output_path:
                message += f": {output_path}"

            return ControllerResult.ok(
                data=script,
                message=message,
                output_path=output_path,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to export workflow: {e}")

    def export_to_toml(
        self,
        workflow,
        output_path: Optional[str] = None,
    ) -> ControllerResult[str]:
        """
        Export a Workflow object to TOML.

        Args:
            workflow: Workflow object to export
            output_path: Path to save TOML file (optional)

        Returns:
            Result with TOML content
        """
        try:
            from bioamla.core.workflow.parser import workflow_to_toml

            toml_content = workflow_to_toml(workflow)

            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(toml_content)

            message = "Generated TOML"
            if output_path:
                message += f": {output_path}"

            return ControllerResult.ok(
                data=toml_content,
                message=message,
                output_path=output_path,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to export workflow: {e}")

    # =========================================================================
    # Workflow Creation Helpers
    # =========================================================================

    def create_workflow(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        description: str = "",
        variables: Optional[Dict[str, Any]] = None,
    ) -> ControllerResult:
        """
        Create a Workflow object programmatically.

        Args:
            name: Workflow name
            steps: List of step dictionaries with 'name', 'action', 'params'
            description: Workflow description
            variables: Workflow variables

        Returns:
            Result with created Workflow object
        """
        try:
            from bioamla.core.workflow.parser import Workflow, WorkflowStep

            workflow_steps = []
            for step_dict in steps:
                step = WorkflowStep(
                    name=step_dict.get("name", ""),
                    action=step_dict.get("action", ""),
                    params=step_dict.get("params", {}),
                    depends_on=step_dict.get("depends_on", []),
                    condition=step_dict.get("condition"),
                    description=step_dict.get("description", ""),
                    on_error=step_dict.get("on_error", "fail"),
                    timeout=step_dict.get("timeout"),
                    retry=step_dict.get("retry", 0),
                )
                workflow_steps.append(step)

            workflow = Workflow(
                name=name,
                description=description,
                steps=workflow_steps,
                variables=variables or {},
            )

            return ControllerResult.ok(
                data=workflow,
                message=f"Created workflow: {name}",
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to create workflow: {e}")

    def get_example_workflow(self) -> ControllerResult[str]:
        """
        Get an example workflow TOML.

        Returns:
            Result with example TOML content
        """
        example = """[workflow]
name = "bioacoustic_analysis"
description = "Example bioacoustic analysis workflow"
version = "1.0.0"

[variables]
input_dir = "./audio"
output_dir = "./results"
sample_rate = 16000

[[steps]]
name = "resample"
action = "audio.resample"
description = "Resample audio to target sample rate"

[steps.params]
input = "{{ input_dir }}"
output = "{{ output_dir }}/resampled"
sample_rate = "{{ sample_rate }}"

[[steps]]
name = "indices"
action = "analysis.indices"
description = "Calculate acoustic indices"
depends_on = ["resample"]

[steps.params]
input = "{{ output_dir }}/resampled"
output = "{{ output_dir }}/indices.csv"

[[steps]]
name = "embed"
action = "inference.embed"
description = "Extract embeddings"
depends_on = ["resample"]

[steps.params]
input = "{{ output_dir }}/resampled"
output = "{{ output_dir }}/embeddings.npy"

[[steps]]
name = "cluster"
action = "analysis.cluster"
description = "Cluster embeddings"
depends_on = ["embed"]

[steps.params]
embeddings = "{{ output_dir }}/embeddings.npy"
output = "{{ output_dir }}/clusters.json"
method = "hdbscan"
"""
        return ControllerResult.ok(
            data=example,
            message="Example workflow",
        )

# controllers/pipeline.py
"""
Pipeline Controller
===================

Controller for pipeline operations.

Orchestrates between CLI/API views and core pipeline engine.
Handles parsing, validation, execution, and export of pipelines.

Example:
    from bioamla.controllers.pipeline import PipelineController

    controller = PipelineController()

    # Parse and validate a pipeline
    result = controller.parse("pipeline.toml")

    # Execute a pipeline
    result = controller.execute("pipeline.toml", variables={"input": "./audio"})

    # Export to shell script
    result = controller.export_to_shell("pipeline.toml", "run_pipeline.sh")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import BaseController, ControllerResult, ToDictMixin


@dataclass
class PipelineSummary(ToDictMixin):
    """Summary of a parsed pipeline."""

    name: str
    description: str
    version: str
    num_steps: int
    step_names: List[str]
    variables: Dict[str, Any]
    execution_order: List[str]


@dataclass
class ExecutionSummary(ToDictMixin):
    """Summary of pipeline execution."""

    pipeline_name: str
    execution_id: str
    status: str
    total_duration: float
    steps_completed: int
    steps_failed: int
    steps_skipped: int
    errors: List[str] = field(default_factory=list)


@dataclass
class ValidationSummary(ToDictMixin):
    """Summary of pipeline validation."""

    valid: bool
    num_errors: int
    num_warnings: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PipelineController(BaseController):
    """
    Controller for pipeline operations.

    Provides high-level methods for:
    - Parsing pipeline TOML files
    - Validating pipeline definitions
    - Executing pipelines with progress tracking
    - Exporting pipelines to shell scripts
    """

    def __init__(self):
        """Initialize pipeline controller."""
        super().__init__()
        self._engine = None
        self._progress_callback: Optional[Callable[[str, int, int, Optional[str]], None]] = None

    def _get_engine(self):
        """Get or create pipeline engine."""
        if self._engine is None:
            from bioamla.core.pipeline.engine import PipelineEngine

            self._engine = PipelineEngine()

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
    ) -> ControllerResult[PipelineSummary]:
        """
        Parse a pipeline TOML file.

        Args:
            filepath: Path to pipeline TOML file
            render_templates: Whether to render Jinja2 templates

        Returns:
            Result with pipeline summary
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.pipeline.parser import parse_pipeline

            pipeline = parse_pipeline(filepath, render_templates=render_templates)
            execution_order = pipeline.get_execution_order()

            summary = PipelineSummary(
                name=pipeline.name,
                description=pipeline.description,
                version=pipeline.version,
                num_steps=len(pipeline.steps),
                step_names=[step.name for step in pipeline.steps],
                variables=pipeline.variables,
                execution_order=execution_order,
            )

            return ControllerResult.ok(
                data=summary,
                message=f"Parsed pipeline: {pipeline.name}",
                pipeline=pipeline,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to parse pipeline: {e}")

    def parse_string(
        self,
        content: str,
        render_templates: bool = True,
    ) -> ControllerResult[PipelineSummary]:
        """
        Parse pipeline from TOML string.

        Args:
            content: TOML pipeline content
            render_templates: Whether to render Jinja2 templates

        Returns:
            Result with pipeline summary
        """
        try:
            from bioamla.core.pipeline.parser import parse_pipeline_string

            pipeline = parse_pipeline_string(content, render_templates=render_templates)
            execution_order = pipeline.get_execution_order()

            summary = PipelineSummary(
                name=pipeline.name,
                description=pipeline.description,
                version=pipeline.version,
                num_steps=len(pipeline.steps),
                step_names=[step.name for step in pipeline.steps],
                variables=pipeline.variables,
                execution_order=execution_order,
            )

            return ControllerResult.ok(
                data=summary,
                message=f"Parsed pipeline: {pipeline.name}",
                pipeline=pipeline,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to parse pipeline: {e}")

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(
        self,
        filepath: str,
        strict: bool = False,
    ) -> ControllerResult[ValidationSummary]:
        """
        Validate a pipeline TOML file.

        Args:
            filepath: Path to pipeline TOML file
            strict: If True, treat warnings as errors

        Returns:
            Result with validation summary
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.pipeline.parser import parse_pipeline
            from bioamla.core.pipeline.validator import validate_pipeline

            pipeline = parse_pipeline(filepath, render_templates=False)
            result = validate_pipeline(pipeline, strict=strict)

            summary = ValidationSummary(
                valid=result.valid,
                num_errors=len(result.errors),
                num_warnings=len(result.warnings),
                errors=[str(e) for e in result.errors],
                warnings=[str(w) for w in result.warnings],
            )

            if result.valid:
                message = "Pipeline is valid"
                if result.warnings:
                    message += f" ({len(result.warnings)} warnings)"
            else:
                message = f"Pipeline has {len(result.errors)} errors"

            return ControllerResult.ok(
                data=summary,
                message=message,
                warnings=summary.warnings,
                validation_result=result,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to validate pipeline: {e}")

    def validate_pipeline(
        self,
        pipeline,
        strict: bool = False,
    ) -> ControllerResult[ValidationSummary]:
        """
        Validate a parsed Pipeline object.

        Args:
            pipeline: Pipeline object to validate
            strict: If True, treat warnings as errors

        Returns:
            Result with validation summary
        """
        try:
            from bioamla.core.pipeline.validator import validate_pipeline

            result = validate_pipeline(pipeline, strict=strict)

            summary = ValidationSummary(
                valid=result.valid,
                num_errors=len(result.errors),
                num_warnings=len(result.warnings),
                errors=[str(e) for e in result.errors],
                warnings=[str(w) for w in result.warnings],
            )

            if result.valid:
                message = "Pipeline is valid"
            else:
                message = f"Pipeline has {len(result.errors)} errors"

            return ControllerResult.ok(
                data=summary,
                message=message,
                warnings=summary.warnings,
                validation_result=result,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to validate pipeline: {e}")

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
        Execute a pipeline from a TOML file.

        Args:
            filepath: Path to pipeline TOML file
            variables: Override variables
            validate_first: Whether to validate before execution

        Returns:
            Result with execution summary
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.pipeline.engine import ExecutionStatus
            from bioamla.core.pipeline.parser import parse_pipeline

            pipeline = parse_pipeline(filepath)

            # Optionally validate first
            if validate_first:
                validation = self.validate_pipeline(pipeline)
                if validation.data and not validation.data.valid:
                    return ControllerResult.fail(
                        f"Pipeline validation failed: {validation.data.errors[0]}"
                    )

            engine = self._get_engine()
            result = engine.execute(pipeline, variables=variables)

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
                pipeline_name=result.pipeline_name,
                execution_id=str(result.execution_id),
                status=result.status.value,
                total_duration=result.total_duration_seconds,
                steps_completed=result.steps_completed,
                steps_failed=result.steps_failed,
                steps_skipped=steps_skipped,
                errors=errors,
            )

            if result.success:
                message = f"Pipeline completed successfully in {result.total_duration_seconds:.2f}s"
            else:
                message = f"Pipeline failed: {result.error}"

            return ControllerResult.ok(
                data=summary,
                message=message,
                execution_result=result,
                outputs=result.outputs,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to execute pipeline: {e}")

    def execute_pipeline(
        self,
        pipeline,
        variables: Optional[Dict[str, Any]] = None,
    ) -> ControllerResult[ExecutionSummary]:
        """
        Execute a parsed Pipeline object.

        Args:
            pipeline: Pipeline object to execute
            variables: Override variables

        Returns:
            Result with execution summary
        """
        try:
            from bioamla.core.pipeline.engine import ExecutionStatus

            engine = self._get_engine()
            result = engine.execute(pipeline, variables=variables)

            steps_skipped = sum(
                1 for r in result.step_results if r.status == ExecutionStatus.SKIPPED
            )

            errors = []
            for step_result in result.step_results:
                if step_result.error:
                    errors.append(f"{step_result.step_name}: {step_result.error}")

            summary = ExecutionSummary(
                pipeline_name=result.pipeline_name,
                execution_id=str(result.execution_id),
                status=result.status.value,
                total_duration=result.total_duration_seconds,
                steps_completed=result.steps_completed,
                steps_failed=result.steps_failed,
                steps_skipped=steps_skipped,
                errors=errors,
            )

            if result.success:
                message = f"Pipeline completed in {result.total_duration_seconds:.2f}s"
            else:
                message = f"Pipeline failed: {result.error}"

            return ControllerResult.ok(
                data=summary,
                message=message,
                execution_result=result,
                outputs=result.outputs,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to execute pipeline: {e}")

    def cancel(self) -> None:
        """Cancel a running pipeline execution."""
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
        Export a pipeline to a shell script.

        Args:
            filepath: Path to pipeline TOML file
            output_path: Path to save shell script (optional)

        Returns:
            Result with shell script content
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.pipeline.parser import parse_pipeline

            pipeline = parse_pipeline(filepath)
            engine = self._get_engine()
            script = engine.export_to_shell(pipeline, output_path=output_path)

            message = "Generated shell script"
            if output_path:
                message += f": {output_path}"

            return ControllerResult.ok(
                data=script,
                message=message,
                output_path=output_path,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to export pipeline: {e}")

    def export_to_toml(
        self,
        pipeline,
        output_path: Optional[str] = None,
    ) -> ControllerResult[str]:
        """
        Export a Pipeline object to TOML.

        Args:
            pipeline: Pipeline object to export
            output_path: Path to save TOML file (optional)

        Returns:
            Result with TOML content
        """
        try:
            from bioamla.core.pipeline.parser import pipeline_to_toml

            toml_content = pipeline_to_toml(pipeline)

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
            return ControllerResult.fail(f"Failed to export pipeline: {e}")

    # =========================================================================
    # Pipeline Creation Helpers
    # =========================================================================

    def create_pipeline(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        description: str = "",
        variables: Optional[Dict[str, Any]] = None,
    ) -> ControllerResult:
        """
        Create a Pipeline object programmatically.

        Args:
            name: Pipeline name
            steps: List of step dictionaries with 'name', 'action', 'params'
            description: Pipeline description
            variables: Pipeline variables

        Returns:
            Result with created Pipeline object
        """
        try:
            from bioamla.core.pipeline.parser import Pipeline, PipelineStep

            pipeline_steps = []
            for step_dict in steps:
                step = PipelineStep(
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
                pipeline_steps.append(step)

            pipeline = Pipeline(
                name=name,
                description=description,
                steps=pipeline_steps,
                variables=variables or {},
            )

            return ControllerResult.ok(
                data=pipeline,
                message=f"Created pipeline: {name}",
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to create pipeline: {e}")

    def get_example_pipeline(self) -> ControllerResult[str]:
        """
        Get an example pipeline TOML.

        Returns:
            Result with example TOML content
        """
        example = """[pipeline]
name = "bioacoustic_analysis"
description = "Example bioacoustic analysis pipeline"
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
            message="Example pipeline",
        )

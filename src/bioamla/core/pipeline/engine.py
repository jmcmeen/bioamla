# core/pipeline/engine.py
"""
Pipeline Execution Engine
=========================

Executes TOML-defined pipelines with support for:
- Dependency resolution and parallel execution
- Progress tracking and callbacks
- Error handling and retry logic
- State persistence and resumption
- Shell script export

Example:
    from bioamla.core.pipeline.engine import PipelineEngine
    from bioamla.core.pipeline.parser import parse_pipeline

    pipeline = parse_pipeline("pipeline.toml")
    engine = PipelineEngine()

    # Execute pipeline
    result = engine.execute(pipeline)

    # Export to shell script
    script = engine.export_to_shell(pipeline)
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID, uuid4

from bioamla.core.logger import get_logger

from .parser import Pipeline, PipelineStep

logger = get_logger(__name__)

__all__ = [
    "PipelineEngine",
    "ExecutionResult",
    "StepExecutionResult",
    "ExecutionStatus",
    "ExecutionContext",
]


class ExecutionStatus(Enum):
    """Status of pipeline or step execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class StepExecutionResult:
    """Result of executing a single step."""

    step_name: str
    action: str
    status: ExecutionStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    output: Any = None
    error: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_name": self.step_name,
            "action": self.action,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class ExecutionResult:
    """Result of pipeline execution."""

    pipeline_name: str
    execution_id: UUID
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    step_results: List[StepExecutionResult] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def steps_completed(self) -> int:
        """Number of completed steps."""
        return sum(1 for r in self.step_results if r.status == ExecutionStatus.COMPLETED)

    @property
    def steps_failed(self) -> int:
        """Number of failed steps."""
        return sum(1 for r in self.step_results if r.status == ExecutionStatus.FAILED)

    @property
    def success(self) -> bool:
        """Whether execution was successful."""
        return self.status == ExecutionStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pipeline_name": self.pipeline_name,
            "execution_id": str(self.execution_id),
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_seconds": self.total_duration_seconds,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "step_results": [r.to_dict() for r in self.step_results],
            "error": self.error,
        }

    def get_step_result(self, step_name: str) -> Optional[StepExecutionResult]:
        """Get result for a specific step."""
        for result in self.step_results:
            if result.step_name == step_name:
                return result
        return None


@dataclass
class ExecutionContext:
    """Context passed to action handlers during execution."""

    pipeline: Pipeline
    step: PipelineStep
    variables: Dict[str, Any]
    outputs: Dict[str, Any]
    execution_id: UUID
    step_index: int
    total_steps: int


# Type for action handlers
ActionHandler = Callable[[ExecutionContext, Dict[str, Any]], Any]

# Type for progress callbacks
ProgressCallback = Callable[[str, int, int, Optional[str]], None]


class PipelineEngine:
    """
    Engine for executing TOML-defined pipelines.

    The engine provides:
    - Action registration for custom operations
    - Built-in actions for common bioamla operations
    - Progress tracking and callbacks
    - Error handling with retry support
    - Shell script export
    """

    def __init__(self):
        """Initialize the pipeline engine."""
        self._actions: Dict[str, ActionHandler] = {}
        self._progress_callback: Optional[ProgressCallback] = None
        self._cancelled = False

        # Register built-in actions
        self._register_builtin_actions()

    def register_action(self, name: str, handler: ActionHandler) -> None:
        """
        Register an action handler.

        Args:
            name: Action name (e.g., "audio.resample")
            handler: Function to execute the action
        """
        self._actions[name] = handler
        logger.debug(f"Registered action: {name}")

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """
        Set callback for progress updates.

        Callback signature: (step_name, current, total, status) -> None
        """
        self._progress_callback = callback

    def cancel(self) -> None:
        """Cancel pipeline execution."""
        self._cancelled = True

    def execute(
        self,
        pipeline: Pipeline,
        variables: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute a pipeline.

        Args:
            pipeline: Pipeline to execute
            variables: Override variables

        Returns:
            ExecutionResult with execution details
        """
        self._cancelled = False
        execution_id = uuid4()
        started_at = datetime.utcnow()

        # Merge variables
        all_variables = {**pipeline.variables}
        if variables:
            all_variables.update(variables)

        # Get execution order
        try:
            execution_order = pipeline.get_execution_order()
        except ValueError as e:
            return ExecutionResult(
                pipeline_name=pipeline.name,
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                error=str(e),
            )

        total_steps = len(execution_order)
        step_results: List[StepExecutionResult] = []
        outputs: Dict[str, Any] = {}
        final_status = ExecutionStatus.COMPLETED
        final_error = None

        for idx, step_name in enumerate(execution_order):
            if self._cancelled:
                final_status = ExecutionStatus.CANCELLED
                final_error = "Execution cancelled by user"
                break

            step = pipeline.get_step(step_name)
            if step is None:
                continue

            # Report progress
            if self._progress_callback:
                self._progress_callback(step_name, idx + 1, total_steps, "running")

            # Create context
            context = ExecutionContext(
                pipeline=pipeline,
                step=step,
                variables=all_variables,
                outputs=outputs,
                execution_id=execution_id,
                step_index=idx,
                total_steps=total_steps,
            )

            # Execute step
            result = self._execute_step(step, context)
            step_results.append(result)

            # Store output
            if result.output is not None:
                outputs[step_name] = result.output

            # Handle failure
            if result.status == ExecutionStatus.FAILED:
                if step.on_error == "fail":
                    final_status = ExecutionStatus.FAILED
                    final_error = f"Step '{step_name}' failed: {result.error}"
                    break
                elif step.on_error == "skip":
                    logger.warning(f"Step '{step_name}' failed, skipping: {result.error}")
                # "continue" just continues

            # Report progress
            if self._progress_callback:
                status = "completed" if result.status == ExecutionStatus.COMPLETED else "failed"
                self._progress_callback(step_name, idx + 1, total_steps, status)

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        return ExecutionResult(
            pipeline_name=pipeline.name,
            execution_id=execution_id,
            status=final_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            step_results=step_results,
            outputs=outputs,
            error=final_error,
        )

    def _execute_step(
        self,
        step: PipelineStep,
        context: ExecutionContext,
    ) -> StepExecutionResult:
        """Execute a single step."""
        started_at = datetime.utcnow()
        result = StepExecutionResult(
            step_name=step.name,
            action=step.action,
            status=ExecutionStatus.RUNNING,
            started_at=started_at,
        )

        # Check condition
        if step.condition:
            try:
                should_run = self._evaluate_condition(step.condition, context)
                if not should_run:
                    result.status = ExecutionStatus.SKIPPED
                    result.completed_at = datetime.utcnow()
                    logger.info(f"Skipping step '{step.name}': condition not met")
                    return result
            except Exception as e:
                result.status = ExecutionStatus.FAILED
                result.error = f"Condition evaluation failed: {e}"
                result.completed_at = datetime.utcnow()
                return result

        # Get action handler
        handler = self._actions.get(step.action)
        if handler is None:
            result.status = ExecutionStatus.FAILED
            result.error = f"Unknown action: {step.action}"
            result.completed_at = datetime.utcnow()
            return result

        # Execute with retry
        max_attempts = step.retry + 1
        last_error = None

        for attempt in range(max_attempts):
            try:
                output = handler(context, step.params)
                result.status = ExecutionStatus.COMPLETED
                result.output = output
                result.retry_count = attempt
                break
            except Exception as e:
                last_error = str(e)
                result.retry_count = attempt + 1
                logger.warning(
                    f"Step '{step.name}' failed (attempt {attempt + 1}/{max_attempts}): {e}"
                )
                if attempt < max_attempts - 1:
                    time.sleep(1)  # Brief delay before retry

        if result.status != ExecutionStatus.COMPLETED:
            result.status = ExecutionStatus.FAILED
            result.error = last_error

        result.completed_at = datetime.utcnow()
        result.duration_seconds = (result.completed_at - result.started_at).total_seconds()

        return result

    def _evaluate_condition(
        self,
        condition: str,
        context: ExecutionContext,
    ) -> bool:
        """Evaluate a condition string."""
        try:
            from jinja2 import BaseLoader, Environment

            env = Environment(loader=BaseLoader())

            # Build evaluation context
            eval_context = {
                **context.variables,
                "outputs": context.outputs,
                "step": context.step.name,
            }

            # Render condition
            template = env.from_string("{{ " + condition + " }}")
            result = template.render(eval_context)

            # Evaluate result
            return result.lower() in ("true", "1", "yes")
        except Exception:
            # Default to True if we can't evaluate
            return True

    def _register_builtin_actions(self) -> None:
        """Register built-in action handlers."""
        # Audio actions
        self.register_action("audio.resample", self._action_audio_resample)
        self.register_action("audio.normalize", self._action_audio_normalize)
        self.register_action("audio.filter", self._action_audio_filter)
        self.register_action("audio.trim", self._action_audio_trim)
        self.register_action("audio.denoise", self._action_audio_denoise)

        # Analysis actions
        self.register_action("analysis.indices", self._action_analysis_indices)
        self.register_action("analysis.cluster", self._action_analysis_cluster)

        # Inference actions
        self.register_action("inference.predict", self._action_inference_predict)
        self.register_action("inference.embed", self._action_inference_embed)

        # Detection actions
        self.register_action("detection.ribbit", self._action_detection_ribbit)

        # Utility actions
        self.register_action("util.copy", self._action_util_copy)
        self.register_action("util.move", self._action_util_move)
        self.register_action("util.log", self._action_util_log)

    # =========================================================================
    # Built-in Action Handlers
    # =========================================================================

    def _action_audio_resample(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Resample audio files."""
        from bioamla.controllers.audio_file import AudioFileController

        controller = AudioFileController()
        result = controller.resample(
            input_path=params.get("input"),
            output_path=params.get("output"),
            target_sample_rate=int(params.get("sample_rate", 16000)),
        )
        if not result.success:
            raise RuntimeError(result.error)
        return result.data

    def _action_audio_normalize(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Normalize audio files.

        Params:
            input: Input file or directory path
            output: Output file or directory path
            target_db: Target loudness in dB (default: -20.0)
            peak: Use peak normalization instead of RMS (default: false)
        """
        from bioamla.controllers.audio import AudioController

        controller = AudioController()
        input_path = Path(params.get("input", ""))
        output_path = Path(params.get("output", ""))
        target_db = float(params.get("target_db", -20.0))
        peak = params.get("peak", False)

        if input_path.is_dir():
            # Batch processing
            output_path.mkdir(parents=True, exist_ok=True)
            audio_files = list(input_path.glob("**/*.wav")) + list(input_path.glob("**/*.mp3"))
            processed = 0
            for audio_file in audio_files:
                rel_path = audio_file.relative_to(input_path)
                out_file = output_path / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)
                result = controller.normalize(
                    str(audio_file), str(out_file), target_db=target_db, peak=peak
                )
                if result.success:
                    processed += 1
            return {"processed": processed, "total": len(audio_files)}
        else:
            # Single file
            result = controller.normalize(
                str(input_path), str(output_path), target_db=target_db, peak=peak
            )
            if not result.success:
                raise RuntimeError(result.error)
            return result.data

    def _action_audio_filter(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Apply frequency filter to audio.

        Params:
            input: Input file or directory path
            output: Output file or directory path
            lowpass: Lowpass cutoff frequency in Hz (optional)
            highpass: Highpass cutoff frequency in Hz (optional)
            bandpass: Comma-separated low,high frequencies for bandpass (optional)
            order: Filter order (default: 5)
        """
        from bioamla.controllers.audio import AudioController

        controller = AudioController()
        input_path = Path(params.get("input", ""))
        output_path = Path(params.get("output", ""))
        lowpass = params.get("lowpass")
        highpass = params.get("highpass")
        bandpass_str = params.get("bandpass")
        order = int(params.get("order", 5))

        # Parse bandpass if provided as string
        bandpass = None
        if bandpass_str:
            if isinstance(bandpass_str, str):
                parts = bandpass_str.split(",")
                bandpass = (float(parts[0]), float(parts[1]))
            elif isinstance(bandpass_str, (list, tuple)):
                bandpass = (float(bandpass_str[0]), float(bandpass_str[1]))

        if lowpass:
            lowpass = float(lowpass)
        if highpass:
            highpass = float(highpass)

        if input_path.is_dir():
            # Batch processing
            output_path.mkdir(parents=True, exist_ok=True)
            audio_files = list(input_path.glob("**/*.wav")) + list(input_path.glob("**/*.mp3"))
            processed = 0
            for audio_file in audio_files:
                rel_path = audio_file.relative_to(input_path)
                out_file = output_path / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)
                result = controller.filter_audio(
                    str(audio_file),
                    str(out_file),
                    lowpass=lowpass,
                    highpass=highpass,
                    bandpass=bandpass,
                    order=order,
                )
                if result.success:
                    processed += 1
            return {"processed": processed, "total": len(audio_files)}
        else:
            # Single file
            result = controller.filter_audio(
                str(input_path),
                str(output_path),
                lowpass=lowpass,
                highpass=highpass,
                bandpass=bandpass,
                order=order,
            )
            if not result.success:
                raise RuntimeError(result.error)
            return result.data

    def _action_audio_trim(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Trim audio files.

        Params:
            input: Input file or directory path
            output: Output file or directory path
            start: Start time in seconds (optional)
            end: End time in seconds (optional)
            trim_silence: Trim silence from start/end (default: false)
            silence_threshold_db: Silence threshold in dB (default: -40.0)
        """
        from bioamla.controllers.audio import AudioController

        controller = AudioController()
        input_path = Path(params.get("input", ""))
        output_path = Path(params.get("output", ""))
        start = params.get("start")
        end = params.get("end")
        trim_silence = params.get("trim_silence", False)
        silence_threshold_db = float(params.get("silence_threshold_db", -40.0))

        if start is not None:
            start = float(start)
        if end is not None:
            end = float(end)

        if input_path.is_dir():
            # Batch processing
            output_path.mkdir(parents=True, exist_ok=True)
            audio_files = list(input_path.glob("**/*.wav")) + list(input_path.glob("**/*.mp3"))
            processed = 0
            for audio_file in audio_files:
                rel_path = audio_file.relative_to(input_path)
                out_file = output_path / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)
                result = controller.trim(
                    str(audio_file),
                    str(out_file),
                    start=start,
                    end=end,
                    trim_silence=trim_silence,
                    silence_threshold_db=silence_threshold_db,
                )
                if result.success:
                    processed += 1
            return {"processed": processed, "total": len(audio_files)}
        else:
            # Single file
            result = controller.trim(
                str(input_path),
                str(output_path),
                start=start,
                end=end,
                trim_silence=trim_silence,
                silence_threshold_db=silence_threshold_db,
            )
            if not result.success:
                raise RuntimeError(result.error)
            return result.data

    def _action_audio_denoise(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Denoise audio files.

        Params:
            input: Input file or directory path
            output: Output file or directory path
            strength: Noise reduction strength 0-2 (default: 1.0)
        """
        from bioamla.controllers.audio import AudioController

        controller = AudioController()
        input_path = Path(params.get("input", ""))
        output_path = Path(params.get("output", ""))
        strength = float(params.get("strength", 1.0))

        if input_path.is_dir():
            # Batch processing
            output_path.mkdir(parents=True, exist_ok=True)
            audio_files = list(input_path.glob("**/*.wav")) + list(input_path.glob("**/*.mp3"))
            processed = 0
            for audio_file in audio_files:
                rel_path = audio_file.relative_to(input_path)
                out_file = output_path / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)
                result = controller.denoise(str(audio_file), str(out_file), strength=strength)
                if result.success:
                    processed += 1
            return {"processed": processed, "total": len(audio_files)}
        else:
            # Single file
            result = controller.denoise(str(input_path), str(output_path), strength=strength)
            if not result.success:
                raise RuntimeError(result.error)
            return result.data

    def _action_analysis_indices(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Calculate acoustic indices."""
        from bioamla.controllers.indices import IndicesController

        controller = IndicesController()
        result = controller.calculate_batch(
            directory=params.get("input"),
            output_path=params.get("output"),
        )
        if not result.success:
            raise RuntimeError(result.error)
        return result.data

    def _action_analysis_cluster(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Cluster embeddings."""
        import numpy as np

        from bioamla.controllers.clustering import ClusteringController

        controller = ClusteringController()

        # Load embeddings
        embeddings_path = params.get("embeddings")
        embeddings = np.load(embeddings_path)

        result = controller.cluster(
            embeddings,
            method=params.get("method", "hdbscan"),
        )
        if not result.success:
            raise RuntimeError(result.error)
        return result.data

    def _action_inference_predict(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Run inference prediction."""
        from bioamla.controllers.inference import InferenceController

        controller = InferenceController(model_path=params.get("model"))
        result = controller.predict_batch(
            directory=params.get("input"),
            output_csv=params.get("output"),
        )
        if not result.success:
            raise RuntimeError(result.error)
        return result.data

    def _action_inference_embed(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Extract embeddings."""
        from bioamla.controllers.embedding import EmbeddingController

        controller = EmbeddingController(model_path=params.get("model"))
        result = controller.extract_batch(
            directory=params.get("input"),
            output_path=params.get("output"),
        )
        if not result.success:
            raise RuntimeError(result.error)
        return result.data

    def _action_detection_ribbit(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Run RIBBIT detection."""
        from bioamla.controllers.ribbit import RibbitController

        controller = RibbitController()
        result = controller.detect_batch(
            directory=params.get("input"),
            preset=params.get("preset"),
            output_csv=params.get("output"),
        )
        if not result.success:
            raise RuntimeError(result.error)
        return result.data

    def _action_util_copy(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Copy files."""
        import shutil

        src = Path(params.get("source"))
        dst = Path(params.get("destination"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return {"copied": str(dst)}

    def _action_util_move(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Move files."""
        import shutil

        src = Path(params.get("source"))
        dst = Path(params.get("destination"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        return {"moved": str(dst)}

    def _action_util_log(
        self,
        context: ExecutionContext,
        params: Dict[str, Any],
    ) -> Any:
        """Log a message."""
        message = params.get("message", "")
        level = params.get("level", "info")
        getattr(logger, level)(message)
        return {"logged": message}

    # =========================================================================
    # Export
    # =========================================================================

    def export_to_shell(
        self,
        pipeline: Pipeline,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Export pipeline to a shell script.

        Args:
            pipeline: Pipeline to export
            output_path: Optional path to save script

        Returns:
            Shell script content
        """
        lines = [
            "#!/bin/bash",
            f"# Pipeline: {pipeline.name}",
            f"# Description: {pipeline.description}",
            f"# Generated: {datetime.utcnow().isoformat()}",
            "",
            "set -e  # Exit on error",
            "",
        ]

        # Add variables
        if pipeline.variables:
            lines.append("# Variables")
            for key, value in pipeline.variables.items():
                if isinstance(value, str):
                    lines.append(f'{key}="{value}"')
                else:
                    lines.append(f"{key}={value}")
            lines.append("")

        # Add steps
        execution_order = pipeline.get_execution_order()

        for step_name in execution_order:
            step = pipeline.get_step(step_name)
            if step is None:
                continue

            lines.append(f"# Step: {step.name}")
            if step.description:
                lines.append(f"# {step.description}")

            # Convert action to bioamla command
            cmd = self._action_to_command(step)
            lines.append(cmd)
            lines.append("")

        script = "\n".join(lines)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(script)
            path.chmod(0o755)

        return script

    def _action_to_command(self, step: PipelineStep) -> str:
        """Convert a step action to a bioamla CLI command."""
        action = step.action
        params = step.params

        # Map actions to CLI commands
        if action == "audio.resample":
            return f'bioamla audio resample "{params.get("input")}" -o "{params.get("output")}" -r {params.get("sample_rate", 16000)}'
        elif action == "audio.normalize":
            return f'bioamla audio normalize "{params.get("input")}" -o "{params.get("output")}"'
        elif action == "analysis.indices":
            return f'bioamla analysis indices "{params.get("input")}" -o "{params.get("output")}"'
        elif action == "inference.predict":
            return f'bioamla ast predict "{params.get("input")}" -m "{params.get("model")}" -o "{params.get("output")}"'
        elif action == "detection.ribbit":
            preset = params.get("preset", "generic_mid_freq")
            return f'bioamla detect ribbit "{params.get("input")}" --preset {preset} -o "{params.get("output")}"'
        else:
            # Generic fallback
            param_str = " ".join(f'--{k}="{v}"' for k, v in params.items())
            return f"# bioamla {action.replace('.', ' ')} {param_str}"

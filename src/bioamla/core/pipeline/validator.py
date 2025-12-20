# core/pipeline/validator.py
"""
Pipeline Validator
==================

Schema validation for TOML pipeline definitions.

Provides comprehensive validation of:
- Pipeline structure and required fields
- Step configuration and parameters
- Dependency graph integrity
- Action validity
- Variable references

Example:
    from bioamla.core.pipeline.validator import (
        validate_pipeline,
        PipelineValidator,
    )
    from bioamla.core.pipeline.parser import parse_pipeline

    pipeline = parse_pipeline("pipeline.toml", render_templates=False)
    result = validate_pipeline(pipeline)

    if not result.valid:
        for error in result.errors:
            print(f"Error: {error}")
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from bioamla.core.logger import get_logger

from .parser import Pipeline, PipelineStep

logger = get_logger(__name__)

__all__ = [
    "ValidationResult",
    "ValidationError",
    "PipelineValidator",
    "validate_pipeline",
]


@dataclass
class ValidationError:
    """A single validation error."""

    message: str
    location: str = ""
    severity: str = "error"  # "error", "warning"

    def __str__(self) -> str:
        if self.location:
            return f"[{self.severity.upper()}] {self.location}: {self.message}"
        return f"[{self.severity.upper()}] {self.message}"


@dataclass
class ValidationResult:
    """Result of pipeline validation."""

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def add_error(
        self,
        message: str,
        location: str = "",
    ) -> None:
        """Add an error."""
        self.errors.append(ValidationError(message, location, "error"))
        self.valid = False

    def add_warning(
        self,
        message: str,
        location: str = "",
    ) -> None:
        """Add a warning."""
        self.warnings.append(ValidationError(message, location, "warning"))

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": [str(e) for e in self.errors],
            "warnings": [str(w) for w in self.warnings],
        }


# Known actions and their required/optional parameters
KNOWN_ACTIONS: Dict[str, Dict[str, Any]] = {
    "audio.resample": {
        "required": ["input"],
        "optional": ["output", "sample_rate"],
    },
    "audio.normalize": {
        "required": ["input"],
        "optional": ["output", "target_db", "mode"],
    },
    "audio.filter": {
        "required": ["input"],
        "optional": ["output", "filter_type", "low_freq", "high_freq"],
    },
    "audio.trim": {
        "required": ["input"],
        "optional": ["output", "start_time", "end_time"],
    },
    "audio.denoise": {
        "required": ["input"],
        "optional": ["output", "strength"],
    },
    "analysis.indices": {
        "required": ["input"],
        "optional": ["output", "indices", "n_fft", "hop_length"],
    },
    "analysis.cluster": {
        "required": ["embeddings"],
        "optional": ["output", "method", "n_clusters", "min_cluster_size"],
    },
    "inference.predict": {
        "required": ["input", "model"],
        "optional": ["output", "top_k", "min_confidence"],
    },
    "inference.embed": {
        "required": ["input"],
        "optional": ["output", "model", "layer", "format"],
    },
    "detection.ribbit": {
        "required": ["input"],
        "optional": ["output", "preset", "profile"],
    },
    "util.copy": {
        "required": ["source", "destination"],
        "optional": [],
    },
    "util.move": {
        "required": ["source", "destination"],
        "optional": [],
    },
    "util.log": {
        "required": ["message"],
        "optional": ["level"],
    },
}


class PipelineValidator:
    """
    Validator for pipeline definitions.

    Performs comprehensive validation including:
    - Structure validation
    - Step validation
    - Dependency validation
    - Action validation
    - Variable reference validation
    """

    def __init__(self):
        """Initialize validator."""
        self._known_actions = KNOWN_ACTIONS.copy()

    def register_action(
        self,
        name: str,
        required_params: List[str],
        optional_params: Optional[List[str]] = None,
    ) -> None:
        """Register a known action for validation."""
        self._known_actions[name] = {
            "required": required_params,
            "optional": optional_params or [],
        }

    def validate(self, pipeline: Pipeline) -> ValidationResult:
        """
        Validate a pipeline.

        Args:
            pipeline: Pipeline to validate

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(valid=True)

        # Validate pipeline metadata
        self._validate_metadata(pipeline, result)

        # Validate steps
        self._validate_steps(pipeline, result)

        # Validate dependencies
        self._validate_dependencies(pipeline, result)

        # Validate variable references
        self._validate_variable_references(pipeline, result)

        return result

    def _validate_metadata(
        self,
        pipeline: Pipeline,
        result: ValidationResult,
    ) -> None:
        """Validate pipeline metadata."""
        if not pipeline.name:
            result.add_error("Pipeline name is required", "pipeline.name")

        if not pipeline.name.replace("_", "").replace("-", "").isalnum():
            result.add_warning(
                "Pipeline name should contain only alphanumeric characters, "
                "underscores, and hyphens",
                "pipeline.name",
            )

        if not pipeline.steps:
            result.add_error("Pipeline must have at least one step", "steps")

    def _validate_steps(
        self,
        pipeline: Pipeline,
        result: ValidationResult,
    ) -> None:
        """Validate individual steps."""
        step_names: Set[str] = set()

        for idx, step in enumerate(pipeline.steps):
            location = f"steps[{idx}]"

            # Check for duplicate names
            if step.name in step_names:
                result.add_error(f"Duplicate step name: '{step.name}'", location)
            step_names.add(step.name)

            # Validate step name
            if not step.name:
                result.add_error("Step name is required", location)
            elif not step.name.replace("_", "").replace("-", "").isalnum():
                result.add_warning(
                    f"Step name '{step.name}' should contain only "
                    "alphanumeric characters, underscores, and hyphens",
                    f"{location}.name",
                )

            # Validate action
            if not step.action:
                result.add_error("Step action is required", f"{location}.action")
            elif step.action not in self._known_actions:
                result.add_warning(
                    f"Unknown action: '{step.action}'. "
                    "Make sure it's registered with the pipeline engine.",
                    f"{location}.action",
                )
            else:
                # Validate parameters for known actions
                self._validate_action_params(step, location, result)

            # Validate on_error
            if step.on_error not in ("fail", "skip", "continue"):
                result.add_error(
                    f"Invalid on_error value: '{step.on_error}'. "
                    "Must be 'fail', 'skip', or 'continue'",
                    f"{location}.on_error",
                )

            # Validate timeout
            if step.timeout is not None and step.timeout <= 0:
                result.add_error(f"Timeout must be positive: {step.timeout}", f"{location}.timeout")

            # Validate retry
            if step.retry < 0:
                result.add_error(
                    f"Retry count must be non-negative: {step.retry}", f"{location}.retry"
                )

    def _validate_action_params(
        self,
        step: PipelineStep,
        location: str,
        result: ValidationResult,
    ) -> None:
        """Validate parameters for a known action."""
        action_spec = self._known_actions.get(step.action)
        if action_spec is None:
            return

        required = set(action_spec.get("required", []))
        optional = set(action_spec.get("optional", []))
        all_known = required | optional

        provided = set(step.params.keys())

        # Check for missing required params
        missing = required - provided
        for param in missing:
            # Check if it might be a template
            if not any("{{" in str(v) for v in step.params.values()):
                result.add_warning(
                    f"Missing required parameter '{param}' for action '{step.action}'",
                    f"{location}.params",
                )

        # Check for unknown params
        unknown = provided - all_known
        for param in unknown:
            result.add_warning(
                f"Unknown parameter '{param}' for action '{step.action}'",
                f"{location}.params.{param}",
            )

    def _validate_dependencies(
        self,
        pipeline: Pipeline,
        result: ValidationResult,
    ) -> None:
        """Validate step dependencies."""
        step_names = {step.name for step in pipeline.steps}

        for idx, step in enumerate(pipeline.steps):
            location = f"steps[{idx}].depends_on"

            for dep in step.depends_on:
                if dep not in step_names:
                    result.add_error(
                        f"Step '{step.name}' depends on unknown step: '{dep}'", location
                    )
                elif dep == step.name:
                    result.add_error(f"Step '{step.name}' cannot depend on itself", location)

        # Check for circular dependencies
        try:
            pipeline.get_execution_order()
        except ValueError as e:
            result.add_error(str(e), "dependencies")

    def _validate_variable_references(
        self,
        pipeline: Pipeline,
        result: ValidationResult,
    ) -> None:
        """Validate variable references in templates."""
        defined_vars = set(pipeline.variables.keys())
        defined_vars.update(pipeline.env.keys())

        # Pattern to find Jinja2 variable references
        var_pattern = re.compile(r"\{\{\s*(\w+)(?:\.\w+)*\s*\}\}")

        for idx, step in enumerate(pipeline.steps):
            location = f"steps[{idx}]"

            # Check params
            refs = self._find_variable_refs(step.params, var_pattern)
            for ref in refs:
                # Skip 'outputs' which is runtime
                if ref != "outputs" and ref not in defined_vars:
                    result.add_warning(
                        f"Reference to undefined variable '{{{{ {ref} }}}}'", f"{location}.params"
                    )

            # Check condition
            if step.condition:
                refs = var_pattern.findall(step.condition)
                for ref in refs:
                    if ref != "outputs" and ref not in defined_vars:
                        result.add_warning(
                            f"Reference to undefined variable '{{{{ {ref} }}}}' in condition",
                            f"{location}.condition",
                        )

    def _find_variable_refs(
        self,
        value: Any,
        pattern: re.Pattern,
    ) -> Set[str]:
        """Find all variable references in a value."""
        refs: Set[str] = set()

        if isinstance(value, str):
            refs.update(pattern.findall(value))
        elif isinstance(value, dict):
            for v in value.values():
                refs.update(self._find_variable_refs(v, pattern))
        elif isinstance(value, list):
            for v in value:
                refs.update(self._find_variable_refs(v, pattern))

        return refs


def validate_pipeline(
    pipeline: Pipeline,
    strict: bool = False,
) -> ValidationResult:
    """
    Validate a pipeline.

    Args:
        pipeline: Pipeline to validate
        strict: If True, treat warnings as errors

    Returns:
        ValidationResult
    """
    validator = PipelineValidator()
    result = validator.validate(pipeline)

    if strict:
        for warning in result.warnings:
            result.add_error(warning.message, warning.location)

    return result

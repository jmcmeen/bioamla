# core/pipeline/parser.py
"""
Pipeline Parser
===============

TOML-based pipeline definition parser with Jinja2 template support.

Pipelines are defined in TOML files with the following structure:

```toml
[pipeline]
name = "my_pipeline"
description = "Example pipeline"
version = "1.0"

[variables]
input_dir = "./audio"
output_dir = "./results"
sample_rate = 16000

[[steps]]
name = "resample"
action = "audio.resample"
params = { input = "{{ input_dir }}", output = "{{ output_dir }}", sample_rate = "{{ sample_rate }}" }

[[steps]]
name = "analyze"
action = "analysis.indices"
depends_on = ["resample"]
params = { input = "{{ output_dir }}" }
```

This module provides:
- Pipeline: Data class representing a parsed pipeline
- PipelineStep: Data class for individual steps
- parse_pipeline: Parse TOML file to Pipeline object
- render_pipeline: Render Jinja2 templates in pipeline
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from bioamla.core.logger import get_logger

logger = get_logger(__name__)

# Use tomli for Python < 3.11, tomllib for Python >= 3.11
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

__all__ = [
    "Pipeline",
    "PipelineStep",
    "parse_pipeline",
    "parse_pipeline_string",
    "render_pipeline",
    "pipeline_to_toml",
]


@dataclass
class PipelineStep:
    """
    A single step in a pipeline.

    Attributes:
        name: Unique step identifier
        action: Action to perform (e.g., "audio.resample")
        params: Parameters for the action
        depends_on: Names of steps this step depends on
        condition: Optional condition for execution (Jinja2 expression)
        enabled: Whether the step is enabled
        description: Human-readable description
        on_error: Error handling strategy ("fail", "skip", "continue")
        timeout: Maximum execution time in seconds
        retry: Number of retry attempts
    """

    name: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    enabled: bool = True
    description: str = ""
    on_error: str = "fail"
    timeout: Optional[float] = None
    retry: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "name": self.name,
            "action": self.action,
            "params": self.params,
        }
        if self.depends_on:
            d["depends_on"] = self.depends_on
        if self.condition:
            d["condition"] = self.condition
        if not self.enabled:
            d["enabled"] = False
        if self.description:
            d["description"] = self.description
        if self.on_error != "fail":
            d["on_error"] = self.on_error
        if self.timeout:
            d["timeout"] = self.timeout
        if self.retry > 0:
            d["retry"] = self.retry
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStep":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            action=data["action"],
            params=data.get("params", {}),
            depends_on=data.get("depends_on", []),
            condition=data.get("condition"),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            on_error=data.get("on_error", "fail"),
            timeout=data.get("timeout"),
            retry=data.get("retry", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Pipeline:
    """
    A complete pipeline definition.

    Attributes:
        name: Pipeline name
        description: Pipeline description
        version: Pipeline version
        steps: List of pipeline steps
        variables: Default variable values
        env: Environment variable mappings
        imports: List of other pipelines to import
        metadata: Additional metadata
    """

    name: str
    steps: List[PipelineStep]
    description: str = ""
    version: str = "1.0"
    variables: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    @property
    def step_names(self) -> Set[str]:
        """Get set of all step names."""
        return {step.name for step in self.steps}

    @property
    def enabled_steps(self) -> List[PipelineStep]:
        """Get list of enabled steps."""
        return [step for step in self.steps if step.enabled]

    def get_step(self, name: str) -> Optional[PipelineStep]:
        """Get step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def get_execution_order(self) -> List[str]:
        """
        Get topologically sorted execution order.

        Returns:
            List of step names in execution order
        """
        # Build dependency graph
        graph = {step.name: set(step.depends_on) for step in self.enabled_steps}
        available = {step.name for step in self.enabled_steps}

        # Validate dependencies
        for name, deps in graph.items():
            missing = deps - available
            if missing:
                raise ValueError(f"Step '{name}' depends on unknown steps: {missing}")

        # Topological sort (Kahn's algorithm)
        in_degree = {name: len(deps) for name, deps in graph.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            current = queue.pop(0)
            result.append(current)

            for name, deps in graph.items():
                if current in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        if len(result) != len(graph):
            raise ValueError("Circular dependency detected in pipeline")

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pipeline": {
                "name": self.name,
                "description": self.description,
                "version": self.version,
            },
            "variables": self.variables,
            "env": self.env if self.env else None,
            "imports": self.imports if self.imports else None,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata if self.metadata else None,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        source_path: Optional[str] = None,
    ) -> "Pipeline":
        """Create from dictionary."""
        pipeline_info = data.get("pipeline", {})
        steps_data = data.get("steps", [])

        return cls(
            name=pipeline_info.get("name", "unnamed"),
            description=pipeline_info.get("description", ""),
            version=pipeline_info.get("version", "1.0"),
            steps=[PipelineStep.from_dict(s) for s in steps_data],
            variables=data.get("variables", {}),
            env=data.get("env", {}),
            imports=data.get("imports", []),
            metadata=data.get("metadata", {}),
            source_path=source_path,
        )


def parse_pipeline(
    filepath: Union[str, Path],
    variables: Optional[Dict[str, Any]] = None,
    render_templates: bool = True,
) -> Pipeline:
    """
    Parse a pipeline from a TOML file.

    Args:
        filepath: Path to the TOML pipeline file
        variables: Override variables
        render_templates: Whether to render Jinja2 templates

    Returns:
        Parsed Pipeline object
    """
    if tomllib is None:
        raise ImportError(
            "TOML support requires tomli for Python < 3.11. Install with: pip install tomli"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Pipeline file not found: {filepath}")

    with open(filepath, "rb") as f:
        data = tomllib.load(f)

    pipeline = Pipeline.from_dict(data, source_path=str(filepath))

    # Merge override variables
    if variables:
        pipeline.variables.update(variables)

    # Render templates if requested
    if render_templates:
        pipeline = render_pipeline(pipeline)

    return pipeline


def parse_pipeline_string(
    content: str,
    variables: Optional[Dict[str, Any]] = None,
    render_templates: bool = True,
) -> Pipeline:
    """
    Parse a pipeline from a TOML string.

    Args:
        content: TOML content string
        variables: Override variables
        render_templates: Whether to render Jinja2 templates

    Returns:
        Parsed Pipeline object
    """
    if tomllib is None:
        raise ImportError(
            "TOML support requires tomli for Python < 3.11. Install with: pip install tomli"
        )

    data = tomllib.loads(content)
    pipeline = Pipeline.from_dict(data)

    if variables:
        pipeline.variables.update(variables)

    if render_templates:
        pipeline = render_pipeline(pipeline)

    return pipeline


def render_pipeline(
    pipeline: Pipeline,
    extra_variables: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """
    Render Jinja2 templates in pipeline parameters.

    Args:
        pipeline: Pipeline to render
        extra_variables: Additional variables for rendering

    Returns:
        New Pipeline with rendered templates
    """
    try:
        from jinja2 import BaseLoader, Environment, UndefinedError
    except ImportError:
        logger.warning(
            "Jinja2 not installed, skipping template rendering. Install with: pip install jinja2"
        )
        return pipeline

    # Combine variables
    context = {**pipeline.variables}
    if extra_variables:
        context.update(extra_variables)

    # Add environment variables
    import os

    for key, env_var in pipeline.env.items():
        context[key] = os.environ.get(env_var, "")

    # Create Jinja2 environment
    env = Environment(loader=BaseLoader())

    def render_value(value: Any) -> Any:
        """Recursively render Jinja2 templates in a value."""
        if isinstance(value, str) and "{{" in value:
            try:
                template = env.from_string(value)
                return template.render(context)
            except UndefinedError as e:
                logger.warning(f"Template rendering error: {e}")
                return value
        elif isinstance(value, dict):
            return {k: render_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [render_value(v) for v in value]
        return value

    # Render each step
    rendered_steps = []
    for step in pipeline.steps:
        rendered_params = render_value(step.params)
        rendered_condition = None
        if step.condition and "{{" in step.condition:
            try:
                template = env.from_string(step.condition)
                rendered_condition = template.render(context)
            except UndefinedError:
                rendered_condition = step.condition
        else:
            rendered_condition = step.condition

        rendered_steps.append(
            PipelineStep(
                name=step.name,
                action=step.action,
                params=rendered_params,
                depends_on=step.depends_on,
                condition=rendered_condition,
                enabled=step.enabled,
                description=step.description,
                on_error=step.on_error,
                timeout=step.timeout,
                retry=step.retry,
                metadata=step.metadata,
            )
        )

    return Pipeline(
        name=pipeline.name,
        description=pipeline.description,
        version=pipeline.version,
        steps=rendered_steps,
        variables=pipeline.variables,
        env=pipeline.env,
        imports=pipeline.imports,
        metadata=pipeline.metadata,
        source_path=pipeline.source_path,
    )


def pipeline_to_toml(pipeline: Pipeline) -> str:
    """
    Convert a Pipeline to TOML string.

    Args:
        pipeline: Pipeline to convert

    Returns:
        TOML string representation
    """
    lines = []

    # Pipeline section
    lines.append("[pipeline]")
    lines.append(f'name = "{pipeline.name}"')
    if pipeline.description:
        lines.append(f'description = "{pipeline.description}"')
    lines.append(f'version = "{pipeline.version}"')
    lines.append("")

    # Variables section
    if pipeline.variables:
        lines.append("[variables]")
        for key, value in pipeline.variables.items():
            lines.append(_format_toml_value(key, value))
        lines.append("")

    # Environment section
    if pipeline.env:
        lines.append("[env]")
        for key, value in pipeline.env.items():
            lines.append(f'{key} = "{value}"')
        lines.append("")

    # Imports section
    if pipeline.imports:
        lines.append("imports = [")
        for imp in pipeline.imports:
            lines.append(f'    "{imp}",')
        lines.append("]")
        lines.append("")

    # Steps section
    for step in pipeline.steps:
        lines.append("[[steps]]")
        lines.append(f'name = "{step.name}"')
        lines.append(f'action = "{step.action}"')

        if step.description:
            lines.append(f'description = "{step.description}"')

        if step.depends_on:
            deps = ", ".join(f'"{d}"' for d in step.depends_on)
            lines.append(f"depends_on = [{deps}]")

        if step.condition:
            lines.append(f'condition = "{step.condition}"')

        if not step.enabled:
            lines.append("enabled = false")

        if step.on_error != "fail":
            lines.append(f'on_error = "{step.on_error}"')

        if step.timeout:
            lines.append(f"timeout = {step.timeout}")

        if step.retry > 0:
            lines.append(f"retry = {step.retry}")

        if step.params:
            lines.append("")
            lines.append("[steps.params]")
            for key, value in step.params.items():
                lines.append(_format_toml_value(key, value))

        lines.append("")

    return "\n".join(lines)


def _format_toml_value(key: str, value: Any) -> str:
    """Format a key-value pair for TOML."""
    if isinstance(value, str):
        return f'{key} = "{value}"'
    elif isinstance(value, bool):
        return f"{key} = {str(value).lower()}"
    elif isinstance(value, (int, float)):
        return f"{key} = {value}"
    elif isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            items = ", ".join(f'"{v}"' for v in value)
            return f"{key} = [{items}]"
        else:
            items = ", ".join(str(v) for v in value)
            return f"{key} = [{items}]"
    elif isinstance(value, dict):
        items = ", ".join(
            f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}" for k, v in value.items()
        )
        return f"{key} = {{ {items} }}"
    else:
        return f'{key} = "{value}"'

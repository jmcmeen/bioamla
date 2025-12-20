"""
Pipeline Package
================

TOML-based pipeline definitions and execution engine.

This package provides:
- PipelineStep: Base class for defining pipeline steps
- Step utilities: TransformStep, BranchStep, PassthroughStep
- Step status tracking and result handling
- Pipeline: TOML pipeline definition parser
- PipelineEngine: Pipeline execution engine
- PipelineValidator: Schema validation
"""

from .engine import (
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    StepExecutionResult,
    PipelineEngine,
)
from .parser import (
    Pipeline,
    PipelineStep,
    parse_pipeline,
    parse_pipeline_string,
    render_pipeline,
    pipeline_to_toml,
)
from .step import (
    BranchStep,
    PassthroughStep,
    PipelineStep,
    StepInfo,
    StepResult,
    StepStatus,
    TransformStep,
)
from .validator import (
    ValidationError,
    ValidationResult,
    PipelineValidator,
    validate_pipeline,
)

__all__ = [
    # Core step classes
    "PipelineStep",
    "StepResult",
    "StepInfo",
    "StepStatus",
    # Utility steps
    "PassthroughStep",
    "TransformStep",
    "BranchStep",
    # TOML pipeline parser
    "Pipeline",
    "PipelineStep",
    "parse_pipeline",
    "parse_pipeline_string",
    "render_pipeline",
    "pipeline_to_toml",
    # Pipeline engine
    "PipelineEngine",
    "ExecutionResult",
    "StepExecutionResult",
    "ExecutionStatus",
    "ExecutionContext",
    # Validation
    "PipelineValidator",
    "ValidationResult",
    "ValidationError",
    "validate_pipeline",
]

"""
Workflow Package
================

TOML-based workflow definitions and execution engine.

This package provides:
- PipelineStep: Base class for defining pipeline steps
- Step utilities: TransformStep, BranchStep, PassthroughStep
- Step status tracking and result handling
- Workflow: TOML workflow definition parser
- WorkflowEngine: Workflow execution engine
- WorkflowValidator: Schema validation
"""

from .step import (
    BranchStep,
    PassthroughStep,
    PipelineStep,
    StepInfo,
    StepResult,
    StepStatus,
    TransformStep,
)

from .parser import (
    Workflow,
    WorkflowStep,
    parse_workflow,
    parse_workflow_string,
    render_workflow,
    workflow_to_toml,
)

from .engine import (
    WorkflowEngine,
    ExecutionResult,
    StepExecutionResult,
    ExecutionStatus,
    ExecutionContext,
)

from .validator import (
    WorkflowValidator,
    ValidationResult,
    ValidationError,
    validate_workflow,
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
    # TOML workflow parser
    "Workflow",
    "WorkflowStep",
    "parse_workflow",
    "parse_workflow_string",
    "render_workflow",
    "workflow_to_toml",
    # Workflow engine
    "WorkflowEngine",
    "ExecutionResult",
    "StepExecutionResult",
    "ExecutionStatus",
    "ExecutionContext",
    # Validation
    "WorkflowValidator",
    "ValidationResult",
    "ValidationError",
    "validate_workflow",
]

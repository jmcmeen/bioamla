"""
Workflow Package
================

TOML-based workflow definitions and execution engine.

This package provides:
- PipelineStep: Base class for defining pipeline steps
- Step utilities: TransformStep, BranchStep, PassthroughStep
- Step status tracking and result handling
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

__all__ = [
    # Core
    "PipelineStep",
    "StepResult",
    "StepInfo",
    "StepStatus",
    # Utility steps
    "PassthroughStep",
    "TransformStep",
    "BranchStep",
]

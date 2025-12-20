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
    PipelineStep,
    StepResult,
    StepInfo,
    StepStatus,
    PassthroughStep,
    TransformStep,
    BranchStep,
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

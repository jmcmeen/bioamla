# controllers/__init__.py
"""
Controllers Package
===================

Application controllers that orchestrate between views (CLI, API) and core business logic.

Controllers provide:
- A clean interface for views to invoke operations
- Input validation and error handling
- Progress reporting and logging
- File I/O and output formatting
- Coordination between multiple services

Architecture:
    View (CLI/API)
        ↓ (config objects, paths)
    Controller
        ↓ (delegates to)
    Core Services (audio, ml, analysis)
        ↓ (uses)
    Database (repositories, UoW)

Audio Controller Architecture:
    AudioFileController - File I/O with undo/redo support
        ↓ produces/consumes
    AudioData - In-memory audio container
        ↓ processed by
    AudioTransformController - In-memory signal processing
        ↓ (no file writes)
    AudioFileController.save() - Only way to persist changes

Usage:
    from bioamla.controllers import AudioFileController, AudioTransformController

    # Load audio through file controller
    file_ctrl = AudioFileController()
    result = file_ctrl.open("input.wav")
    audio = result.data

    # Apply transforms in memory
    transform_ctrl = AudioTransformController()
    audio = transform_ctrl.apply_bandpass(audio, 500, 8000).data
    audio = transform_ctrl.normalize_loudness(audio, -20).data

    # Save through file controller (with undo support)
    file_ctrl.save(audio, "output.wav")

    # Undo the save
    file_ctrl.undo()
"""

from .annotation_controller import AnnotationController, AnnotationResult, ClipExtractionResult
from .audio import AudioController  # Legacy controller for CLI compatibility
from .audio_file import AudioData, AudioFileController
from .audio_transform import AudioTransformController
from .base import BaseController, ControllerResult
from .clustering import ClusteringController
from .embedding import EmbeddingController
from .inaturalist import (
    INaturalistController,
    SearchResult as INatSearchResult,
    DownloadResult as INatDownloadResult,
    TaxonInfo,
    ProjectStats as INatProjectStats,
    ObservationInfo,
)
from .indices import BatchIndicesResult, IndicesController, IndicesResult
from .inference import InferenceController
from .pipeline import PipelineController, PipelineProgress, PipelineResult
from .project import (
    ProjectController,
    ProjectSummary,
    ProjectStatistics,
    ConfigSummary,
    ModelInfo,
    DatasetInfo,
    RunInfo,
)
from .ribbit import RibbitController, DetectionSummary, BatchDetectionSummary
from .workflow import WorkflowController, WorkflowSummary, ExecutionSummary, ValidationSummary

__all__ = [
    # Base
    "BaseController",
    "ControllerResult",
    # Audio
    "AudioController",  # Legacy, for CLI file-based operations
    "AudioFileController",  # New: File I/O with undo/redo
    "AudioTransformController",  # New: In-memory transforms
    "AudioData",  # Audio data container
    # Annotations
    "AnnotationController",
    "AnnotationResult",
    "ClipExtractionResult",
    # ML
    "InferenceController",
    "EmbeddingController",
    "ClusteringController",
    # Detection
    "RibbitController",
    "DetectionSummary",
    "BatchDetectionSummary",
    # iNaturalist
    "INaturalistController",
    "INatSearchResult",
    "INatDownloadResult",
    "TaxonInfo",
    "INatProjectStats",
    "ObservationInfo",
    # Pipeline
    "PipelineController",
    "PipelineResult",
    "PipelineProgress",
    # Project
    "ProjectController",
    "ProjectSummary",
    "ProjectStatistics",
    "ConfigSummary",
    "ModelInfo",
    "DatasetInfo",
    "RunInfo",
    # Workflow
    "WorkflowController",
    "WorkflowSummary",
    "ExecutionSummary",
    "ValidationSummary",
    # Analysis
    "IndicesController",
    "IndicesResult",
    "BatchIndicesResult",
]

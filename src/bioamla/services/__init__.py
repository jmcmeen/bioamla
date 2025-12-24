# services/__init__.py
"""
Services Package
================

Application services that orchestrate between views (CLI, API) and core business logic.

Services provide:
- A clean interface for views to invoke operations
- Input validation and error handling
- Progress reporting and logging
- File I/O and output formatting
- Coordination between multiple services

Architecture:
    View (CLI/API)
        ↓ (config objects, paths)
    Service
        ↓ (delegates to)
    Core Services (audio, ml, analysis)
        ↓ (uses)
    Database (repositories, UoW)

Audio Service Architecture:
    AudioFileController - File I/O operations
        ↓ produces/consumes
    AudioData - In-memory audio container
        ↓ processed by
    AudioTransformController - In-memory signal processing
        ↓ (no file writes)
    AudioFileController.save() - Only way to persist changes

Usage:
    from bioamla.services import AudioFileController, AudioTransformController

    # Load audio through file controller
    file_ctrl = AudioFileController()
    result = file_ctrl.open("input.wav")
    audio = result.data

    # Apply transforms in memory
    transform_ctrl = AudioTransformController()
    audio = transform_ctrl.apply_bandpass(audio, 500, 8000).data
    audio = transform_ctrl.normalize_loudness(audio, -20).data

    # Save through file controller
    file_ctrl.save(audio, "output.wav")
"""

from .annotation import AnnotationService, AnnotationResult, ClipExtractionResult
from .audio import AudioController  # Legacy controller for CLI compatibility
from .audio_file import AudioData, AudioFileController
from .audio_transform import AudioTransformController
from .base import BaseService, ControllerResult
from .clustering import ClusteringController
from .embedding import EmbeddingController
from .inaturalist import (
    DownloadResult as INatDownloadResult,
)
from .inaturalist import (
    INaturalistController,
    ObservationInfo,
    TaxonInfo,
)
from .inaturalist import (
    ProjectStats as INatProjectStats,
)
from .inaturalist import (
    SearchResult as INatSearchResult,
)
from .indices import BatchIndicesResult, IndicesController, IndicesResult
from .inference import InferenceController
from .pipeline import (
    ExecutionSummary,
    PipelineController,
    PipelineSummary,
    ValidationSummary,
)
from .ribbit import BatchDetectionSummary, DetectionSummary, RibbitService

__all__ = [
    # Base
    "BaseService",
    "ControllerResult",
    # Audio
    "AudioController",  # Legacy, for CLI file-based operations
    "AudioFileController",  # File I/O operations
    "AudioTransformController",  # In-memory transforms
    "AudioData",  # Audio data container
    # Annotations
    "AnnotationService",
    "AnnotationResult",
    "ClipExtractionResult",
    # ML
    "InferenceController",
    "EmbeddingController",
    "ClusteringController",
    # Detection
    "RibbitService",
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
    "PipelineSummary",
    "ExecutionSummary",
    "ValidationSummary",
    # Analysis
    "IndicesController",
    "IndicesResult",
    "BatchIndicesResult",
]

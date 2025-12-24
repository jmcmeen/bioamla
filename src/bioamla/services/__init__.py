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
    Core (audio processing, ML, analysis)

Audio Service Architecture:
    AudioFileService - File I/O operations
        ↓ produces/consumes
    AudioData - In-memory audio container
        ↓ processed by
    AudioTransformService - In-memory signal processing + file-based batch operations
        ↓ (can persist via file methods or delegate to AudioFileService)

Usage:
    from bioamla.services import AudioFileService, AudioTransformService

    # Load audio through file service
    file_svc = AudioFileService()
    result = file_svc.open("input.wav")
    audio = result.data

    # Apply transforms in memory
    transform_svc = AudioTransformService()
    audio = transform_svc.apply_bandpass(audio, 500, 8000).data
    audio = transform_svc.normalize_loudness(audio, -20).data

    # Save through file service
    file_svc.save(audio, "output.wav")

    # Or use file-based batch operations
    result = transform_svc.resample_batch("input_dir", "output_dir", 16000)
"""

from .annotation import AnnotationService, AnnotationResult, ClipExtractionResult
from .audio_file import AudioData, AudioFileService
from .audio_transform import (
    AudioTransformService,
    AudioMetadata,
    ProcessedAudio,
    AnalysisResult,
    BatchResult,
)
from .base import BaseService, ServiceResult
from .clustering import ClusteringService
from .embedding import EmbeddingService
from .inaturalist import (
    DownloadResult as INatDownloadResult,
)
from .inaturalist import (
    INaturalistService,
    ObservationInfo,
    TaxonInfo,
)
from .inaturalist import (
    ProjectStats as INatProjectStats,
)
from .inaturalist import (
    SearchResult as INatSearchResult,
)
from .indices import BatchIndicesResult, IndicesService, IndicesResult
from .inference import InferenceService
from .ribbit import BatchDetectionSummary, DetectionSummary, RibbitService

__all__ = [
    # Base
    "BaseService",
    "ServiceResult",
    # Audio
    "AudioFileService",  # File I/O operations
    "AudioTransformService",  # In-memory transforms + file-based batch operations
    "AudioData",  # Audio data container
    "AudioMetadata",  # File metadata
    "ProcessedAudio",  # Processing result
    "AnalysisResult",  # Analysis result
    "BatchResult",  # Batch operation result
    # Annotations
    "AnnotationService",
    "AnnotationResult",
    "ClipExtractionResult",
    # ML
    "InferenceService",
    "EmbeddingService",
    "ClusteringService",
    # Detection
    "RibbitService",
    "DetectionSummary",
    "BatchDetectionSummary",
    # iNaturalist
    "INaturalistService",
    "INatSearchResult",
    "INatDownloadResult",
    "TaxonInfo",
    "INatProjectStats",
    "ObservationInfo",
    # Analysis
    "IndicesService",
    "IndicesResult",
    "BatchIndicesResult",
]

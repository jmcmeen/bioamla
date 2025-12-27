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

from bioamla.models.detection import BatchDetectionResult, DetectionInfo, DetectionResult
from bioamla.models.ribbit import BatchDetectionSummary, DetectionSummary

from .annotation import AnnotationResult, AnnotationService, ClipExtractionResult
from .ast import ASTService
from .audio_file import AudioData, AudioFileService
from .audio_transform import (
    AnalysisResult,
    AudioMetadata,
    AudioTransformService,
    BatchResult,
    ProcessedAudio,
)
from .base import BaseService, ServiceResult
from .batch_audio_info import BatchAudioInfoService
from .batch_audio_transform import BatchAudioTransformService
from .batch_base import BatchServiceBase
from .batch_clustering import BatchClusteringService
from .batch_detection import BatchDetectionService
from .batch_indices import BatchIndicesService
from .batch_inference import BatchInferenceService
from .birdnet import BirdNETService
from .clustering import ClusteringService
from .cnn import CNNService
from .config import ConfigService
from .dataset import AugmentResult, DatasetService, LicenseResult, MergeResult
from .dependency import DependencyInfo, DependencyReport, DependencyService
from .detection import DetectionService
from .ebird import EBirdService
from .embedding import EmbeddingService
from .factory import ServiceFactory

# New services for CLI layer separation
from .file import FileService
from .huggingface import HuggingFaceService
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
from .indices import BatchIndicesResult, IndicesResult, IndicesService
from .inference import InferenceService
from .macaulay import MacaulayService
from .ribbit import RibbitService
from .species import SpeciesService

# Catalog services
from .xeno_canto import XenoCantoService

__all__ = [
    # Base
    "BaseService",
    "ServiceResult",
    # Batch Base
    "BatchServiceBase",
    # Factory
    "ServiceFactory",
    # Audio
    "AudioFileService",  # File I/O operations
    "AudioTransformService",  # In-memory transforms + file-based batch operations
    "AudioData",  # Audio data container
    "AudioMetadata",  # File metadata
    "ProcessedAudio",  # Processing result
    "AnalysisResult",  # Analysis result
    "BatchResult",  # Batch operation result
    # Batch Services
    "BatchAudioTransformService",
    "BatchDetectionService",
    "BatchIndicesService",
    "BatchInferenceService",
    "BatchClusteringService",
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
    # Catalogs
    "XenoCantoService",
    "MacaulayService",
    "SpeciesService",
    "EBirdService",
    "HuggingFaceService",
    # File I/O
    "FileService",
    # Configuration
    "ConfigService",
    # System Dependencies
    "DependencyService",
    "DependencyInfo",
    "DependencyReport",
    # Detection
    "DetectionService",
    "DetectionInfo",
    "DetectionResult",
    "BatchDetectionResult",
    # Dataset
    "DatasetService",
    "MergeResult",
    "AugmentResult",
    "LicenseResult",
    # Model Services
    "ASTService",
    "CNNService",
    "BirdNETService",
]

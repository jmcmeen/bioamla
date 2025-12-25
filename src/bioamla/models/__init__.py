"""Data models for BioAMLA."""

# File types
from bioamla.models.file_types import FileMetadata, FileType

# Audio models
from bioamla.models.audio import (
    AnalysisResult,
    AudioData,
    AudioMetadata,
    BatchResult,
    ProcessedAudio,
    TransformResult,
)

# Detection models
from bioamla.models.detection import (
    BatchDetectionResult,
    DetectionInfo,
    DetectionResult,
)

# Annotation models
from bioamla.models.annotation import (
    AnnotationResult,
    ClipExtractionResult,
    MeasurementResult,
)

# Embedding models
from bioamla.models.embedding import (
    BatchEmbeddingSummary,
    EmbeddingInfo,
)

# Clustering models
from bioamla.models.clustering import (
    ClusterAnalysis,
    ClusteringSummary,
    NoveltyDetectionSummary,
)

# Inference models
from bioamla.models.inference import (
    BatchInferenceResult,
    InferenceSummary,
    PredictionResult,
)

# iNaturalist models
from bioamla.models.inaturalist import (
    DownloadResult,
    ObservationInfo,
    ProjectStats,
    SearchResult,
    TaxonInfo,
)

# Indices models
from bioamla.models.indices import (
    BatchIndicesResult,
    IndicesResult,
    TemporalIndicesResult,
)

# RIBBIT models
from bioamla.models.ribbit import (
    BatchDetectionSummary,
    DetectionSummary,
)

# Dataset models
from bioamla.models.dataset import (
    AugmentResult,
    BatchLicenseResult,
    LicenseResult,
    MergeResult,
)

# Dependency models
from bioamla.models.dependency import (
    DependencyInfo,
    DependencyReport,
)

# Batch models
from bioamla.models.batch import (
    BatchConfig,
)

__all__ = [
    "FileType",
    "FileMetadata",
    "AudioData",
    "AudioMetadata",
    "ProcessedAudio",
    "AnalysisResult",
    "BatchResult",
    "TransformResult",
    "DetectionInfo",
    "DetectionResult",
    "BatchDetectionResult",
    "AnnotationResult",
    "ClipExtractionResult",
    "MeasurementResult",
    "EmbeddingInfo",
    "BatchEmbeddingSummary",
    "ClusteringSummary",
    "NoveltyDetectionSummary",
    "ClusterAnalysis",
    "PredictionResult",
    "InferenceSummary",
    "BatchInferenceResult",
    "SearchResult",
    "DownloadResult",
    "TaxonInfo",
    "ObservationInfo",
    "ProjectStats",
    "IndicesResult",
    "TemporalIndicesResult",
    "BatchIndicesResult",
    "DetectionSummary",
    "BatchDetectionSummary",
    "MergeResult",
    "AugmentResult",
    "LicenseResult",
    "BatchLicenseResult",
    "DependencyInfo",
    "DependencyReport",
    "BatchConfig",
]

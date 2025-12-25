"""Data models for BioAMLA."""

# File types
# Annotation models
from bioamla.models.annotation import (
    AnnotationResult,
    ClipExtractionResult,
    MeasurementResult,
)

# Audio models
from bioamla.models.audio import (
    AnalysisResult,
    AudioData,
    AudioMetadata,
    BatchResult,
    ProcessedAudio,
    TransformResult,
)

# Batch models
from bioamla.models.batch import (
    BatchConfig,
)

# Clustering models
from bioamla.models.clustering import (
    ClusterAnalysis,
    ClusteringSummary,
    NoveltyDetectionSummary,
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

# Detection models
from bioamla.models.detection import (
    BatchDetectionResult,
    DetectionInfo,
    DetectionResult,
)

# Embedding models
from bioamla.models.embedding import (
    BatchEmbeddingSummary,
    EmbeddingInfo,
)
from bioamla.models.file_types import FileMetadata, FileType

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

# Inference models
from bioamla.models.inference import (
    BatchInferenceResult,
    InferenceSummary,
    PredictionResult,
)

# RIBBIT models
from bioamla.models.ribbit import (
    BatchDetectionSummary,
    DetectionSummary,
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

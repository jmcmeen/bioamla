"""Data models for BioAMLA."""

# File types
# Annotation models
from bioamla.models.annotation import (
    AnnotationResult,
    ClipExtractionResult,
    MeasurementResult,
)

# AST models
from bioamla.models.ast import (
    EvaluationResult as ASTEvaluationResult,
)
from bioamla.models.ast import (
    PredictionResult as ASTPredictionResult,
)
from bioamla.models.ast import (
    TrainResult as ASTTrainResult,
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

# BirdNET models
from bioamla.models.birdnet import (
    PredictionResult as BirdNETPredictionResult,
)

# Clustering models
from bioamla.models.clustering import (
    ClusterAnalysis,
    ClusteringSummary,
    NoveltyDetectionSummary,
)

# CNN models
from bioamla.models.cnn import (
    EvaluationResult as CNNEvaluationResult,
)
from bioamla.models.cnn import (
    PredictionResult as CNNPredictionResult,
)
from bioamla.models.cnn import (
    TrainResult as CNNTrainResult,
)

# Config models
from bioamla.models.config import (
    ConfigInfo,
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

# eBird models
from bioamla.models.ebird import (
    EBirdObservation,
)
from bioamla.models.ebird import (
    NearbyResult as EBirdNearbyResult,
)
from bioamla.models.ebird import (
    ValidationResult as EBirdValidationResult,
)

# Embedding models
from bioamla.models.embedding import (
    BatchEmbeddingSummary,
    EmbeddingInfo,
)
from bioamla.models.file_types import FileMetadata, FileType

# HuggingFace models
from bioamla.models.huggingface import (
    PushResult,
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

# Inference models
from bioamla.models.inference import (
    BatchInferenceResult,
    InferenceSummary,
    PredictionResult,
)

# Macaulay models
from bioamla.models.macaulay import (
    DownloadResult as MacaulayDownloadResult,
)
from bioamla.models.macaulay import (
    MLRecording,
)
from bioamla.models.macaulay import (
    SearchResult as MacaulaySearchResult,
)

# RIBBIT models
from bioamla.models.ribbit import (
    BatchDetectionSummary,
    DetectionSummary,
)

# Species models
from bioamla.models.species import (
    SearchMatch,
    SpeciesInfo,
)

# Utility models
from bioamla.models.util import (
    DeviceInfo,
    DevicesData,
    VersionData,
)

# Xeno-canto models
from bioamla.models.xeno_canto import (
    DownloadResult as XenoCantoDownloadResult,
)
from bioamla.models.xeno_canto import (
    SearchResult as XenoCantoSearchResult,
)
from bioamla.models.xeno_canto import (
    XCRecording,
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
    # AST models
    "ASTPredictionResult",
    "ASTTrainResult",
    "ASTEvaluationResult",
    # BirdNET models
    "BirdNETPredictionResult",
    # CNN models
    "CNNPredictionResult",
    "CNNTrainResult",
    "CNNEvaluationResult",
    # Config models
    "ConfigInfo",
    # eBird models
    "EBirdObservation",
    "EBirdValidationResult",
    "EBirdNearbyResult",
    # HuggingFace models
    "PushResult",
    # Macaulay models
    "MLRecording",
    "MacaulaySearchResult",
    "MacaulayDownloadResult",
    # Species models
    "SpeciesInfo",
    "SearchMatch",
    # Utility models
    "DeviceInfo",
    "DevicesData",
    "VersionData",
    # Xeno-canto models
    "XCRecording",
    "XenoCantoSearchResult",
    "XenoCantoDownloadResult",
]

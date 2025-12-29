"""
Service Factory
===============

Reusable factory for instantiating services with proper dependency injection.

This factory lives in the services layer to be reusable across all applications
(CLI, web API, GUI, tests, etc.). Applications can use it as-is or override
behavior through dependency injection.

Architecture Principle:
- Factory provides sensible defaults (LocalFileRepository for file-based services)
- Applications can override by passing custom repository to factory constructor
- Services remain decoupled from concrete implementations

Usage:
    # CLI usage - use default LocalFileRepository
    from bioamla.services.factory import ServiceFactory

    factory = ServiceFactory()
    audio_svc = factory.create_audio_file_service()
    detection_svc = factory.create_detection_service()

    # Custom application - inject custom repository
    from my_app.cloud_repo import S3FileRepository

    factory = ServiceFactory(file_repository=S3FileRepository())
    audio_svc = factory.create_audio_file_service()  # Uses S3 instead of local
"""

from typing import Optional

from bioamla.repository import LocalFileRepository
from bioamla.repository.protocol import FileRepositoryProtocol

# Import all services
from .annotation import AnnotationService
from .ast import ASTService
from .audio_file import AudioFileService
from .audio_transform import AudioTransformService
from .batch_audio_info import BatchAudioInfoService
from .batch_audio_transform import BatchAudioTransformService
from .batch_clustering import BatchClusteringService
from .batch_detection import BatchDetectionService
from .batch_indices import BatchIndicesService
from .batch_inference import BatchInferenceService
from .birdnet import BirdNETService
from .clustering import ClusteringService
from .cnn import CNNService
from .config import ConfigService
from .dataset import DatasetService
from .dependency import DependencyService
from .detection import DetectionService
from .ebird import EBirdService
from .embedding import EmbeddingService
from .file import FileService
from .huggingface import HuggingFaceService
from .inaturalist import INaturalistService
from .indices import IndicesService
from .inference import InferenceService
from .macaulay import MacaulayService
from .ribbit import RibbitService
from .species import SpeciesService
from .util import UtilityService
from .xeno_canto import XenoCantoService


class ServiceFactory:
    """
    Factory for creating service instances with proper dependency injection.

    This factory provides a centralized way to instantiate services with their
    required dependencies. By default, it uses LocalFileRepository for file-based
    services, but applications can inject custom repositories.

    Attributes:
        file_repository: File repository implementation for file-based services
    """

    def __init__(self, file_repository: Optional[FileRepositoryProtocol] = None):
        """
        Initialize the service factory.

        Args:
            file_repository: Optional custom file repository. If None, uses LocalFileRepository.
        """
        self.file_repository = file_repository or LocalFileRepository()

    # ========================================================================
    # Single-File Services (File-Based)
    # ========================================================================

    def create_audio_file_service(self) -> AudioFileService:
        """Create AudioFileService with file repository."""
        return AudioFileService(file_repository=self.file_repository)

    def create_audio_transform_service(self) -> AudioTransformService:
        """Create AudioTransformService with file repository."""
        return AudioTransformService(file_repository=self.file_repository)

    def create_file_service(self) -> FileService:
        """Create FileService with file repository."""
        return FileService(file_repository=self.file_repository)

    def create_dataset_service(self) -> DatasetService:
        """Create DatasetService with file repository."""
        return DatasetService(file_repository=self.file_repository)

    def create_annotation_service(self) -> AnnotationService:
        """Create AnnotationService with file repository."""
        return AnnotationService(file_repository=self.file_repository)

    def create_detection_service(self) -> DetectionService:
        """Create DetectionService with file repository."""
        return DetectionService(file_repository=self.file_repository)

    def create_indices_service(self) -> IndicesService:
        """Create IndicesService with file repository."""
        return IndicesService(file_repository=self.file_repository)

    def create_ribbit_service(self) -> RibbitService:
        """Create RibbitService with file repository."""
        return RibbitService(file_repository=self.file_repository)

    def create_embedding_service(self) -> EmbeddingService:
        """Create EmbeddingService with file repository."""
        return EmbeddingService(file_repository=self.file_repository)

    def create_inference_service(self) -> InferenceService:
        """Create InferenceService with file repository."""
        return InferenceService(file_repository=self.file_repository)

    def create_ast_service(self) -> ASTService:
        """Create ASTService with file repository (AST-only for now)."""
        return ASTService(file_repository=self.file_repository)

    def create_birdnet_service(self) -> BirdNETService:
        """Create BirdNETService with file repository (deferred to future)."""
        return BirdNETService(file_repository=self.file_repository)

    def create_cnn_service(self) -> CNNService:
        """Create CNNService with file repository (deferred to future)."""
        return CNNService(file_repository=self.file_repository)

    # ========================================================================
    # Single-File Services (Non-File-Based)
    # ========================================================================

    def create_inaturalist_service(self) -> INaturalistService:
        """Create INaturalistService (API service, no file repository)."""
        return INaturalistService()

    def create_xeno_canto_service(self) -> XenoCantoService:
        """Create XenoCantoService (API service, no file repository)."""
        return XenoCantoService()

    def create_macaulay_service(self) -> MacaulayService:
        """Create MacaulayService (API service, no file repository)."""
        return MacaulayService()

    def create_ebird_service(self) -> EBirdService:
        """Create EBirdService (API service, no file repository)."""
        return EBirdService()

    def create_huggingface_service(self) -> HuggingFaceService:
        """Create HuggingFaceService (API service, no file repository)."""
        return HuggingFaceService()

    def create_species_service(self) -> SpeciesService:
        """Create SpeciesService (in-memory service, no file repository)."""
        return SpeciesService()

    def create_clustering_service(self) -> ClusteringService:
        """Create ClusteringService (in-memory service, no file repository)."""
        return ClusteringService()

    def create_config_service(self) -> ConfigService:
        """Create ConfigService (config service, has own system)."""
        return ConfigService()

    def create_dependency_service(self) -> DependencyService:
        """Create DependencyService (system checker, no file repository)."""
        return DependencyService()

    def create_utility_service(self) -> UtilityService:
        """Create UtilityService (system info, no file repository)."""
        return UtilityService()

    # ========================================================================
    # Batch Services (File-Based)
    # ========================================================================

    def create_batch_audio_info_service(self) -> BatchAudioInfoService:
        """Create BatchAudioInfoService with file repository."""
        return BatchAudioInfoService(
            file_repository=self.file_repository,
        )

    def create_batch_audio_transform_service(self) -> BatchAudioTransformService:
        """Create BatchAudioTransformService with file repository and audio transform service."""
        audio_transform_service = self.create_audio_transform_service()
        return BatchAudioTransformService(
            file_repository=self.file_repository,
            audio_transform_service=audio_transform_service,
        )

    def create_batch_detection_service(self) -> BatchDetectionService:
        """Create BatchDetectionService with file repository and detection service."""
        detection_service = self.create_detection_service()
        return BatchDetectionService(
            file_repository=self.file_repository,
            detection_service=detection_service,
        )

    def create_batch_indices_service(self) -> BatchIndicesService:
        """Create BatchIndicesService with file repository and indices service."""
        return BatchIndicesService(
            file_repository=self.file_repository,
            indices_service=self.create_indices_service(),
        )

    def create_batch_inference_service(self) -> BatchInferenceService:
        """Create BatchInferenceService with file repository and inference service (AST-only)."""
        inference_service = self.create_inference_service()
        return BatchInferenceService(
            file_repository=self.file_repository,
            inference_service=inference_service,
        )

    def create_batch_clustering_service(self) -> BatchClusteringService:
        """Create BatchClusteringService with file repository and clustering service."""
        clustering_service = self.create_clustering_service()
        return BatchClusteringService(
            file_repository=self.file_repository,
            clustering_service=clustering_service,
        )

    # ========================================================================
    # Property-Based Access (Convenience for CLI and other applications)
    # ========================================================================

    @property
    def audio_file(self) -> AudioFileService:
        """Convenience property for create_audio_file_service()."""
        return self.create_audio_file_service()

    @property
    def audio_transform(self) -> AudioTransformService:
        """Convenience property for create_audio_transform_service()."""
        return self.create_audio_transform_service()

    @property
    def file(self) -> FileService:
        """Convenience property for create_file_service()."""
        return self.create_file_service()

    @property
    def dataset(self) -> DatasetService:
        """Convenience property for create_dataset_service()."""
        return self.create_dataset_service()

    @property
    def annotation(self) -> AnnotationService:
        """Convenience property for create_annotation_service()."""
        return self.create_annotation_service()

    @property
    def detection(self) -> DetectionService:
        """Convenience property for create_detection_service()."""
        return self.create_detection_service()

    @property
    def indices(self) -> IndicesService:
        """Convenience property for create_indices_service()."""
        return self.create_indices_service()

    @property
    def ribbit(self) -> RibbitService:
        """Convenience property for create_ribbit_service()."""
        return self.create_ribbit_service()

    @property
    def embedding(self) -> EmbeddingService:
        """Convenience property for create_embedding_service()."""
        return self.create_embedding_service()

    @property
    def inference(self) -> InferenceService:
        """Convenience property for create_inference_service()."""
        return self.create_inference_service()

    @property
    def ast(self) -> ASTService:
        """Convenience property for create_ast_service()."""
        return self.create_ast_service()

    @property
    def birdnet(self) -> BirdNETService:
        """Convenience property for create_birdnet_service()."""
        return self.create_birdnet_service()

    @property
    def cnn(self) -> CNNService:
        """Convenience property for create_cnn_service()."""
        return self.create_cnn_service()

    @property
    def inaturalist(self) -> INaturalistService:
        """Convenience property for create_inaturalist_service()."""
        return self.create_inaturalist_service()

    @property
    def xeno_canto(self) -> XenoCantoService:
        """Convenience property for create_xeno_canto_service()."""
        return self.create_xeno_canto_service()

    @property
    def macaulay(self) -> MacaulayService:
        """Convenience property for create_macaulay_service()."""
        return self.create_macaulay_service()

    @property
    def ebird(self) -> EBirdService:
        """Convenience property for create_ebird_service()."""
        return self.create_ebird_service()

    @property
    def huggingface(self) -> HuggingFaceService:
        """Convenience property for create_huggingface_service()."""
        return self.create_huggingface_service()

    @property
    def species(self) -> SpeciesService:
        """Convenience property for create_species_service()."""
        return self.create_species_service()

    @property
    def clustering(self) -> ClusteringService:
        """Convenience property for create_clustering_service()."""
        return self.create_clustering_service()

    @property
    def config(self) -> ConfigService:
        """Convenience property for create_config_service()."""
        return self.create_config_service()

    @property
    def dependency(self) -> DependencyService:
        """Convenience property for create_dependency_service()."""
        return self.create_dependency_service()

    @property
    def util(self) -> UtilityService:
        """Convenience property for create_utility_service()."""
        return self.create_utility_service()

    @property
    def batch_audio_info(self) -> BatchAudioInfoService:
        """Convenience property for create_batch_audio_info_service()."""
        return self.create_batch_audio_info_service()

    @property
    def batch_audio_transform(self) -> BatchAudioTransformService:
        """Convenience property for create_batch_audio_transform_service()."""
        return self.create_batch_audio_transform_service()

    @property
    def batch_detection(self) -> BatchDetectionService:
        """Convenience property for create_batch_detection_service()."""
        return self.create_batch_detection_service()

    @property
    def batch_indices(self) -> BatchIndicesService:
        """Convenience property for create_batch_indices_service()."""
        return self.create_batch_indices_service()

    @property
    def batch_inference(self) -> BatchInferenceService:
        """Convenience property for create_batch_inference_service()."""
        return self.create_batch_inference_service()

    @property
    def batch_clustering(self) -> BatchClusteringService:
        """Convenience property for create_batch_clustering_service()."""
        return self.create_batch_clustering_service()


__all__ = ["ServiceFactory"]

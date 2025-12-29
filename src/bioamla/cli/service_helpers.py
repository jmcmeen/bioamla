"""
CLI Service Helpers
===================

CLI-specific utilities for working with services and the ServiceFactory.

This module provides convenience functions for CLI commands to:
1. Access a singleton ServiceFactory instance
2. Handle service result errors consistently
3. Reduce boilerplate in command implementations

Architecture:
- Module-level singleton factory for consistent service instances across CLI
- Error handling utilities for consistent user feedback
- CLI-specific conveniences without polluting the reusable ServiceFactory
- LAZY IMPORTS: ServiceFactory is not imported until first service access

Usage:
    from bioamla.cli.service_helpers import services, handle_result

    # Access services via singleton factory
    audio_file_svc = services.audio_file
    result = audio_file_svc.open("file.wav")

    # Handle result with automatic error reporting
    audio_data = handle_result(result)  # Exits with error message if failed

    # Or manually check
    if not result.success:
        exit_with_error(result.error)
"""

from typing import TYPE_CHECKING, Any, TypeVar

import click

# LAZY IMPORT: ServiceFactory is only imported when first service is accessed
# This avoids loading all service dependencies (pyinaturalist, pandas, etc.) at CLI startup
if TYPE_CHECKING:
    from bioamla.services import ServiceFactory
    from bioamla.services.base import ServiceResult

if TYPE_CHECKING:
    from bioamla.services.annotation import AnnotationService
    from bioamla.services.ast import ASTService
    from bioamla.services.audio_file import AudioFileService
    from bioamla.services.audio_transform import AudioTransformService
    from bioamla.services.batch_audio_info import BatchAudioInfoService
    from bioamla.services.batch_audio_transform import BatchAudioTransformService
    from bioamla.services.batch_detection import BatchDetectionService
    from bioamla.services.batch_indices import BatchIndicesService
    from bioamla.services.clustering import ClusteringService
    from bioamla.services.cnn import CNNService
    from bioamla.services.config import ConfigService
    from bioamla.services.dataset import DatasetService
    from bioamla.services.dependency import DependencyService
    from bioamla.services.detection import DetectionService
    from bioamla.services.ebird import EBirdService
    from bioamla.services.embedding import EmbeddingService
    from bioamla.services.file import FileService
    from bioamla.services.huggingface import HuggingFaceService
    from bioamla.services.inaturalist import INaturalistService
    from bioamla.services.indices import IndicesService
    from bioamla.services.inference import InferenceService
    from bioamla.services.macaulay import MacaulayService
    from bioamla.services.ribbit import RibbitService
    from bioamla.services.species import SpeciesService
    from bioamla.services.util import UtilityService
    from bioamla.services.xeno_canto import XenoCantoService

# Type variable for generic result handling
T = TypeVar("T")


# ============================================================================
# Singleton Factory Instance
# ============================================================================

# Module-level singleton factory for CLI commands
# All CLI commands share this factory instance for consistency
# Type annotation uses string to avoid import at module level
_factory: "ServiceFactory | None" = None


def get_factory() -> "ServiceFactory":
    """
    Get the singleton ServiceFactory instance for CLI.

    This is lazily initialized on first access. For testing or custom
    configurations, use set_factory() to inject a custom instance.

    Returns:
        ServiceFactory: The singleton factory instance with LocalFileRepository
    """
    global _factory
    if _factory is None:
        # Lazy import to avoid loading all service dependencies at CLI startup
        from bioamla.services import ServiceFactory

        _factory = ServiceFactory()
    return _factory


def set_factory(factory: "ServiceFactory") -> None:
    """
    Set a custom ServiceFactory instance.

    This is primarily useful for testing or when you need to inject
    a custom file repository (e.g., S3Repository for cloud environments).

    Args:
        factory: Custom ServiceFactory instance to use

    Example:
        # In tests
        from bioamla.repository.mock import MockFileRepository
        mock_repo = MockFileRepository()
        custom_factory = ServiceFactory(file_repository=mock_repo)
        set_factory(custom_factory)
    """
    global _factory
    _factory = factory


class _ServiceAccessor:
    """
    Lazy accessor for services that provides type hints and autocomplete.

    This class provides property-based access to all services while maintaining
    type safety and IDE support. Services are accessed through the singleton
    factory, which is only created when first accessed.
    """

    @property
    def audio_file(self) -> "AudioFileService":
        """Get AudioFileService instance."""
        return get_factory().audio_file

    @property
    def audio_transform(self) -> "AudioTransformService":
        """Get AudioTransformService instance."""
        return get_factory().audio_transform

    @property
    def file(self) -> "FileService":
        """Get FileService instance."""
        return get_factory().file

    @property
    def dataset(self) -> "DatasetService":
        """Get DatasetService instance."""
        return get_factory().dataset

    @property
    def annotation(self) -> "AnnotationService":
        """Get AnnotationService instance."""
        return get_factory().annotation

    @property
    def detection(self) -> "DetectionService":
        """Get DetectionService instance."""
        return get_factory().detection

    @property
    def indices(self) -> "IndicesService":
        """Get IndicesService instance."""
        return get_factory().indices

    @property
    def ribbit(self) -> "RibbitService":
        """Get RibbitService instance."""
        return get_factory().ribbit

    @property
    def embedding(self) -> "EmbeddingService":
        """Get EmbeddingService instance."""
        return get_factory().embedding

    @property
    def inference(self) -> "InferenceService":
        """Get InferenceService instance."""
        return get_factory().inference

    @property
    def ast(self) -> "ASTService":
        """Get ASTService instance."""
        return get_factory().ast

    @property
    def cnn(self) -> "CNNService":
        """Get CNNService instance."""
        return get_factory().cnn

    @property
    def clustering(self) -> "ClusteringService":
        """Get ClusteringService instance."""
        return get_factory().clustering

    @property
    def config(self) -> "ConfigService":
        """Get ConfigService instance."""
        return get_factory().config

    @property
    def inaturalist(self) -> "INaturalistService":
        """Get INaturalistService instance."""
        return get_factory().inaturalist

    @property
    def xeno_canto(self) -> "XenoCantoService":
        """Get XenoCantoService instance."""
        return get_factory().xeno_canto

    @property
    def macaulay(self) -> "MacaulayService":
        """Get MacaulayService instance."""
        return get_factory().macaulay

    @property
    def ebird(self) -> "EBirdService":
        """Get EBirdService instance."""
        return get_factory().ebird

    @property
    def species(self) -> "SpeciesService":
        """Get SpeciesService instance."""
        return get_factory().species

    @property
    def huggingface(self) -> "HuggingFaceService":
        """Get HuggingFaceService instance."""
        return get_factory().huggingface

    @property
    def dependency(self) -> "DependencyService":
        """Get DependencyService instance."""
        return get_factory().dependency

    @property
    def util(self) -> "UtilityService":
        """Get UtilityService instance."""
        return get_factory().util

    # Add batch services
    @property
    def batch_audio_info(self) -> "BatchAudioInfoService":
        """Get BatchAudioInfoService instance."""
        return get_factory().batch_audio_info

    @property
    def batch_audio_transform(self) -> "BatchAudioTransformService":
        """Get BatchAudioTransformService instance."""
        return get_factory().batch_audio_transform

    @property
    def batch_detection(self) -> "BatchDetectionService":
        """Get BatchDetectionService instance."""
        return get_factory().batch_detection

    @property
    def batch_indices(self) -> "BatchIndicesService":
        """Get BatchIndicesService instance."""
        return get_factory().batch_indices


# Lazy service accessor - factory only created when services are actually accessed
services = _ServiceAccessor()


# ============================================================================
# Result Handling Utilities
# ============================================================================


def handle_result(result: "ServiceResult[T]") -> T:
    """
    Handle a service result, exiting with error if failed.

    This is a convenience function for CLI commands that want to
    automatically handle errors and extract the data from successful results.

    Args:
        result: Service result to handle

    Returns:
        The result data if successful

    Raises:
        SystemExit: If result indicates failure (exits with code 1)

    Example:
        result = services.audio_file.open("file.wav")
        audio_data = handle_result(result)  # Auto-exits on error
        # Use audio_data...
    """
    if not result.success:
        exit_with_error(result.error or "Unknown error")
    return result.data


def exit_with_error(message: str, code: int = 1) -> None:
    """
    Print error message and exit.

    Args:
        message: Error message to display
        code: Exit code (default: 1)

    Raises:
        SystemExit: Always exits with specified code
    """
    click.echo(f"Error: {message}", err=True)
    raise SystemExit(code)


def check_result(result: "ServiceResult[Any]", error_message: str | None = None) -> bool:
    """
    Check if a result is successful, exit with error if not.

    This is useful when you don't need the result data, only to verify success.

    Args:
        result: Service result to check
        error_message: Optional custom error message (uses result.error if None)

    Returns:
        True if successful (always returns True, exits otherwise)

    Raises:
        SystemExit: If result indicates failure (exits with code 1)

    Example:
        result = services.file.write_json(data, "output.json")
        check_result(result)  # Exits if failed
        click.echo("Success!")
    """
    if not result.success:
        msg = error_message or result.error or "Unknown error"
        exit_with_error(msg)
    return True


# ============================================================================
# Factory Reset (for testing)
# ============================================================================


def reset_factory() -> None:
    """
    Reset the singleton factory instance.

    This is primarily useful for testing to ensure a clean factory state.
    Normal CLI commands should not need to call this.
    """
    global _factory
    _factory = None


__all__ = [
    "services",
    "get_factory",
    "set_factory",
    "handle_result",
    "exit_with_error",
    "check_result",
    "reset_factory",
]

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

from typing import Any, TypeVar

import click

from bioamla.services import ServiceFactory
from bioamla.services.base import ServiceResult

# Type variable for generic result handling
T = TypeVar("T")


# ============================================================================
# Singleton Factory Instance
# ============================================================================

# Module-level singleton factory for CLI commands
# All CLI commands share this factory instance for consistency
_factory: ServiceFactory | None = None


def get_factory() -> ServiceFactory:
    """
    Get the singleton ServiceFactory instance for CLI.

    This is lazily initialized on first access. For testing or custom
    configurations, use set_factory() to inject a custom instance.

    Returns:
        ServiceFactory: The singleton factory instance with LocalFileRepository
    """
    global _factory
    if _factory is None:
        _factory = ServiceFactory()
    return _factory


def set_factory(factory: ServiceFactory) -> None:
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
    def audio_file(self):
        """Get AudioFileService instance."""
        from bioamla.services import AudioFileService
        return get_factory().audio_file

    @property
    def audio_transform(self):
        """Get AudioTransformService instance."""
        from bioamla.services import AudioTransformService
        return get_factory().audio_transform

    @property
    def file(self):
        """Get FileService instance."""
        from bioamla.services import FileService
        return get_factory().file

    @property
    def dataset(self):
        """Get DatasetService instance."""
        from bioamla.services import DatasetService
        return get_factory().dataset

    @property
    def annotation(self):
        """Get AnnotationService instance."""
        from bioamla.services import AnnotationService
        return get_factory().annotation

    @property
    def detection(self):
        """Get DetectionService instance."""
        from bioamla.services import DetectionService
        return get_factory().detection

    @property
    def indices(self):
        """Get IndicesService instance."""
        from bioamla.services import IndicesService
        return get_factory().indices

    @property
    def ribbit(self):
        """Get RibbitService instance."""
        from bioamla.services import RibbitService
        return get_factory().ribbit

    @property
    def embedding(self):
        """Get EmbeddingService instance."""
        from bioamla.services import EmbeddingService
        return get_factory().embedding

    @property
    def inference(self):
        """Get InferenceService instance."""
        from bioamla.services import InferenceService
        return get_factory().inference

    @property
    def ast(self):
        """Get ASTService instance."""
        from bioamla.services import ASTService
        return get_factory().ast

    @property
    def clustering(self):
        """Get ClusteringService instance."""
        from bioamla.services import ClusteringService
        return get_factory().clustering

    @property
    def config(self):
        """Get ConfigService instance."""
        from bioamla.services import ConfigService
        return get_factory().config

    @property
    def inaturalist(self):
        """Get INaturalistService instance."""
        from bioamla.services import INaturalistService
        return get_factory().inaturalist

    @property
    def xeno_canto(self):
        """Get XenoCantoService instance."""
        from bioamla.services import XenoCantoService
        return get_factory().xeno_canto

    # Add batch services
    @property
    def batch_audio_transform(self):
        """Get BatchAudioTransformService instance."""
        from bioamla.services import BatchAudioTransformService
        return get_factory().batch_audio_transform

    @property
    def batch_detection(self):
        """Get BatchDetectionService instance."""
        from bioamla.services import BatchDetectionService
        return get_factory().batch_detection

    @property
    def batch_indices(self):
        """Get BatchIndicesService instance."""
        from bioamla.services import BatchIndicesService
        return get_factory().batch_indices


# Lazy service accessor - factory only created when services are actually accessed
services = _ServiceAccessor()


# ============================================================================
# Result Handling Utilities
# ============================================================================


def handle_result(result: ServiceResult[T]) -> T:
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


def check_result(result: ServiceResult[Any], error_message: str | None = None) -> bool:
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

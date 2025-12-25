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

    Returns:
        ServiceFactory: The singleton factory instance with LocalFileRepository
    """
    global _factory
    if _factory is None:
        _factory = ServiceFactory()
    return _factory


# Convenient alias for CLI commands
services = get_factory()


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
    "handle_result",
    "exit_with_error",
    "check_result",
    "reset_factory",
]

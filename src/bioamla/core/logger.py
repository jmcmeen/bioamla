"""
Logging Configuration
=====================

This module provides centralized logging configuration for the bioamla package.
It replaces scattered verbose flags with proper Python logging, allowing
consistent and configurable log output across all modules.

Usage:
    from bioamla.logging import get_logger, configure_logging

    # In module code:
    logger = get_logger(__name__)
    logger.info("Processing started")

    # To configure logging level:
    configure_logging(verbose=True)  # Sets DEBUG level
    configure_logging(verbose=False)  # Sets WARNING level
"""

import logging
import sys
from typing import Optional

# Package-level logger name
PACKAGE_NAME = "bioamla"

# Default format for log messages
DEFAULT_FORMAT = "%(levelname)s: %(message)s"
VERBOSE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.

    Args:
        name: The module name (typically __name__)

    Returns:
        A configured Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing file...")
    """
    return logging.getLogger(name)


def configure_logging(
    verbose: bool = False,
    level: Optional[int] = None,
    format_string: Optional[str] = None,
    stream: Optional[object] = None,
) -> None:
    """
    Configure logging for the bioamla package.

    Args:
        verbose: If True, set level to DEBUG. If False, set to WARNING.
                 Ignored if level is explicitly provided.
        level: Explicit logging level (e.g., logging.INFO). Overrides verbose.
        format_string: Custom format string for log messages.
        stream: Output stream for logs (default: sys.stderr)

    Example:
        # For quiet operation:
        configure_logging(verbose=False)

        # For detailed output:
        configure_logging(verbose=True)

        # For custom level:
        configure_logging(level=logging.INFO)
    """
    # Determine the logging level
    if level is not None:
        log_level = level
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING

    # Determine format
    if format_string is None:
        format_string = VERBOSE_FORMAT if verbose else DEFAULT_FORMAT

    # Configure the package logger
    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create and configure handler
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(format_string))

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False


def set_verbosity(verbose: bool) -> None:
    """
    Quick helper to set verbosity level.

    Args:
        verbose: If True, enables INFO level logging.
                 If False, enables WARNING level only.

    This is a convenience function for use in CLI commands to translate
    the common verbose/quiet flags into logging configuration.
    """
    level = logging.INFO if verbose else logging.WARNING
    configure_logging(level=level)


class LoggingContext:
    """
    Context manager for temporarily changing logging level.

    Example:
        with LoggingContext(logging.DEBUG):
            # Detailed logging here
            process_data()
        # Back to original level
    """

    def __init__(self, level: int, logger_name: str = PACKAGE_NAME):
        self.level = level
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.original_level: Optional[int] = None

    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_level is not None:
            self.logger.setLevel(self.original_level)
        return False

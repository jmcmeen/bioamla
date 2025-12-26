"""
Logging Configuration
=====================

Centralized logging for the bioamla package.

Usage:
    from bioamla.core.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started")
"""

import logging
import sys

PACKAGE_NAME = "bioamla"
LOG_FORMAT = "%(levelname)s - %(name)s - %(message)s"

_configured = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.

    Args:
        name: The module name (typically __name__)

    Returns:
        A configured Logger instance
    """
    _ensure_configured()
    return logging.getLogger(name)


def _ensure_configured() -> None:
    """Configure the package logger if not already done."""
    global _configured
    if _configured:
        return

    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    logger.propagate = False
    _configured = True


def set_level(level: int) -> None:
    """
    Set the logging level for the bioamla package.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.WARNING)
    """
    _ensure_configured()
    logging.getLogger(PACKAGE_NAME).setLevel(level)

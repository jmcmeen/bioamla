"""Tests for bioamla.cli.logging_setup."""

from __future__ import annotations

import logging

import pytest

from bioamla.cli.logging_setup import configure_cli_logging


@pytest.fixture
def clean_bioamla_logger():
    """Snapshot and restore the bioamla logger state around each test."""
    logger = logging.getLogger("bioamla")
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    saved_propagate = logger.propagate
    logger.handlers = []
    yield logger
    logger.handlers = saved_handlers
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate


def test_configure_adds_handler(clean_bioamla_logger):
    logger = clean_bioamla_logger
    assert logger.handlers == []
    configure_cli_logging()
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.level == logging.INFO
    assert logger.propagate is False


def test_configure_idempotent(clean_bioamla_logger):
    logger = clean_bioamla_logger
    configure_cli_logging()
    configure_cli_logging()
    configure_cli_logging()
    # Only one handler added despite repeated calls
    assert len(logger.handlers) == 1


def test_configure_custom_level(clean_bioamla_logger):
    logger = clean_bioamla_logger
    configure_cli_logging(level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_configure_updates_level_without_new_handler(clean_bioamla_logger):
    logger = clean_bioamla_logger
    configure_cli_logging(level=logging.WARNING)
    configure_cli_logging(level=logging.ERROR)
    assert logger.level == logging.ERROR
    assert len(logger.handlers) == 1

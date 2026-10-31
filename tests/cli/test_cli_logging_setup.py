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
    assert len(logger.handlers) == 2
    assert all(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert logger.level == logging.INFO
    assert logger.propagate is False


def test_progress_to_stdout_problems_to_stderr(clean_bioamla_logger, capsys):
    configure_cli_logging()
    log = logging.getLogger("bioamla.test")
    log.info("progress line")
    log.warning("problem line")
    captured = capsys.readouterr()
    assert captured.out == "progress line\n"
    assert captured.err == "problem line\n"


def test_configure_idempotent(clean_bioamla_logger):
    logger = clean_bioamla_logger
    configure_cli_logging()
    configure_cli_logging()
    configure_cli_logging()
    # Handlers are added once despite repeated calls
    assert len(logger.handlers) == 2


def test_configure_custom_level(clean_bioamla_logger):
    logger = clean_bioamla_logger
    configure_cli_logging(level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_configure_updates_level_without_new_handler(clean_bioamla_logger):
    logger = clean_bioamla_logger
    configure_cli_logging(level=logging.WARNING)
    configure_cli_logging(level=logging.ERROR)
    assert logger.level == logging.ERROR
    assert len(logger.handlers) == 2

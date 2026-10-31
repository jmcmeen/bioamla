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


# --- --log-format / --log-level ----------------------------------------------


@pytest.fixture
def reset_cli_log_options():
    from bioamla.cli import logging_setup

    yield
    logging_setup.set_cli_logging("text", None)


def test_json_format_writes_json_lines_to_stderr(
    clean_bioamla_logger, reset_cli_log_options, capsys
):
    import json

    from bioamla.cli.logging_setup import set_cli_logging
    from bioamla.common.log import log_fields

    set_cli_logging("json", None)
    logging.getLogger("bioamla.test").info("hi", **log_fields(file="a.wav"))
    captured = capsys.readouterr()
    assert captured.out == ""
    record = json.loads(captured.err)
    assert (record["message"], record["file"]) == ("hi", "a.wav")


def test_log_level_overrides_command_default(clean_bioamla_logger, reset_cli_log_options):
    from bioamla.cli.logging_setup import set_cli_logging

    set_cli_logging("text", "debug")
    configure_cli_logging()  # a command asking for INFO keeps the chosen DEBUG
    assert clean_bioamla_logger.level == logging.DEBUG


def test_root_options_reach_a_batch_command(clean_bioamla_logger, tmp_path, test_audio_path):
    import json
    import shutil

    from click.testing import CliRunner

    from bioamla.cli.cli import cli

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    shutil.copy(test_audio_path, in_dir / "a.wav")
    result = CliRunner().invoke(
        cli,
        [
            "--log-format",
            "json",
            "--log-level",
            "debug",
            "batch",
            "audio",
            "info",
            "--input-dir",
            str(in_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "-q",
        ],
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
    batch = [r for r in records if r["logger"] == "bioamla.batch"]
    assert batch and batch[0]["operation"] == "info"  # the domain function names it
    assert batch[-1]["status"] == "finished"

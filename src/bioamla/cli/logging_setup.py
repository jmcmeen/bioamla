"""Console logging setup for CLI commands that run long library operations.

Library code logs progress via ``logging.getLogger(__name__)`` rather than
printing. For long-running commands (e.g. ``models ast train``) we attach plain
handlers to the ``bioamla`` logger so that progress is visible: records below
WARNING go to stdout, warnings and errors to stderr.

The root options ``--log-format json`` and ``--log-level`` (see
:func:`set_cli_logging`) apply to every command. In JSON mode every record goes
to stderr as one JSON object per line, with its structured fields
(:data:`bioamla.common.log.LOG_FIELDS`), so stdout keeps only the command's output.
"""

import logging
import sys

LOG_FORMATS = ("text", "json")
LOG_LEVELS = ("debug", "info", "warning", "error")

# Chosen with the root options; None/"text" mean "not set on the command line".
_log_format = "text"
_log_level: int | None = None


def set_cli_logging(log_format: str = "text", log_level: str | None = None) -> None:
    """Record the root ``--log-format``/``--log-level`` choices and apply them.

    With either option given, the ``bioamla`` logger is configured right away, so
    every command's log records follow it, not only the commands that turn on
    progress logging themselves.
    """
    global _log_format, _log_level
    _log_format = log_format
    _log_level = getattr(logging, log_level.upper()) if log_level else None
    if log_format != "text" or log_level:
        configure_cli_logging()


def configure_cli_logging(level: int = logging.INFO) -> None:
    """Ensure the ``bioamla`` logger emits records to the console at ``level``.

    A level chosen with ``--log-level`` takes precedence over ``level``.
    Idempotent: handlers are added only if the logger has none, so repeated
    calls (and an already-configured root logger) don't duplicate output.
    """
    bioamla_logger = logging.getLogger("bioamla")
    if not bioamla_logger.handlers and _log_format == "json":
        from bioamla.common.log import JsonFormatter

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        bioamla_logger.addHandler(handler)
        bioamla_logger.propagate = False
    elif not bioamla_logger.handlers:
        formatter = logging.Formatter("%(message)s")

        progress = logging.StreamHandler(sys.stdout)
        progress.addFilter(lambda record: record.levelno < logging.WARNING)
        progress.setFormatter(formatter)

        problems = logging.StreamHandler(sys.stderr)
        problems.setLevel(logging.WARNING)
        problems.setFormatter(formatter)

        bioamla_logger.addHandler(progress)
        bioamla_logger.addHandler(problems)
        bioamla_logger.propagate = False
    bioamla_logger.setLevel(_log_level if _log_level is not None else level)

"""Console logging setup for CLI commands that run long library operations.

Library code logs progress via ``logging.getLogger(__name__)`` rather than
printing. For long-running commands (e.g. ``models ast train``) we attach plain
handlers to the ``bioamla`` logger so that progress is visible: records below
WARNING go to stdout, warnings and errors to stderr.
"""

import logging
import sys


def configure_cli_logging(level: int = logging.INFO) -> None:
    """Ensure the ``bioamla`` logger emits records to the console at ``level``.

    Idempotent: handlers are added only if the logger has none, so repeated
    calls (and an already-configured root logger) don't duplicate output.
    """
    bioamla_logger = logging.getLogger("bioamla")
    if not bioamla_logger.handlers:
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
    bioamla_logger.setLevel(level)

"""Console logging setup for CLI commands that run long library operations.

Library code logs progress via ``logging.getLogger(__name__)`` rather than
printing. For long-running commands (e.g. ``models ast train``) we attach a
plain stdout handler to the ``bioamla`` logger so that progress is visible.
"""

import logging


def configure_cli_logging(level: int = logging.INFO) -> None:
    """Ensure the ``bioamla`` logger emits records to stdout at ``level``.

    Idempotent: a handler is added only if the logger has none, so repeated
    calls (and an already-configured root logger) don't duplicate output.
    """
    bioamla_logger = logging.getLogger("bioamla")
    if not bioamla_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        bioamla_logger.addHandler(handler)
        bioamla_logger.propagate = False
    bioamla_logger.setLevel(level)

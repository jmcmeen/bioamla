"""Reusable Click option decorators shared across command groups.

These reproduce options that were previously re-declared inline in every batch
command (``--max-workers``/``--recursive``/``--quiet``), so names, short flags,
defaults, and help stay identical everywhere. Complements the input/output
decorators in :mod:`bioamla.cli.batch_options`.
"""

from collections.abc import Callable

import click


def max_workers_option(f: Callable) -> Callable:
    """Add the standard ``--max-workers`` / ``-w`` parallelism option."""
    return click.option(
        "--max-workers", "-w", default=1, type=int, help="Number of parallel workers"
    )(f)


def recursive_option(f: Callable) -> Callable:
    """Add the standard ``--recursive/--no-recursive`` directory-search option."""
    return click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")(f)


def quiet_option(f: Callable) -> Callable:
    """Add the standard ``--quiet`` / ``-q`` progress-suppression flag."""
    return click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")(f)

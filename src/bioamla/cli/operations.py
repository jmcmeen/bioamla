"""Running operations from commands.

A converted command builds its operation's request from its options and calls
:func:`run_operation`, then prints the result. See :mod:`bioamla.operations`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click

from bioamla.exceptions import BioamlaError


def run_operation(
    name: str,
    /,
    on_progress: Callable[[int, int], None] | None = None,
    secrets: dict[str, str] | None = None,
    **request: Any,
) -> Any:
    """Run the operation ``name`` with ``request`` as its fields: locally, or on the
    run's remote server (see :mod:`bioamla.cli.remote_mode`).

    ``on_progress(done, total)`` receives the operation's progress, e.g. a progress
    bar's ``on_progress``. ``secrets`` holds credentials given on the command line
    (never put in the request). The operation's notices print as they come.

    Raises:
        click.UsageError: If arguments conflict (:class:`~bioamla.exceptions.OptionConflictError`).
        click.ClickException: If the request is invalid or the operation fails.
    """
    from bioamla import operations
    from bioamla.cli import remote_mode

    target = remote_mode.current()
    if target is not None:
        try:
            remote_ok = operations.get(name).remote
        except BioamlaError:
            remote_ok = True  # can't load it here; the server decides
        if remote_ok:
            return remote_mode.run(target, name, request, on_progress=on_progress,
                                   on_notice=_print_notice, secrets=secrets)  # fmt: skip
        if target.explicit:
            command = name.replace(".", " ")
            raise click.UsageError(f"`{command}` runs only on this machine; drop --remote")

    context = operations.OperationContext(
        on_progress=on_progress, secrets=dict(secrets or {}), on_notice=_print_notice
    )
    from bioamla.exceptions import OptionConflictError

    try:
        return operations.run(name, request, context)
    except OptionConflictError as e:
        raise click.UsageError(str(e)) from e
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


def _print_notice(level: str, message: str) -> None:
    from bioamla.cli.console import print_success, print_warning

    (print_warning if level == "warning" else print_success)(message)


__all__ = ["run_operation"]

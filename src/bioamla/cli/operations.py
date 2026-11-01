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
    name: str, /, on_progress: Callable[[int, int], None] | None = None, **request: Any
) -> Any:
    """Run the operation ``name`` locally with ``request`` as its fields.

    ``on_progress(done, total)`` receives the operation's progress, e.g. a progress
    bar's ``on_progress``.

    Raises:
        click.ClickException: If the request is invalid or the operation fails.
    """
    from bioamla import operations

    context = operations.OperationContext(on_progress=on_progress)
    try:
        return operations.run(name, request, context)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


__all__ = ["run_operation"]

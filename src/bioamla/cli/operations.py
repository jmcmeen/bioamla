"""Running operations from commands.

A converted command builds its operation's request from its options and calls
:func:`run_operation`, then prints the result. See :mod:`bioamla.operations`.
"""

from __future__ import annotations

from typing import Any

import click

from bioamla.exceptions import BioamlaError


def run_operation(name: str, **request: Any) -> Any:
    """Run the operation ``name`` locally with ``request`` as its fields.

    Raises:
        click.ClickException: If the request is invalid or the operation fails.
    """
    from bioamla import operations

    try:
        return operations.run(name, request, operations.OperationContext())
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


__all__ = ["run_operation"]

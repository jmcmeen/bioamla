"""Deprecated names kept importable for a while after a rename.

A module that renames a public name keeps the old one working through a PEP 562
module ``__getattr__`` built here. Accessing the old name returns the new object
and emits a :class:`DeprecationWarning` that points at the caller's line.
"""

import warnings
from collections.abc import Callable, Mapping
from typing import Any


def deprecated_aliases(
    module_name: str, namespace: dict[str, Any], aliases: Mapping[str, str]
) -> Callable[[str], Any]:
    """Build a module ``__getattr__`` that serves old names for renamed ones.

    Args:
        module_name: The module's ``__name__``, used in messages.
        namespace: The module's ``globals()``, where the new names live.
        aliases: Old name -> new name.

    Returns:
        A function to assign to the module's ``__getattr__``.

    Example:
        >>> __getattr__ = deprecated_aliases(__name__, globals(), {"OldName": "NewName"})
    """

    def __getattr__(name: str) -> Any:
        new = aliases.get(name)
        if new is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        warnings.warn(
            f"{module_name}.{name} is deprecated; use {module_name}.{new} instead. "
            "The old name will be removed in a future minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return namespace[new]

    return __getattr__

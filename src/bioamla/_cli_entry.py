"""Console-script entry point for the ``bioamla`` command.

The CLI dependencies (``click``, ``rich``) live in the optional ``[cli]`` extra,
not the base install (see ``pyproject.toml``). This thin shim is the target of
the ``bioamla`` console script: it imports nothing CLI-specific at module top,
so that on a library-only install (``pip install bioamla``) running ``bioamla``
prints a friendly "install the extra" message instead of a raw
``ModuleNotFoundError`` traceback.

Keep this module dependency-light: importing it must not pull in ``click`` /
``rich``, otherwise the guard below never gets a chance to run.
"""

from __future__ import annotations

import sys

# Names of the CLI-only dependencies. A missing one of these means the CLI extra
# was not installed; anything else is a genuine import error we let propagate.
_CLI_DEPS = frozenset({"click", "rich"})

_INSTALL_HINT = (
    "The bioamla CLI requires extra dependencies that are not installed.\n"
    "Install them with:  pip install 'bioamla[cli]'\n"
)


def main() -> None:
    """Run the bioamla CLI, or explain how to install it if the extra is absent."""
    try:
        from bioamla.cli.cli import main as _cli_main
    except ModuleNotFoundError as exc:
        if exc.name in _CLI_DEPS:
            sys.stderr.write(_INSTALL_HINT)
            raise SystemExit(1) from exc
        raise
    _cli_main()


if __name__ == "__main__":
    main()

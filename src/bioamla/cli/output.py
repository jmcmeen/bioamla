"""Machine-readable output: the ``bioamla --json`` contract.

In JSON mode stdout carries exactly one JSON document, an envelope:

    {"bioamla": "<version>", "command": "detect energy", "ok": true, "result": {...}}

or, when the command fails (exit status 1, or 2 for a usage error):

    {"bioamla": "<version>", "command": "detect energy", "ok": false,
     "error": {"type": "InvalidInputError", "message": "..."}}

``error.type`` is the :class:`~bioamla.exceptions.BioamlaError` subclass name, or a
Click error name (``UsageError``, ``BadParameter``, ...). Everything a command
prints for people (rich tables, progress bars, ``click.echo``) goes to stderr
instead: while JSON mode is on, ``sys.stdout`` points at stderr, and only the
envelope is written to the real stdout.

A command reports its result with :func:`emit_result` and keeps its human output
as it is; outside JSON mode :func:`emit_result` does nothing. A command that runs
in JSON mode without calling it gets ``"result": null``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import json
import math
import sys
from pathlib import PurePath
from typing import IO, Any

_real_stdout: IO[str] | None = None
_installed: IO[str] | None = None  # the stream we put on sys.stdout
_emitted = False


def json_mode() -> bool:
    """True while the command is running with ``--json``."""
    return _real_stdout is not None


def enable_json_mode() -> None:
    """Turn JSON mode on: from now on, anything printed goes to stderr.

    Idempotent. Called for the root ``--json`` flag and for the older per-command
    ``--json`` flags, which mean the same thing.
    """
    global _real_stdout, _installed
    if _real_stdout is None:
        _real_stdout = sys.stdout
        _installed = sys.stderr
        sys.stdout = _installed


def disable_json_mode() -> None:
    """Restore stdout and reset the state at the end of a run.

    stdout is put back only if it is still the stream JSON mode installed, so a
    caller that has since replaced it (a test runner, say) keeps its own.
    """
    global _real_stdout, _installed, _emitted
    if _real_stdout is not None and sys.stdout is _installed:
        sys.stdout = _real_stdout
    _real_stdout = None
    _installed = None
    _emitted = False


def to_jsonable(value: Any) -> Any:
    """Convert a result into JSON types.

    Dataclasses (their ``to_dict()`` when they have one), mappings and sequences
    are converted recursively. Paths, datetimes and enums become strings or their
    values, numpy scalars and arrays become numbers and lists, and NaN and
    infinity become ``None``. Anything else becomes ``str(value)``.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return to_jsonable(to_dict())
        # Field by field (not dataclasses.asdict, which deep-copies and skips the
        # to_dict() of nested dataclasses).
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    tolist = getattr(value, "tolist", None)  # numpy scalars and arrays
    if callable(tolist):
        return to_jsonable(tolist())
    return str(value)


def _write(envelope: dict[str, Any]) -> None:
    global _emitted
    out = _real_stdout or sys.stdout
    out.write(json.dumps(envelope, ensure_ascii=False) + "\n")
    out.flush()
    _emitted = True


def _envelope(command: str | None, **fields: Any) -> dict[str, Any]:
    from bioamla import __version__

    return {"bioamla": __version__, "command": command, **fields}


def current_command() -> str | None:
    """The running command's path without the program name, e.g. ``detect energy``."""
    import click

    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return None
    names = []
    while ctx.parent is not None:
        names.append(ctx.info_name or "")
        ctx = ctx.parent
    return " ".join(reversed(names)) or None


def emit_result(result: Any) -> None:
    """Report a command's result. Writes the success envelope in JSON mode only."""
    if json_mode():
        _write(_envelope(current_command(), ok=True, result=to_jsonable(result)))


def legacy_json(data: Any) -> None:
    """Old per-command ``--json`` output: bare JSON, no envelope, with a deprecation note.

    ``annotation summary``, ``audio metadata`` and ``dataset stats`` had their own
    ``--json`` before the root flag. They keep printing what they always printed;
    ``bioamla --json <command>`` gives the envelope.
    """
    import click

    click.echo(
        "Note: the command's own --json is deprecated and will be removed in a future minor "
        "release; use `bioamla --json "
        f"{current_command() or '<command>'}` for the standard JSON envelope.",
        err=True,
    )
    click.echo(json.dumps(data, indent=2))  # exactly what these flags printed before


def emit_error(command: str | None, error_type: str, message: str) -> None:
    """Write the failure envelope (JSON mode only)."""
    if json_mode():
        _write(_envelope(command, ok=False, error={"type": error_type, "message": message}))


def emit_failure(command: str | None, exc: BaseException) -> None:
    """Write the failure envelope for whatever ended the command (JSON mode only).

    A command that already reported its result, or exited with status 0, gets
    nothing more (:func:`finish` covers the latter).
    """
    import click

    from bioamla.exceptions import BioamlaError

    if not json_mode() or _emitted:
        return
    if isinstance(exc, click.ClickException):
        # Command bodies wrap a BioamlaError in a ClickException; report the original type.
        cause = exc.__cause__ if isinstance(exc.__cause__, BioamlaError) else exc
        emit_error(command, type(cause).__name__, exc.format_message())
    elif isinstance(exc, (click.exceptions.Abort, KeyboardInterrupt)):
        emit_error(command, "Aborted", "Interrupted")
    elif isinstance(exc, (SystemExit, click.exceptions.Exit)):
        code = exc.code if isinstance(exc, SystemExit) else exc.exit_code
        if code in (0, None):
            finish(command)
        else:
            emit_error(command, "Exit", f"The command exited with status {code}")
    elif isinstance(exc, ModuleNotFoundError):
        from bioamla._extras import extra_hint

        hint = extra_hint(exc)
        if hint is not None:
            emit_error(command, "DependencyError", f"{exc}. {hint}")
        else:
            emit_error(command, type(exc).__name__, str(exc))
    else:
        emit_error(command, type(exc).__name__, str(exc))


def json_flag_given(group: Any, args: list[str]) -> bool:
    """True if the root ``--json`` flag appears before the first subcommand.

    A ``--json`` after the subcommand belongs to that command (the older
    per-command flags), not to the root.
    """
    for arg in args:
        if arg == "--json":
            return True
        if not arg.startswith("-") and arg in group.commands:
            return False
    return False


def finish(command: str | None) -> None:
    """End of a successful run: a command that reported nothing gets ``result: null``."""
    if json_mode() and not _emitted:
        _write(_envelope(command, ok=True, result=None))


def command_from_args(group: Any, args: list[str]) -> str | None:
    """Best-effort command path from raw arguments, for errors raised outside a command."""
    import click

    names: list[str] = []
    node = group
    for arg in args:
        if arg.startswith("-") or not isinstance(node, click.Group):
            continue
        sub = node.commands.get(arg)
        if sub is None:
            continue
        names.append(arg)
        node = sub
    return " ".join(names) or None


__all__ = [
    "command_from_args",
    "emit_failure",
    "json_flag_given",
    "current_command",
    "disable_json_mode",
    "emit_error",
    "emit_result",
    "enable_json_mode",
    "finish",
    "json_mode",
    "legacy_json",
    "to_jsonable",
]

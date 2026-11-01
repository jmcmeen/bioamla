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

import json
import sys
from typing import IO, Any, NoReturn

_real_stdout: IO[str] | None = None
_installed: IO[str] | None = None  # the stream we put on sys.stdout
_emitted = False
# The run's result and error, remembered in every mode (not only --json) so the
# workspace history can record what a command produced.
_last_result: Any = None
_last_error: dict[str, Any] | None = None
_run_workspace: Any = None


def reset_run_record() -> None:
    """Forget the previous run's result, error and workspace (start and end of a run)."""
    global _last_result, _last_error, _run_workspace
    _last_result = None
    _last_error = None
    _run_workspace = None


def last_result() -> Any:
    """The result the running command reported, as the command passed it, or ``None``."""
    return _last_result


def note_workspace(workspace: Any) -> None:
    """Say which workspace the running command works on (for its history record)."""
    global _run_workspace
    _run_workspace = workspace


def run_workspace() -> Any:
    """The workspace the running command said it works on, or ``None``."""
    return _run_workspace


def last_error() -> dict[str, Any] | None:
    """The error the running command ended with, as ``{"type", "message"}``, or ``None``."""
    return _last_error


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


from bioamla.common.jsonable import to_jsonable  # noqa: E402  (re-exported)


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


def command_options(exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """The running command's option values, keyed by their long option name.

    ``--threshold-db`` becomes ``threshold_db`` (the option name with ``-`` as
    ``_``, not the Python parameter name), so each key names an option a script can
    find in ``--help``. Unset options are ``None``. Arguments and the parameters
    named in ``exclude`` are left out.
    """
    import click

    ctx = click.get_current_context()
    options: dict[str, Any] = {}
    for param in ctx.command.params:
        # --quiet only changes what people see, so it isn't part of the result.
        if not isinstance(param, click.Option) or param.name in (*exclude, "quiet"):
            continue
        if param.name not in ctx.params:
            continue
        longs = [o for o in param.opts if o.startswith("--")]
        key = (longs[0][2:] if longs else param.name or "").replace("-", "_")
        options[key] = ctx.params[param.name]
    return options


_NO_OUTPUT = object()


def emit_file_result(file: str, output: Any = _NO_OUTPUT, **fields: Any) -> bool:
    """Report a per-file command's result: the file, its options and ``fields``.

    The result is ``{"file", "params", **fields}`` plus ``"output"``: the
    ``output`` argument when given (the path the command wrote), else the
    ``--output`` value for commands that have that option. Returns True in JSON
    mode, so the caller can skip its human output:
    ``if emit_file_result(file, count=n): return``.
    """
    import click

    ctx = click.get_current_context()
    result: dict[str, Any] = {
        "file": file,
        "params": command_options(exclude=("output", "output_format")),
        **fields,
    }
    if output is not _NO_OUTPUT:
        result["output"] = None if output is None else str(output)
    elif "output" in ctx.params:
        result["output"] = ctx.params["output"]
    return emit_result(result)


def emit_result(result: Any) -> bool:
    """Report a command's result: the success envelope, in JSON mode only.

    Returns True if it reported, so a command can skip its human output:
    ``if emit_result({...}): return``.
    """
    global _last_result
    _last_result = result  # kept as-is; the history reads only its output paths
    if not json_mode():
        return False
    _write(_envelope(current_command(), ok=True, result=to_jsonable(result)))
    return True


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


def emit_error(command: str | None, error_type: str, message: str, details: Any = None) -> None:
    """Write the failure envelope (JSON mode only).

    ``details`` (optional) goes in ``error.details``: structured data about the
    failure, such as which items of a batch failed and why.
    """
    global _last_error
    _last_error = {"type": error_type, "message": message}
    if json_mode():
        error = dict(_last_error)
        if details is not None:
            error["details"] = to_jsonable(details)
        _write(_envelope(command, ok=False, error=error))


def fail(message: str, error_type: str, details: Any = None) -> NoReturn:
    """End the command with an error that carries structured ``details`` in JSON mode.

    People see ``Error: <message>`` as usual; a ``--json`` caller also gets
    ``error.details``.
    """
    import click

    emit_error(current_command(), error_type, message, details)
    raise click.ClickException(message)


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
    "command_options",
    "emit_failure",
    "emit_file_result",
    "json_flag_given",
    "current_command",
    "disable_json_mode",
    "emit_error",
    "emit_result",
    "fail",
    "enable_json_mode",
    "finish",
    "json_mode",
    "last_error",
    "last_result",
    "note_workspace",
    "reset_run_record",
    "run_workspace",
    "legacy_json",
    "to_jsonable",
]

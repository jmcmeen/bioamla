"""Running commands on a server: ``bioamla --remote NAME[:WORKSPACE] ...``.

Where a command runs is configuration, not a different command. The remote comes from,
first to last:

1. ``--remote NAME[:WORKSPACE]`` on the command line;
2. the ``BIOAMLA_REMOTE`` environment variable (same form);
3. ``remote = "NAME"`` and ``workspace = "WORKSPACE"`` at the top of the local
   workspace's ``.bioamla/config.toml``.

With a remote, a command builds its operation's request from its options as usual,
and :func:`run` sends it to the server (``bioamla.client``) instead of running it:
progress and notices stream back, and the result comes back as the same objects a
local run returns, so the command prints exactly what it prints locally. Paths are the
server's, relative to the remote workspace, so local existence checks are skipped.

Commands that only make sense here (``serve``, ``remote ...``, ``system commands``,
``workspace config ...``) and operations that only run locally (``audio play``, ...)
refuse an explicit ``--remote``; with a remote from the environment or a workspace's
config they run locally as usual.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import click

from bioamla.exceptions import BioamlaError

ENV = "BIOAMLA_REMOTE"

#: Command paths that always run on this machine.
LOCAL_COMMANDS = ("serve", "remote", "system commands", "workspace config")


@dataclass(frozen=True)
class Target:
    """Where this run's operations go.

    Attributes:
        name: The remote's name.
        workspace: The workspace on it (``None`` if none was given).
        explicit: Whether it came from ``--remote`` (rather than the environment or a
            workspace's config).
        source: Where it came from, for messages.
    """

    name: str
    workspace: str | None
    explicit: bool
    source: str


_current: Target | None = None
_used = False  # whether this run sent anything to a server


def current() -> Target | None:
    """The remote for this run, if any."""
    return _current


def used() -> bool:
    """Whether this run sent an operation to a server (so it isn't recorded locally)."""
    return _used


def parse(value: str, explicit: bool, source: str) -> Target:
    """``NAME[:WORKSPACE]`` as a :class:`Target`."""
    name, _, workspace = value.strip().partition(":")
    if not name:
        raise click.UsageError(f"{source}: give a remote as NAME or NAME:WORKSPACE")
    return Target(name, workspace or None, explicit, source)


def resolve(option: str | None, start: str | None = None) -> Target | None:
    """The remote for a run: ``option`` (``--remote``), else ``$BIOAMLA_REMOTE``, else
    the local workspace's ``remote``/``workspace`` settings."""
    if option:
        return parse(option, True, "--remote")
    if os.environ.get(ENV, "").strip():
        return parse(os.environ[ENV], False, f"${ENV}")
    from bioamla.workspace import find_workspace

    try:
        ws = find_workspace(start)
        settings = ws.settings() if ws is not None else {}
    except BioamlaError:
        return None  # a broken settings file is reported by the settings loader
    remote = settings.get("remote")
    if isinstance(remote, str) and remote.strip():
        workspace = settings.get("workspace")
        target = parse(remote, False, f"{ws.config_path}")
        if target.workspace is None and isinstance(workspace, str) and workspace.strip():
            target = Target(target.name, workspace.strip(), False, target.source)
        return target
    return None


def begin(target: Target | None, command: str | None) -> None:
    """Start a run with ``target`` for ``command`` (its path, e.g. ``detect energy``).

    Raises:
        click.UsageError: If ``command`` only runs here and ``--remote`` was given.
    """
    global _current, _used
    end()
    if target is not None and command and _is_local(command):
        if target.explicit:
            raise click.UsageError(f"`{command}` runs only on this machine; drop --remote")
        target = None
    _current, _used = target, False
    if target is not None:
        _relax_path_checks()


def end() -> None:
    """Forget this run's remote (and restore local path checks)."""
    global _current
    _current = None
    _restore_path_checks()


def _is_local(command: str) -> bool:
    return any(command == c or command.startswith(c + " ") for c in LOCAL_COMMANDS)


def run(
    target: Target,
    name: str,
    request: dict[str, Any],
    *,
    on_progress: Callable[[int, int], None] | None,
    on_notice: Callable[[str, str], None],
    secrets: dict[str, str] | None,
) -> Any:
    """Run the operation ``name`` on ``target`` and return its result, typed as a local
    run returns it.

    If the result can't be rebuilt here (its type needs an extra this machine
    doesn't have), the server's JSON result is reported in place of the command's own
    output and the command stops there.
    """
    global _used
    from bioamla.client import UntypedResult, connect

    if secrets:
        raise click.UsageError(
            f"Credentials given on the command line ({', '.join(sorted(secrets))}) aren't "
            f"sent to a server; set them in the environment of the server on {target.name!r}"
        )
    if target.workspace is None:
        raise click.UsageError(
            f"Which workspace on {target.name!r}? Use --remote {target.name}:WORKSPACE, or set "
            "`workspace` in the local workspace's config"
        )

    def on_event(event: dict[str, Any]) -> None:
        if event["event"] == "notice":
            on_notice(event["level"], event["message"])

    try:
        client = connect(target.name)
        _used = True
        return client.run(target.workspace, name, _jsonable_request(name, request),
                          on_progress=on_progress, on_event=on_event, typed=True)  # fmt: skip
    except UntypedResult as e:
        _report_untyped(name, e.result)
    except BioamlaError as e:
        error = _as_local_error(e)  # keeps its own __cause__: the local error type
        raise error from error.__cause__


def _jsonable_request(name: str, request: dict[str, Any]) -> dict[str, Any]:
    """``request`` checked as a local run would check it (when this machine can load
    the operation), as JSON."""
    import dataclasses

    from bioamla import operations
    from bioamla.common.jsonable import to_jsonable
    from bioamla.exceptions import DependencyError

    try:
        op = operations.get(name)
    except DependencyError:  # the server may have the extra; let it check
        return to_jsonable(request)
    return to_jsonable(dataclasses.asdict(operations.build_request(op, request)))


def _report_untyped(name: str, result: Any) -> None:
    from bioamla.cli.console import print_warning
    from bioamla.cli.output import emit_result

    if not emit_result(result):
        import json

        print_warning(
            f"{name} ran on the server, but its result can't be shown as usual here "
            "(install the extra its domain needs); the result:"
        )
        click.echo(json.dumps(result, indent=2, default=str))
    raise click.exceptions.Exit(0)


def _as_local_error(error: BioamlaError) -> Exception:
    """The error a local run would have raised for the server's ``error``, so the
    command reports it the same way (exit code, JSON ``error.type``)."""
    from bioamla import exceptions
    from bioamla.exceptions import OptionConflictError, RemoteError

    kind = getattr(error, "error_type", None)
    cls = getattr(exceptions, kind, None) if kind else None
    if isinstance(cls, type) and issubclass(cls, BioamlaError) and not issubclass(cls, RemoteError):
        try:
            local: BioamlaError = cls(str(error))
        except TypeError:
            local = error
    else:
        local = error
    if isinstance(local, OptionConflictError):
        return click.UsageError(str(local))
    exc = click.ClickException(str(local))
    exc.__cause__ = local
    return exc


# Paths on the command line are the server's: don't check them here. ------------

_original_convert: Any = None


def _relax_path_checks() -> None:
    global _original_convert
    if _original_convert is not None:
        return
    original = click.Path.convert

    def convert(self: click.Path, value: Any, param: Any, ctx: Any) -> Any:
        saved = (self.exists, self.resolve_path)
        self.exists, self.resolve_path = False, False
        try:
            return original(self, value, param, ctx)
        finally:
            self.exists, self.resolve_path = saved

    _original_convert = original
    click.Path.convert = convert  # type: ignore[method-assign]


def _restore_path_checks() -> None:
    global _original_convert
    if _original_convert is not None:
        click.Path.convert = _original_convert  # type: ignore[method-assign]
        _original_convert = None


__all__ = ["ENV", "LOCAL_COMMANDS", "Target", "begin", "current", "end", "resolve", "run", "used"]

"""Record each command run inside a workspace in its ``history.jsonl``.

One JSON object per run (format version ``v``)::

    {"v": 1, "time": "2026-10-31T19:40:02+00:00", "duration": 1.84,
     "command": "detect energy", "argv": ["detect", "energy", "rec.wav", "-o", "d.csv"],
     "cwd": "recordings", "host": "laptop", "user": "john", "bioamla": "0.4.0",
     "ok": true, "exit_code": 0, "error": null, "outputs": ["recordings/d.csv"]}

The workspace is the one the command reports working on (``workspace -C PATH``, the
directory ``workspace init`` creates), else the one around the current directory. ``cwd`` and
``outputs`` are relative to the workspace root when inside it.
Values of options whose name suggests a secret (``--api-key``, ``--token``,
``--password``, ...) are replaced with ``***``. ``--help``/``--version`` runs and
``workspace history`` itself aren't recorded. Recording never fails a command: a
problem writing the file is logged as a warning.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("bioamla.cli")

from bioamla.workspace._core import HISTORY_RECORD_VERSION as RECORD_VERSION  # noqa: E402

_NOT_RECORDED = {"workspace history"}
_SECRET = re.compile(r"(key|token|password|passwd|secret)", re.IGNORECASE)


def should_record(group: Any, args: list[str]) -> bool:
    """False for ``--help``/``--version`` runs, bare ``bioamla``, and ``workspace history``."""
    from bioamla.cli.output import command_from_args

    if any(a in ("--help", "--version") for a in args):
        return False
    command = command_from_args(group, args)
    return bool(command) and command not in _NOT_RECORDED


def redact(argv: list[str], group: Any = None) -> list[str]:
    """``argv`` with the values of secret-looking options replaced by ``***``.

    With ``group`` (the root command), only options of the running command that take
    a value are redacted, so a secret-named flag never hides the next argument;
    without it, any secret-named ``--option`` is assumed to take one.
    """
    flags = _flag_names(group, argv) if group is not None else set()
    out: list[str] = []
    hide_next = False
    for arg in argv:
        if hide_next:
            out.append("***")
            hide_next = False
            continue
        if arg.startswith("--") and _SECRET.search(arg):
            name = arg.split("=", 1)[0]
            if "=" in arg:
                out.append(name + "=***")
            else:
                out.append(arg)
                hide_next = name not in flags
            continue
        out.append(arg)
    return out


def _flag_names(group: Any, argv: list[str]) -> set[str]:
    """Long names of the flags (options taking no value) of the command ``argv`` runs."""
    import click

    command = group
    for arg in argv:
        if isinstance(command, click.Group) and arg in command.commands:
            command = command.commands[arg]
    names: set[str] = set()
    for param in getattr(command, "params", []):
        if isinstance(param, click.Option) and (param.is_flag or param.count):
            names.update(o for o in (*param.opts, *param.secondary_opts) if o.startswith("--"))
    return names


def record_run(
    group: Any,
    args: list[str],
    *,
    started: float,
    ended: float,
    ok: bool,
    exit_code: int,
    error: dict[str, Any] | None,
) -> None:
    """Append this run to the history of the workspace around the current directory."""
    import getpass
    import socket

    from bioamla import __version__
    from bioamla.cli.output import command_from_args, last_result, run_workspace
    from bioamla.exceptions import BioamlaError
    from bioamla.workspace import find_workspace

    try:
        # The workspace the command said it worked on (workspace -C, init DIR), else
        # the one around the current directory.
        ws = run_workspace() or find_workspace()
        if ws is None:
            return
        try:
            user = getpass.getuser()
        except (KeyError, OSError):  # no user database entry (some containers)
            user = None
        ws.record_run({
            "v": RECORD_VERSION,
            "time": datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds"),
            "duration": round(ended - started, 3),
            "command": command_from_args(group, args),
            "argv": redact(args, group),
            "cwd": ws.relative(os.getcwd()),
            "host": socket.gethostname(),
            "user": user,
            "bioamla": __version__,
            "ok": ok,
            "exit_code": exit_code,
            "error": error,
            "outputs": ws.output_paths(last_result()),
        })  # fmt: skip
    except (OSError, BioamlaError) as e:
        logger.warning("Couldn't record this run in the workspace history: %s", e)


__all__ = ["RECORD_VERSION", "record_run", "redact", "should_record"]

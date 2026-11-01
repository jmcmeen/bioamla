"""Run bioamla commands for a skill script and read their JSON results.

Every skill carries its own copy of this file (a skill folder is installed on its
own), and the copies must stay identical. Scripts use only this module and the
standard library: they drive bioamla through its command line (`bioamla --json`),
never by importing it, so the same script works against a local install or a remote
server.

Set ``BIOAMLA_BIN`` to the command to run instead of ``bioamla`` (for example a
virtual environment's ``.../bin/bioamla``).

When a remote is set (``BIOAMLA_REMOTE=NAME:WORKSPACE``, or ``remote`` and
``workspace`` in the local workspace's settings), bioamla commands run on that server
and their paths are the server workspace's. A script's own files belong there too:
:func:`save_json` and :func:`publish` upload what a step writes, and :func:`fetch`
brings a server file down to merge into. Without a remote they only work locally.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class BioamlaCommandError(RuntimeError):
    """A bioamla command failed; the message is its error.

    Attributes:
        kind: bioamla's error type (e.g. ``NotFoundError``).
    """

    def __init__(self, message: str, kind: str = "Error"):
        super().__init__(message)
        self.kind = kind


def command() -> list[str]:
    """The command that runs bioamla."""
    return shlex.split(os.environ.get("BIOAMLA_BIN", "bioamla"))


def run(*args: str, cwd: str | Path | None = None) -> Any:
    """Run ``bioamla --json ARGS`` and return its result.

    Raises:
        BioamlaCommandError: If the command fails (with bioamla's own message).
    """
    proc = subprocess.run(
        [*command(), "--json", *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        envelope = json.loads(proc.stdout)
    except ValueError:
        raise BioamlaCommandError(
            f"bioamla {' '.join(map(str, args))} gave no JSON (exit {proc.returncode}): "
            f"{proc.stderr.strip()[-500:]}"
        ) from None
    if not envelope.get("ok"):
        error = envelope.get("error") or {}
        kind = error.get("type", "Error")
        raise BioamlaCommandError(f"{kind}: {error.get('message', '')}", kind)
    return envelope["result"]


def run_dir(skill: str, root: str | Path = ".") -> Path:
    """A new ``runs/<skill>/<UTC timestamp>/`` folder under ``root`` (the workspace)."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = Path(root) / "runs" / skill
    path = base / stamp
    n = 1
    while path.exists():
        n += 1
        path = base / f"{stamp}-{n}"
    path.mkdir(parents=True)
    return path


def write_json(path: str | Path, data: Any) -> Path:
    """Write ``data`` as indented JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return out


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


_remote: dict | None = None


def use_remote(target: str | None) -> None:
    """Run this script's bioamla commands on ``target`` (``NAME:WORKSPACE``), as
    ``BIOAMLA_REMOTE`` would (``None`` leaves the setting as it is)."""
    global _remote
    if target:
        os.environ["BIOAMLA_REMOTE"] = target
        _remote = None


def remote(root: str | Path = ".") -> dict | None:
    """The remote commands run on (``{"remote", "workspace", ...}``), or ``None``."""
    global _remote
    if _remote is None:
        _remote = run("remote", "current", cwd=root)
    return _remote if _remote.get("remote") else None


def fetch(path: str, into: str | Path, root: str | Path = ".") -> Path:
    """The workspace file ``path`` as a local file under ``into`` (same relative path).

    With a remote, the server's copy is downloaded (the local one under ``into`` is
    removed first when the server has none, so it never stands in for it); without one,
    the workspace's own file is copied. The returned file may not exist.
    """
    local = Path(root) / into / path
    target = remote(root)
    if target is None:
        source = Path(root) / path
        if source.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(source.read_bytes())
        return local
    local.unlink(missing_ok=True)
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        run("remote", "download", target["remote"], target["workspace"], path,
            "-o", str(local.parent), "--overwrite", cwd=root)  # fmt: skip
    except BioamlaCommandError as e:
        if e.kind not in ("NotFoundError", "RemoteNotFound"):
            raise
    return local


def publish(local: str | Path, path: str, root: str | Path = ".") -> None:
    """Put the local file ``local`` at ``path`` in the workspace: uploaded to the
    server with a remote (replacing it), else copied into place (unless it is there)."""
    target = remote(root)
    source = Path(local) if Path(local).is_absolute() else Path(root) / local
    if target is None:
        dest = Path(root) / path
        if dest.resolve() != source.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            tmp.write_bytes(source.read_bytes())
            os.replace(tmp, dest)
        return
    if Path(path).name != source.name:
        raise ValueError(f"{source.name} can't be uploaded as {path} (names must match)")
    folder = Path(path).parent.as_posix()
    args = ["--to", folder] if folder not in ("", ".") else []
    run("remote", "upload", target["remote"], target["workspace"], str(source), *args,
        "--overwrite", cwd=root)  # fmt: skip


def save_json(path: str, data: Any, root: str | Path = ".") -> Path:
    """Write ``data`` to the workspace file ``path`` (relative to ``root``), and upload
    it with a remote."""
    out = write_json(Path(root) / path, data)
    if remote(root) is not None:
        publish(out, path, root)
    return out

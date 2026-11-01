"""Run bioamla commands for a skill script and read their JSON results.

Every skill carries its own copy of this file (a skill folder is installed on its
own), and the copies must stay identical. Scripts use only this module and the
standard library: they drive bioamla through its command line (`bioamla --json`),
never by importing it, so the same script works against a local install or a remote
server.

Set ``BIOAMLA_BIN`` to the command to run instead of ``bioamla`` (for example a
virtual environment's ``.../bin/bioamla``).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BioamlaCommandError(RuntimeError):
    """A bioamla command failed; the message is its error."""


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
        raise BioamlaCommandError(f"{error.get('type', 'Error')}: {error.get('message', '')}")
    return envelope["result"]


def run_dir(skill: str, root: str | Path = ".") -> Path:
    """A new ``runs/<skill>/<UTC timestamp>/`` folder under ``root`` (the workspace)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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

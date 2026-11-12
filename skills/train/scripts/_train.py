"""Helpers shared by this skill's scripts (not by other skills: it isn't the identical
``_bioamla.py``)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import posixpath
import shutil
import sys
from pathlib import Path

from _bioamla import BioamlaCommandError, remote, run, run_dir

#: The overall metrics ``bioamla models ast evaluate`` reports, any of which compare.py
#: can decide on.
METRICS = ("accuracy", "precision", "recall", "f1_score")


class Parser(argparse.ArgumentParser):
    """Bad arguments print ``{"error": ...}`` and exit 1, not argparse's 2, which a step
    uses for a refused or discarded run."""

    def error(self, message: str):
        print(json.dumps({"error": f"{self.prog}: {message}"}))
        sys.exit(1)


def fraction(value: str) -> float:
    """An argparse type: a number from 0 to 1."""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not (math.isfinite(number) and 0 <= number <= 1):
        raise argparse.ArgumentTypeError(f"{value!r} is not a number from 0 to 1")
    return number


def positive(value: str) -> int:
    """An argparse type: a whole number of at least 1."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"{value!r} is less than 1")
    return number


def sha256(path: str, root: Path) -> str:
    """The SHA-256 of the workspace file ``path`` (computed on the server with a remote,
    so nothing is downloaded).

    Raises:
        FileNotFoundError: If the workspace has no such file.
    """
    path = _plain(path)
    if remote(root) is not None:
        listing = _server_files(path, root, recursive=False)
        entry = next((f for f in listing["files"] if f["path"] == path), None)
        if entry is None or entry["type"] != "file":
            raise FileNotFoundError(f"{path} is not a file in the workspace")
        if entry.get("sha256") is None:
            raise OSError(f"The server can't read {path}")
        return entry["sha256"]
    local = root / path
    if not local.is_file():
        raise FileNotFoundError(f"{path} is not in the workspace")
    return _file_sha256(local)


def new_run(root: Path) -> Path:
    """A new run folder, relative to ``root``."""
    return run_dir("train", root).relative_to(root)


def abandon(rel: Path, root: Path) -> None:
    """Remove a run folder this step made but didn't get to use (an error before it wrote
    its result), so it doesn't linger as a half-started run."""
    shutil.rmtree(root / rel, ignore_errors=True)


def _file_sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(files: dict) -> str:
    """The SHA-256 of ``path<TAB>sha256`` lines, one per file in path order."""
    lines = "".join(f"{rel}\t{files[rel]}\n" for rel in sorted(files))
    return hashlib.sha256(lines.encode()).hexdigest()


def _plain(path: str) -> str:
    """``path`` as the server writes it: relative to the workspace, normalized."""
    normal = posixpath.normpath(path.replace("\\", "/"))
    if posixpath.isabs(normal) or normal == ".." or normal.startswith("../"):
        raise ValueError(f"{path} is not a path inside the workspace")
    return normal


def _server_files(path: str, root: Path, recursive: bool) -> dict:
    """``bioamla remote files --sha256`` for ``path`` on the remote workspace.

    Raises:
        FileNotFoundError: If the server has no such file or folder.
    """
    target = remote(root)
    args = ["remote", "files", target["remote"], target["workspace"], "--sha256"]
    try:
        return run(*args, *(["--recursive"] if recursive else []), "--", path)
    except BioamlaCommandError as e:
        if e.kind in ("NotFoundError", "RemoteNotFound"):
            raise FileNotFoundError(f"{path} is not in the workspace") from e
        raise


def folder_sha256(path: str, root: Path) -> str | None:
    """A fingerprint of the workspace folder ``path``: the SHA-256 of each file's path
    (relative to the folder) and SHA-256, in path order. With a remote, the server hashes
    its files (``bioamla remote files --sha256``), so nothing is downloaded; the result
    is the same for the same files either way. Unfinished uploads (``*.part`` files) and
    workspace metadata (``.bioamla``) are left out, as the server's listing does. None
    with a remote folder too big for one listing (over 10,000 entries): the set is then
    known by its path alone.

    Raises:
        FileNotFoundError: If the workspace has no such folder.
        OSError: If a file in it can't be read.
    """
    path = _plain(path)
    if remote(root) is not None:
        listing = _server_files(path, root, recursive=True)
        if listing.get("truncated"):
            return None
        files = listing["files"]
        if len(files) == 1 and files[0]["path"] == path and files[0]["type"] == "file":
            raise FileNotFoundError(f"{path} is not a folder in the workspace")
        prefix = path + "/"
        found = {}
        for f in files:
            if f["type"] != "file":
                continue
            if f.get("sha256") is None:
                raise OSError(f"The server can't read {f['path']}")
            found[f["path"][len(prefix) :] if f["path"].startswith(prefix) else f["path"]] = f[
                "sha256"
            ]
        return _fingerprint(found)
    base = root / path
    if not base.is_dir():
        raise FileNotFoundError(f"{path} is not a folder in the workspace")
    return _fingerprint({
        file.relative_to(base).as_posix(): _file_sha256(file)
        for file in base.rglob("*")
        if file.is_file() and not file.name.endswith(".part")
        and ".bioamla" not in file.relative_to(base).parts
    })  # fmt: skip


def read_metrics(path: Path) -> list[dict]:
    """The records of a metrics file (``ast train --metrics-file``), skipping a last line
    cut short by a run that was killed while writing it."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def is_measure(name: str) -> bool:
    """Whether an ``eval_*`` value measures the model (not the evaluation's speed)."""
    return not ("runtime" in name or "per_second" in name)


def lower_is_better(name: str) -> bool:
    return "loss" in name


def summarise(records: list[dict]) -> dict:
    """What a run measured: how it ended, its last and best evaluation metrics (each
    ``eval_*`` value; the best is the lowest for a loss, else the highest), its last
    training loss, the steps it reached and its ``run_id``."""
    status = None
    run_id = None
    steps = 0
    loss = None
    final: dict = {}
    best: dict = {}
    for record in records:
        event = record.get("event")
        if event == "start":
            run_id = (record.get("params") or {}).get("run_id")
        elif event == "end":
            status = record.get("status")
        elif event == "metrics":
            steps = max(steps, record.get("step") or 0)
            metrics = record.get("metrics") or {}
            if isinstance(metrics.get("loss"), (int, float)):
                loss = metrics["loss"]
            evals = {
                k: v for k, v in metrics.items()
                if k.startswith("eval_") and isinstance(v, (int, float)) and is_measure(k)
            }  # fmt: skip
            if evals:
                final = evals
            for name, value in evals.items():
                if name not in best or (
                    value < best[name] if lower_is_better(name) else value > best[name]
                ):
                    best[name] = value
    return {"status": status, "run_id": run_id, "steps": steps, "train_loss": loss,
            "final_eval": final, "best_eval": best}  # fmt: skip

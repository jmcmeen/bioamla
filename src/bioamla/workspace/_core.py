"""The ``.bioamla/`` workspace format.

A workspace is a project directory with a ``.bioamla/`` folder inside::

    my-survey/
    ├── .bioamla/
    │   ├── workspace.toml      # format version, id, name, created
    │   ├── config.toml         # settings for this workspace
    │   ├── artifacts.json      # linked and imported recordings, datasets, models
    │   ├── annotations/
    │   ├── imported/
    │   ├── datasets/
    │   ├── models/
    │   ├── runs/
    │   └── history.jsonl
    └── ...                     # the project's own files

An :class:`Artifact` is anything the workspace points at. It is either *linked*
(referenced where it is, never copied, so several workspaces can share it) or
*imported* (a copy under ``.bioamla/imported/``). A linked path inside the
workspace is stored relative to the workspace root, so the workspace can move;
one outside it is stored absolute (symlinks kept as given). Imported paths are
relative to ``imported/``.

The format is adapted from MagPy's workspace bundle so both open the same
workspaces. This module uses only the standard library.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from bioamla.common.constants import SUPPORTED_AUDIO_EXTENSIONS
from bioamla.common.files import require_exists, write_text_atomic
from bioamla.exceptions import InvalidInputError, NotFoundError, WorkspaceError

#: Name of the folder that marks a workspace.
WORKSPACE_DIR = ".bioamla"
#: Version of the on-disk format this code writes and reads.
FORMAT_VERSION = 1
#: Version of the run records in ``history.jsonl`` (their ``v`` field).
HISTORY_RECORD_VERSION = 1
#: Commands that only read a workspace, so aren't recorded in its history (a client
#: polling them would otherwise bury the real runs). Operation names use dots.
UNRECORDED_COMMANDS = frozenset({
    "workspace history",
    "workspace status",
    "workspace list",
    # Reads of another machine's state, not work in this workspace:
    "remote list",
    "remote ping",
    "remote workspaces",
    "remote jobs",
})  # fmt: skip

ArtifactKind = Literal["audio_file", "folder", "dataset", "model", "file"]
ArtifactMode = Literal["linked", "imported"]
ARTIFACT_KINDS: tuple[str, ...] = ("audio_file", "folder", "dataset", "model", "file")
ARTIFACT_MODES: tuple[str, ...] = ("linked", "imported")

_SUBDIRS = ("annotations", "imported", "datasets", "models", "runs")


def _new_id() -> str:
    return uuid4().hex[:12]


@dataclass
class Artifact:
    """One thing a workspace points at.

    Attributes:
        kind: ``audio_file``, ``folder``, ``dataset``, ``model`` or ``file``.
        path: Where it is: relative to the workspace root (linked, inside the
            workspace), absolute (linked, outside it), or relative to
            ``.bioamla/imported/`` (imported). Always ``/``-separated when relative.
        mode: ``linked`` or ``imported``.
        id: Stable identifier, unique within the workspace.
        name: Optional display name.
    """

    kind: str
    path: str
    mode: str = "linked"
    id: str = field(default_factory=_new_id)
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_write_atomic = write_text_atomic


def _toml_scalars(values: dict[str, Any]) -> str:
    """A flat TOML document of strings, ints and bools (enough for workspace.toml)."""
    lines = []
    for key, value in values.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, int):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f"{key} = {json.dumps(str(value), ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def _guess_kind(path: Path) -> str:
    if path.is_dir():
        return "folder"
    if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
        return "audio_file"
    return "file"


#: Result keys that name a file a run wrote (see :meth:`Workspace.output_paths`).
OUTPUT_KEYS = ("output", "output_path", "output_dir", "metadata_file", "results_file")


class Workspace:
    """An open workspace: its root directory and the contents of ``.bioamla/``.

    Use :func:`create_workspace`, :func:`open_workspace` or :func:`find_workspace`
    rather than constructing one directly.

    Attributes:
        root: The workspace's project directory (the parent of ``.bioamla/``).
        id: Stable identifier.
        name: Display name.
        created: ISO 8601 creation time (UTC).
        format_version: The on-disk format version.
    """

    def __init__(self, root: Path, meta: dict[str, Any]) -> None:
        self.root = root
        self.id = str(meta["id"])
        self.name = str(meta.get("name") or root.name)
        self.created = str(meta.get("created", ""))
        self.format_version = int(meta["format_version"])

    def __repr__(self) -> str:
        return f"Workspace({self.name!r}, root={str(self.root)!r})"

    # -- layout ------------------------------------------------------------

    @property
    def dir(self) -> Path:
        """The ``.bioamla/`` folder."""
        return self.root / WORKSPACE_DIR

    @property
    def config_path(self) -> Path:
        return self.dir / "config.toml"

    @property
    def history_path(self) -> Path:
        """``history.jsonl``; created by the first run recorded in the workspace."""
        return self.dir / "history.jsonl"

    @property
    def annotations_dir(self) -> Path:
        return self.dir / "annotations"

    @property
    def imported_dir(self) -> Path:
        return self.dir / "imported"

    @property
    def datasets_dir(self) -> Path:
        return self.dir / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.dir / "models"

    @property
    def runs_dir(self) -> Path:
        return self.dir / "runs"

    @property
    def _artifacts_path(self) -> Path:
        return self.dir / "artifacts.json"

    # -- artifacts ---------------------------------------------------------

    def artifacts(self, kind: str | None = None) -> list[Artifact]:
        """The workspace's artifacts, optionally only those of one ``kind``."""
        items = self._load_artifacts()
        return [a for a in items if kind is None or a.kind == kind]

    def _load_artifacts(self) -> list[Artifact]:
        if not self._artifacts_path.exists():
            return []
        try:
            raw = json.loads(self._artifacts_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("artifacts", []), list):
                raise ValueError("expected an object with an 'artifacts' list")
            items = []
            for entry in raw.get("artifacts", []):
                if not isinstance(entry, dict):
                    raise ValueError(f"artifact entry is not an object: {entry!r}")
                fields = {k: entry[k] for k in entry if k in Artifact.__dataclass_fields__}
                artifact = Artifact(**fields)
                if artifact.mode not in ARTIFACT_MODES:
                    raise ValueError(f"artifact {artifact.id!r} has unknown mode {artifact.mode!r}")
                items.append(artifact)
        except (OSError, ValueError, TypeError) as e:
            raise WorkspaceError(f"Unreadable {self._artifacts_path}: {e}") from e
        return items

    def _save_artifacts(self, artifacts: list[Artifact]) -> None:
        data = {"artifacts": [a.to_dict() for a in artifacts]}
        _write_atomic(self._artifacts_path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the workspace's exclusive lock while changing its files (the artifact
        list, the run history), so two processes (bioamla and MagPy, say) never lose
        or interleave each other's updates."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / ".artifacts.lock", "a+") as handle:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def get_artifact(self, artifact_id: str) -> Artifact:
        """The artifact with ``artifact_id``.

        Raises:
            NotFoundError: If there is none.
        """
        for artifact in self.artifacts():
            if artifact.id == artifact_id:
                return artifact
        raise NotFoundError(f"No artifact {artifact_id!r} in workspace {self.name!r}")

    def relative(self, path: str | Path) -> str:
        """``path`` relative to the workspace root (``/``-separated) if it's inside the
        workspace, also when one of them is reached through a symlink (e.g. /var vs
        /private/var on macOS); otherwise absolute, keeping symlinks as given."""
        return self._stored_link_path(Path(path))

    def _stored_link_path(self, path: Path) -> str:
        absolute = Path(os.path.abspath(path))
        for candidate, root in (
            (absolute, Path(os.path.abspath(self.root))),
            (Path(os.path.realpath(path)), Path(os.path.realpath(self.root))),
        ):
            try:
                return candidate.relative_to(root).as_posix()
            except ValueError:
                continue
        return str(absolute)

    def link(self, path: str | Path, kind: str | None = None, name: str = "") -> Artifact:
        """Point the workspace at ``path`` without copying it.

        Linking a path that is already linked returns the existing artifact.

        Raises:
            NotFoundError: If ``path`` does not exist.
            InvalidInputError: If ``kind`` is not a known kind.
        """
        return self.link_many([path], kind=kind, name=name)[0]

    def link_many(
        self, paths: Iterable[str | Path], kind: str | None = None, name: str = ""
    ) -> list[Artifact]:
        """:meth:`link` for several paths at once: all are checked first, then
        recorded in one write, so either every path is linked or none is.

        Raises:
            NotFoundError: If any path does not exist (nothing is linked).
            InvalidInputError: If ``kind`` is not a known kind.
        """
        checked = [require_exists(p) for p in paths]
        kinds = [self._check_kind(kind or _guess_kind(p)) for p in checked]
        stored = [self._stored_link_path(p) for p in checked]
        result: list[Artifact] = []
        with self._locked():
            artifacts = self._load_artifacts()
            by_path = {a.path: a for a in artifacts if a.mode == "linked"}
            for path_kind, path_stored in zip(kinds, stored, strict=True):
                existing = by_path.get(path_stored)
                if existing is None:
                    existing = Artifact(kind=path_kind, path=path_stored, mode="linked", name=name)
                    artifacts.append(existing)
                    by_path[path_stored] = existing
                result.append(existing)
            self._save_artifacts(artifacts)
        return result

    def import_copy(self, path: str | Path, kind: str | None = None, name: str = "") -> Artifact:
        """Copy ``path`` (a file or folder) into ``.bioamla/imported/`` and record it.

        A name already taken in ``imported/`` gets a short id prefix rather than
        overwriting.

        Raises:
            NotFoundError: If ``path`` does not exist.
            InvalidInputError: If ``kind`` is not a known kind, or ``path`` contains
                the workspace's ``.bioamla/`` folder (it would copy into itself).
        """
        return self.import_many([path], kind=kind, name=name)[0]

    def import_many(
        self, paths: Iterable[str | Path], kind: str | None = None, name: str = ""
    ) -> list[Artifact]:
        """:meth:`import_copy` for several paths: all are checked first, then copied
        and recorded in one write. If anything fails, the copies made are removed and
        nothing is recorded.

        Raises:
            NotFoundError: If any path does not exist.
            InvalidInputError: If ``kind`` is not a known kind, or a path contains
                this workspace.
        """
        checked = [require_exists(p) for p in paths]
        kinds = [self._check_kind(kind or _guess_kind(p)) for p in checked]
        inside = Path(os.path.abspath(self.dir))
        for path in checked:
            if path.is_dir() and inside.is_relative_to(Path(os.path.abspath(path))):
                raise InvalidInputError(f"Can't import {path}: it contains this workspace")
        copies: list[Path] = []
        result: list[Artifact] = []
        with self._locked():
            artifacts = self._load_artifacts()  # fail before copying anything
            self.imported_dir.mkdir(parents=True, exist_ok=True)
            try:
                for path, path_kind in zip(checked, kinds, strict=True):
                    dest = self.imported_dir / path.name
                    if dest.exists():
                        dest = self.imported_dir / f"{_new_id()[:8]}_{path.name}"
                    copies.append(dest)
                    if path.is_dir():
                        shutil.copytree(path, dest)
                    else:
                        shutil.copy2(path, dest)
                    result.append(Artifact(
                        kind=path_kind, path=dest.relative_to(self.imported_dir).as_posix(),
                        mode="imported", name=name,
                    ))  # fmt: skip
                self._save_artifacts([*artifacts, *result])
            except BaseException:
                for dest in copies:
                    if dest.is_dir():
                        shutil.rmtree(dest, ignore_errors=True)
                    else:
                        dest.unlink(missing_ok=True)
                raise
        return result

    def remove(self, artifact_id: str, delete_import: bool = False) -> Artifact:
        """Forget an artifact. Linked files are never touched; an imported copy is
        deleted only with ``delete_import=True``.

        Raises:
            NotFoundError: If there is no such artifact.
            WorkspaceError: If an imported artifact's path points outside ``imported/``.
        """
        with self._locked():
            artifacts = self._load_artifacts()
            matches = [a for a in artifacts if a.id == artifact_id]
            if not matches:
                raise NotFoundError(f"No artifact {artifact_id!r} in workspace {self.name!r}")
            artifact = matches[0]
            target = (
                self.resolve(artifact) if delete_import and artifact.mode == "imported" else None
            )
            self._save_artifacts([a for a in artifacts if a.id != artifact_id])
        if target is not None:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
        return artifact

    def resolve(self, artifact: Artifact) -> Path:
        """The artifact's absolute path on this machine.

        Raises:
            WorkspaceError: If an imported artifact's path is empty, absolute, or
                leaves ``imported/`` (``..``); the artifact list is then corrupt or
                hand-edited, and nothing should act on it.
        """
        if artifact.mode == "imported":
            rel = Path(artifact.path)
            if not artifact.path or rel.is_absolute() or ".." in rel.parts:
                raise WorkspaceError(
                    f"Imported artifact {artifact.id!r} has a path outside imported/: "
                    f"{artifact.path!r}"
                )
            return self.imported_dir / rel
        path = Path(artifact.path)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _check_kind(kind: str) -> str:
        if kind not in ARTIFACT_KINDS:
            raise InvalidInputError(
                f"Unknown artifact kind {kind!r}; expected one of {', '.join(ARTIFACT_KINDS)}"
            )
        return kind

    # -- run history ---------------------------------------------------------

    def record_run(self, record: dict[str, Any]) -> None:
        """Append one run record to ``history.jsonl`` (one JSON object per line).

        Appends happen under the workspace lock, so concurrent runs never interleave.
        """
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._locked():
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(line)

    def run_record(
        self,
        *,
        command: str,
        started: float,
        ended: float,
        ok: bool,
        exit_code: int,
        error: dict[str, Any] | None,
        result: Any = None,
        user: str | None = None,
        argv: list[str] | None = None,
        cwd: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """A run record for :meth:`record_run`, in the current format.

        Args:
            command: The command (``"detect energy"``).
            started: When it started, and ``ended`` when it ended (epoch seconds).
            ok: Whether it succeeded, with its ``exit_code`` and ``error``
                (``{"type", "message"}``).
            result: Its result as JSON-ready data; the files it names become
                ``outputs`` (see :meth:`output_paths`).
            user: Who ran it.
            argv: Its command line (``None`` when it didn't come from one).
            cwd: Where it ran, relative to the workspace.
            extra: More fields (e.g. a server run's ``operation``, ``request``,
                ``via`` and ``job``).
        """
        import socket
        from datetime import datetime, timezone

        from bioamla import __version__

        return {
            "v": HISTORY_RECORD_VERSION,
            "time": datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds"),
            "duration": round(ended - started, 3),
            "command": command,
            "argv": argv,
            "cwd": cwd,
            "host": socket.gethostname(),
            "user": user,
            "bioamla": __version__,
            "ok": ok,
            "exit_code": exit_code,
            "error": error,
            "outputs": self.output_paths(result),
            **extra,
        }

    def history(self, limit: int | None = None, failed_only: bool = False) -> list[dict]:
        """Run records, oldest first; the last ``limit`` of them if given.

        Lines that aren't valid JSON objects (a crash mid-write, a hand edit) are
        skipped rather than failing.
        """
        if not self.history_path.exists():
            return []
        from collections import deque

        # Keep only what's needed: the last `limit` records unless filtering first.
        records: Any = deque(maxlen=limit) if limit and not failed_only else []
        with open(self.history_path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and (not failed_only or not record.get("ok", True)):
                    records.append(record)
        records = list(records)
        return records[-limit:] if limit else records

    def output_paths(self, result: Any) -> list[str]:
        """The files a run's result says it wrote (its ``output``, ``output_dir``,
        ``metadata_file``, ``results_file``, ``output_path`` and ``output_files``),
        relative to the workspace where they're inside it."""
        if not isinstance(result, dict):
            return []
        paths: list[str] = []
        for key in OUTPUT_KEYS:
            if isinstance(result.get(key), (str, os.PathLike)) and str(result[key]):
                paths.append(str(result[key]))
        files = result.get("output_files")
        if isinstance(files, (list, tuple)):
            paths += [str(f) for f in files if isinstance(f, (str, os.PathLike))]
        return [self.relative(p) for p in dict.fromkeys(paths)]

    def artifact_row(self, artifact: Artifact) -> dict[str, Any]:
        """``artifact`` as a dict, with where it resolves to and whether it's there
        (``resolved``, ``exists``)."""
        path = self.resolve(artifact)
        return {**artifact.to_dict(), "resolved": str(path), "exists": path.exists()}

    def artifact_rows(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Every artifact as :meth:`artifact_row`; only those of ``kind`` if given."""
        return [self.artifact_row(a) for a in self.artifacts(kind=kind)]

    def status(self, last_runs: int = 5) -> dict[str, Any]:
        """:meth:`to_dict` plus the last ``last_runs`` run records (``last_runs``)."""
        return {**self.to_dict(), "last_runs": self.history(limit=last_runs)}

    def settings(self) -> dict[str, Any]:
        """This workspace's own settings (``config.toml``) as dotted keys, e.g.
        ``{"detect.energy.threshold_db": -25}``.

        Raises:
            ConfigError: If the file can't be read.
        """
        from bioamla.common.settings import flatten, read_settings_file

        return flatten(read_settings_file(self.config_path))

    def set_setting(self, key: str, value: Any) -> None:
        """Set the dotted ``key`` to ``value`` in this workspace's ``config.toml``.

        The value is stored as given (no conversion to an option's type; the CLI's
        ``workspace config set`` does that). Another spelling of the same key (``-``
        for ``_``) is replaced. Comments in the file are not kept.

        Raises:
            ConfigError: If the file can't be read, or ``key`` or ``value`` can't be
                stored (e.g. a key that is also a table).
        """
        from bioamla.common.settings import (
            dump_toml,
            flatten,
            read_settings_file,
            set_dotted,
            unset_dotted,
        )

        with self._locked():
            settings = read_settings_file(self.config_path)
            for name in list(flatten(settings)):  # one spelling per key in the file
                if name != key and _same_setting(name, key):
                    unset_dotted(settings, name)
            set_dotted(settings, key, value)
            write_text_atomic(self.config_path, dump_toml(settings))

    def unset_setting(self, key: str) -> bool:
        """Remove the dotted ``key`` (in any spelling: ``-`` or ``_``) from this
        workspace's ``config.toml``.

        Returns:
            Whether it was set.
        """
        from bioamla.common.settings import dump_toml, flatten, read_settings_file, unset_dotted

        with self._locked():
            settings = read_settings_file(self.config_path)
            removed = False
            for name in list(flatten(settings)):
                if _same_setting(name, key):
                    removed = unset_dotted(settings, name) or removed
            if removed:
                write_text_atomic(self.config_path, dump_toml(settings))
        return removed

    def list_files(
        self, path: str | Path = ".", recursive: bool = False, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """The files and folders in ``path`` (relative to the workspace, or absolute),
        sorted by folder. Workspace metadata folders (``.bioamla``) and unfinished
        uploads (``*.part``) are left out; links to folders are listed but not followed,
        and folders that can't be read are skipped.

        Args:
            path: The folder.
            recursive: Include subfolders.
            limit: Stop after this many entries.

        Returns:
            One ``{"path", "type", "size", "modified"}`` per entry: ``path`` relative
            to the workspace where it's inside it, ``type`` ``"file"``, ``"dir"`` or
            ``"link"`` (a link to a folder), ``size`` in bytes for files and
            ``modified`` as epoch seconds.

        Raises:
            NotFoundError: If ``path`` isn't a folder.
        """
        start = Path(path)
        if not start.is_absolute():
            start = self.root / start
        if not start.is_dir():
            raise NotFoundError(f"No folder {path}")
        entries: list[dict[str, Any]] = []
        folders = [start]
        while folders and (limit is None or len(entries) < limit):
            folder = folders.pop(0)
            try:
                with os.scandir(folder) as it:
                    children = sorted(it, key=lambda e: e.name)
            except OSError:
                continue
            subfolders = []
            for entry in children:
                if entry.name == WORKSPACE_DIR or entry.name.endswith(".part"):
                    continue
                try:
                    is_dir = entry.is_dir()  # follows links
                    link = entry.is_symlink()
                    stat = entry.stat()
                except OSError:
                    continue  # a broken link
                kind = "link" if link and is_dir else "dir" if is_dir else "file"
                entries.append({
                    "path": self.relative(entry.path),
                    "type": kind,
                    "size": stat.st_size if kind == "file" else None,
                    "modified": stat.st_mtime,
                })  # fmt: skip
                if limit is not None and len(entries) >= limit:
                    break
                if recursive and kind == "dir":
                    subfolders.append(Path(entry.path))
            folders[:0] = subfolders
        return entries

    def to_dict(self) -> dict[str, Any]:
        """Workspace metadata and a count of artifacts per kind."""
        counts: dict[str, int] = {}
        for artifact in self.artifacts():
            counts[artifact.kind] = counts.get(artifact.kind, 0) + 1
        return {
            "root": str(self.root),
            "id": self.id,
            "name": self.name,
            "created": self.created,
            "format_version": self.format_version,
            "artifacts": counts,
        }


def _same_setting(a: str, b: str) -> bool:
    """Whether two setting keys name the same thing (``-`` and ``_`` are alike, as in
    option names)."""
    return a.replace("-", "_") == b.replace("-", "_")


def is_workspace(path: str | Path) -> bool:
    """True if ``path`` is a directory with a ``.bioamla/workspace.toml``."""
    return (Path(path) / WORKSPACE_DIR / "workspace.toml").is_file()


def create_workspace(root: str | Path, name: str | None = None) -> Workspace:
    """Make ``root`` a workspace (creating the directory if needed).

    Raises:
        WorkspaceError: If ``root`` is already a workspace.
    """
    root = Path(os.path.abspath(root))
    if is_workspace(root):
        raise WorkspaceError(f"{root} is already a workspace")
    meta = {
        "format_version": FORMAT_VERSION,
        "id": _new_id(),
        "name": name or root.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    dot = root / WORKSPACE_DIR
    for sub in _SUBDIRS:
        (dot / sub).mkdir(parents=True, exist_ok=True)
    _write_atomic(dot / "workspace.toml", _toml_scalars(meta))
    _write_atomic(dot / "artifacts.json", json.dumps({"artifacts": []}, indent=2) + "\n")
    _write_atomic(dot / "config.toml", "# Settings for this workspace.\n")
    return Workspace(root, meta)


def open_workspace(root: str | Path) -> Workspace:
    """Open the workspace at ``root``.

    Raises:
        NotFoundError: If ``root`` is not a workspace.
        WorkspaceError: If its metadata is unreadable, or it was written by a newer
            bioamla (a higher format version).
    """
    root = Path(os.path.abspath(root))
    meta_path = root / WORKSPACE_DIR / "workspace.toml"
    if not meta_path.is_file():
        raise NotFoundError(f"Not a bioamla workspace: {root} (no {WORKSPACE_DIR}/workspace.toml)")
    from bioamla.common.config import load_toml
    from bioamla.exceptions import ConfigError

    try:
        meta = load_toml(meta_path)
        version = int(meta["format_version"])
        meta["id"]
    except ConfigError as e:
        if "tomli" in str(e):  # a missing dependency, not a broken workspace
            raise
        raise WorkspaceError(f"Unreadable {meta_path}: {e}") from e
    except (KeyError, ValueError, TypeError) as e:
        raise WorkspaceError(f"Unreadable {meta_path}: {e}") from e
    if version > FORMAT_VERSION:
        raise WorkspaceError(
            f"{root} uses workspace format {version}, newer than this bioamla supports "
            f"({FORMAT_VERSION}); upgrade bioamla to open it"
        )
    return Workspace(root, meta)


def current_workspace(start: str | Path | None = None) -> Workspace:
    """The workspace containing ``start`` (default: the current directory).

    Raises:
        NotFoundError: If there is none.
    """
    ws = find_workspace(start)
    if ws is None:
        where = start or "the current directory"
        raise NotFoundError(
            f"Not inside a bioamla workspace ({where}). "
            "Create one with `bioamla workspace init` (create_workspace())."
        )
    return ws


def find_workspace(start: str | Path | None = None) -> Workspace | None:
    """The workspace containing ``start`` (default: the current directory), found by
    walking up the directory tree; ``None`` if there is none."""
    here = Path(os.path.abspath(start if start is not None else os.getcwd()))
    for candidate in (here, *here.parents):
        if is_workspace(candidate):
            return open_workspace(candidate)
    return None

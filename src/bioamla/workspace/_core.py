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
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from bioamla.common.constants import SUPPORTED_AUDIO_EXTENSIONS
from bioamla.common.files import require_exists
from bioamla.exceptions import InvalidInputError, NotFoundError, WorkspaceError

#: Name of the folder that marks a workspace.
WORKSPACE_DIR = ".bioamla"
#: Version of the on-disk format this code writes and reads.
FORMAT_VERSION = 1

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


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file and rename, so readers never see half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


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
        """Hold an exclusive lock on the workspace's artifact list while changing it,
        so two processes (bioamla and MagPy, say) never lose each other's updates."""
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

    def _stored_link_path(self, path: Path) -> str:
        absolute = Path(os.path.abspath(path))
        try:
            return absolute.relative_to(os.path.abspath(self.root)).as_posix()
        except ValueError:
            return str(absolute)

    def link(self, path: str | Path, kind: str | None = None, name: str = "") -> Artifact:
        """Point the workspace at ``path`` without copying it.

        Linking a path that is already linked returns the existing artifact.

        Raises:
            NotFoundError: If ``path`` does not exist.
            InvalidInputError: If ``kind`` is not a known kind.
        """
        path = require_exists(path)
        kind = self._check_kind(kind or _guess_kind(path))
        stored = self._stored_link_path(path)
        with self._locked():
            artifacts = self._load_artifacts()
            for existing in artifacts:
                if existing.mode == "linked" and existing.path == stored:
                    return existing
            artifact = Artifact(kind=kind, path=stored, mode="linked", name=name)
            self._save_artifacts([*artifacts, artifact])
        return artifact

    def import_copy(self, path: str | Path, kind: str | None = None, name: str = "") -> Artifact:
        """Copy ``path`` (a file or folder) into ``.bioamla/imported/`` and record it.

        A name already taken in ``imported/`` gets a short id prefix rather than
        overwriting.

        Raises:
            NotFoundError: If ``path`` does not exist.
            InvalidInputError: If ``kind`` is not a known kind, or ``path`` contains
                the workspace's ``.bioamla/`` folder (it would copy into itself).
        """
        path = require_exists(path)
        kind = self._check_kind(kind or _guess_kind(path))
        source = Path(os.path.abspath(path))
        if path.is_dir() and Path(os.path.abspath(self.dir)).is_relative_to(source):
            raise InvalidInputError(f"Can't import {path}: it contains this workspace")
        with self._locked():
            artifacts = self._load_artifacts()  # fail before copying anything
            self.imported_dir.mkdir(parents=True, exist_ok=True)
            dest = self.imported_dir / path.name
            if dest.exists():
                dest = self.imported_dir / f"{_new_id()[:8]}_{path.name}"
            if path.is_dir():
                shutil.copytree(path, dest)
            else:
                shutil.copy2(path, dest)
            artifact = Artifact(
                kind=kind, path=dest.relative_to(self.imported_dir).as_posix(), mode="imported",
                name=name,
            )  # fmt: skip
            try:
                self._save_artifacts([*artifacts, artifact])
            except BaseException:
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink(missing_ok=True)
                raise
        return artifact

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


def find_workspace(start: str | Path | None = None) -> Workspace | None:
    """The workspace containing ``start`` (default: the current directory), found by
    walking up the directory tree; ``None`` if there is none."""
    here = Path(os.path.abspath(start if start is not None else os.getcwd()))
    for candidate in (here, *here.parents):
        if is_workspace(candidate):
            return open_workspace(candidate)
    return None

"""Operations for ``workspace``: thin adapters over :mod:`bioamla.workspace`.

See :mod:`bioamla.operations`. ``directory`` is where to look for the workspace (it is
found by walking up, like git); ``init`` makes it one. When the run already has a
workspace (a server run in one), that is the workspace, and ``directory`` must be
inside it. The settings commands
(``workspace config``) have no operations: settings are defaults for command-line
options, so they mean nothing to a request, which always says what it wants.
"""

from __future__ import annotations

from typing import Any, Literal

from bioamla.common.config import config_dataclass
from bioamla.operations import OperationContext, PathStr, operation


def _workspace(directory: str, context: OperationContext) -> Any:
    """The workspace ``directory`` is in: the run's own when it has one (``directory``
    must then be inside it), else found by walking up from ``directory``."""
    from pathlib import Path

    from bioamla.workspace import current_workspace

    ws = context.workspace
    if ws is None:
        return current_workspace(directory)
    from bioamla.exceptions import InvalidInputError

    root = Path(ws.root).resolve()
    where = Path(directory).resolve()
    if where != root and not where.is_relative_to(root):
        raise InvalidInputError(f"{directory} is not inside the workspace {ws.name!r}")
    return ws


ArtifactKind = Literal["audio_file", "folder", "dataset", "model", "file"]


@config_dataclass
class InitRequest:
    """Make a folder a workspace.

    Attributes:
        directory: The folder.
        name: Workspace name (default: the folder's name).
    """

    directory: PathStr = "."
    name: str | None = None


@operation("workspace.init", request=InitRequest)
def init_op(request: InitRequest, context: OperationContext) -> dict[str, Any]:
    """Make a folder a workspace."""
    from bioamla.workspace import create_workspace

    return create_workspace(request.directory, name=request.name).to_dict()


@config_dataclass
class WorkspaceRequest:
    """A workspace, found from a folder.

    Attributes:
        directory: Where to start looking.
    """

    directory: PathStr = "."


@operation("workspace.status", request=WorkspaceRequest)
def status_op(request: WorkspaceRequest, context: OperationContext) -> dict[str, Any]:
    """A workspace's name, location, artifact counts and last runs."""
    return _workspace(request.directory, context).status()


@config_dataclass
class AddRequest:
    """Add files or folders to a workspace.

    Attributes:
        directory: Where to start looking for the workspace.
        paths: The files or folders.
        kind: Artifact kind (default: guessed).
        name: Display name (only with a single path).
    """

    paths: list[PathStr]
    directory: PathStr = "."
    kind: ArtifactKind | None = None
    name: str | None = None


def _add(request: AddRequest, context: OperationContext, copy: bool) -> dict[str, Any]:
    from bioamla.exceptions import OptionConflictError

    if request.name is not None and len(request.paths) > 1:
        raise OptionConflictError("--name applies to a single PATH")
    ws = _workspace(request.directory, context)
    many = ws.import_many if copy else ws.link_many
    added = many(list(request.paths), kind=request.kind, name=request.name or "")
    return {"workspace": str(ws.root), "artifacts": [ws.artifact_row(a) for a in added]}


@operation("workspace.add", request=AddRequest)
def add_op(request: AddRequest, context: OperationContext) -> dict[str, Any]:
    """Link files or folders into a workspace, leaving them where they are."""
    return _add(request, context, copy=False)


@operation("workspace.import", request=AddRequest, kind="job")
def import_op(request: AddRequest, context: OperationContext) -> dict[str, Any]:
    """Copy files or folders into a workspace."""
    return _add(request, context, copy=True)


@config_dataclass
class ListRequest:
    """A workspace's artifacts.

    Attributes:
        directory: Where to start looking for the workspace.
        kind: Only this kind.
    """

    directory: PathStr = "."
    kind: ArtifactKind | None = None


@operation("workspace.list", request=ListRequest)
def list_op(request: ListRequest, context: OperationContext) -> dict[str, Any]:
    """List a workspace's artifacts and whether each is still there."""
    ws = _workspace(request.directory, context)
    return {"workspace": str(ws.root), "artifacts": ws.artifact_rows(kind=request.kind)}


@config_dataclass
class RemoveRequest:
    """Forget an artifact.

    Attributes:
        artifact_id: The artifact.
        directory: Where to start looking for the workspace.
        delete_import: Also delete an imported artifact's copy.
    """

    artifact_id: str
    directory: PathStr = "."
    delete_import: bool = False


@operation("workspace.remove", request=RemoveRequest)
def remove_op(request: RemoveRequest, context: OperationContext) -> dict[str, Any]:
    """Forget an artifact (linked files are never touched)."""
    ws = _workspace(request.directory, context)
    artifact = ws.remove(request.artifact_id, delete_import=request.delete_import)
    return {
        "workspace": str(ws.root),
        "removed": artifact.to_dict(),
        "deleted_copy": request.delete_import and artifact.mode == "imported",
    }


@config_dataclass
class HistoryRequest:
    """A workspace's run history.

    Attributes:
        directory: Where to start looking for the workspace.
        limit: The last N runs (``None``: all).
        failed: Only runs that failed.
    """

    directory: PathStr = "."
    limit: int | None = 20
    failed: bool = False


@operation("workspace.history", request=HistoryRequest)
def history_op(request: HistoryRequest, context: OperationContext) -> dict[str, Any]:
    """The commands run in a workspace."""
    ws = _workspace(request.directory, context)
    return {
        "workspace": str(ws.root),
        "runs": ws.history(limit=request.limit, failed_only=request.failed),
    }

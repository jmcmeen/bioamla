"""Workspace and file routes: the server's workspaces, their settings and their files.

- ``GET /v1/workspaces``: the workspaces under the server root
- ``POST /v1/workspaces``: create one (``{"name", "title"}``)
- ``GET /v1/workspaces/{ws}``: its status (metadata, artifact counts, last runs)
- ``GET /v1/workspaces/{ws}/config``: its settings (``.bioamla/config.toml``)
- ``PUT|DELETE /v1/workspaces/{ws}/config/{key}``: set (``{"value"}``) or remove one
- ``GET /v1/workspaces/{ws}/files?path=&recursive=``: list a folder
- ``GET /v1/workspaces/{ws}/files/{path}``: download a file
- ``PUT /v1/workspaces/{ws}/files/{path}``: upload a file (the request body, as is;
  ``?overwrite=true`` to replace one)

Every path goes through the server's path policy, so it must resolve inside the
workspace root or a linkable folder, and never to a workspace's metadata. Uploads go
only into the workspace named in the URL. To use recordings in a linkable folder without copying them, link
them with the ``workspace.add`` operation.

The server only writes a workspace's own settings, never the server user's file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from bioamla.exceptions import ConfigError, InvalidInputError, NotFoundError, WorkspaceError
from bioamla.server._security import AccessDenied

#: Most entries a file listing returns (``truncated`` says when there were more).
LIST_LIMIT = 10_000


def add_workspace_routes(app: FastAPI, api_version: str, envelope: Any) -> None:
    """Add the workspace and file routes to ``app``."""
    from bioamla.workspace import create_workspace, is_workspace, open_workspace

    identity = app.state.identity
    policy = app.state.policy
    v = api_version

    def workspace(who: Any, ws: str) -> Any:
        return open_workspace(policy.workspace_dir(who, ws))

    def in_workspace(who: Any, ws: str, path: str) -> tuple[Any, Path]:
        """The workspace and ``path`` in it, checked against the policy."""
        space = workspace(who, ws)
        return space, policy.authorize(who, path, base=space.root)

    @app.get(f"/{v}/workspaces", tags=["workspaces"])
    def list_workspaces(who: Any = Depends(identity)) -> dict[str, Any]:
        """The workspaces under the server root: name (its folder), title, id, created
        and artifact counts."""
        found = []
        for child in sorted(policy.root.iterdir()):
            if child.name.startswith(".") or not is_workspace(child):
                continue
            try:
                space = open_workspace(policy.workspace_dir(who, child.name))
            except Exception:  # noqa: BLE001 - one bad workspace doesn't hide the rest
                continue
            found.append({**space.to_dict(), "workspace": child.name})
        return envelope(found)

    @app.post(f"/{v}/workspaces", tags=["workspaces"], status_code=201)
    def create(
        name: str = Body(..., embed=True, description="Its folder name under the root"),
        title: str | None = Body(None, embed=True, description="Its display name"),
        who: Any = Depends(identity),
    ) -> dict[str, Any]:
        """Create a workspace under the server root."""
        try:
            path = policy.new_workspace_dir(who, name)
            path.mkdir()  # exclusively: of two creates, one gets the folder
            space = create_workspace(path, title)
        except (WorkspaceError, FileExistsError) as e:
            raise StarletteHTTPException(409, f"{name!r} already exists on this server") from e
        return envelope({**space.to_dict(), "workspace": name})

    @app.get(f"/{v}/workspaces/{{ws}}", tags=["workspaces"])
    def status(
        ws: str,
        last_runs: int = Query(5, ge=0, description="How many recent runs to include"),
        who: Any = Depends(identity),
    ) -> dict[str, Any]:
        """A workspace's metadata, artifact counts and last runs."""
        return envelope({**workspace(who, ws).status(last_runs), "workspace": ws})

    @app.get(f"/{v}/workspaces/{{ws}}/config", tags=["workspaces"])
    def get_config(ws: str, who: Any = Depends(identity)) -> dict[str, Any]:
        """The workspace's own settings, as dotted keys."""
        return envelope({"settings": workspace(who, ws).settings()})

    @app.put(f"/{v}/workspaces/{{ws}}/config/{{key}}", tags=["workspaces"])
    def set_config(
        ws: str,
        key: str,
        value: Any = Body(..., embed=True, description="A string, number, boolean or list"),
        who: Any = Depends(identity),
    ) -> dict[str, Any]:
        """Set a setting (``<operation>.<request field>``) in the workspace."""
        from bioamla.common.settings import check_setting_value
        from bioamla.operations import check_setting_key

        space = workspace(who, ws)
        try:
            space.set_setting(check_setting_key(key), check_setting_value(value))
        except ConfigError as e:
            raise InvalidInputError(str(e)) from e
        return envelope({"key": key, "value": value})

    @app.delete(f"/{v}/workspaces/{{ws}}/config/{{key}}", tags=["workspaces"])
    def unset_config(ws: str, key: str, who: Any = Depends(identity)) -> dict[str, Any]:
        """Remove a setting from the workspace."""
        return envelope({"key": key, "removed": workspace(who, ws).unset_setting(key)})

    @app.get(f"/{v}/workspaces/{{ws}}/files", tags=["files"])
    def list_files(
        ws: str,
        path: str = Query(".", description="A folder, relative to the workspace"),
        recursive: bool = Query(False, description="Include subfolders"),
        who: Any = Depends(identity),
    ) -> dict[str, Any]:
        """A folder's files and subfolders (``path``, ``type``, ``size``, ``modified``),
        at most 10,000 (``truncated`` says when there were more: list a subfolder)."""
        space, folder = in_workspace(who, ws, path)
        files = space.list_files(folder, recursive=recursive, limit=LIST_LIMIT + 1)
        return envelope(
            {"path": path, "files": files[:LIST_LIMIT], "truncated": len(files) > LIST_LIMIT}
        )

    @app.get(
        f"/{v}/workspaces/{{ws}}/files/{{path:path}}",
        tags=["files"],
        response_class=FileResponse,
        responses={200: {"content": {"application/octet-stream": {}}}},
    )
    def download(ws: str, path: str, who: Any = Depends(identity)) -> FileResponse:
        """Download a file."""
        _, target = in_workspace(who, ws, path)
        if not target.is_file():
            raise NotFoundError(f"No file {path}")
        return FileResponse(target, filename=target.name)

    @app.put(f"/{v}/workspaces/{{ws}}/files/{{path:path}}", tags=["files"], status_code=201)
    async def upload(
        ws: str,
        path: str,
        request: Request,
        overwrite: bool = Query(False, description="Replace an existing file"),
        who: Any = Depends(identity),
    ) -> dict[str, Any]:
        """Upload a file into the workspace: the request body, written as is. Missing
        folders are created; the file appears whole or not at all."""
        from bioamla.common.files import AtomicFileWriter

        space, target = await run_in_threadpool(in_workspace, who, ws, path)
        home = space.root.resolve()
        if not target.is_relative_to(home):
            raise AccessDenied(f"{path} is outside workspace {ws!r}")
        if target.is_dir():
            raise InvalidInputError(f"{path} is a folder")
        if target.exists() and not overwrite:
            raise StarletteHTTPException(409, f"{path} exists; add ?overwrite=true to replace it")
        try:
            writer = await run_in_threadpool(AtomicFileWriter, target, overwrite)
        except OSError as e:
            raise InvalidInputError(f"Can't write {path}: {e.strerror or e}") from e
        with writer:
            async for chunk in request.stream():
                await run_in_threadpool(writer.write, chunk)
            try:
                await run_in_threadpool(writer.commit)
            except FileExistsError as e:
                raise StarletteHTTPException(
                    409, f"{path} exists; add ?overwrite=true to replace it"
                ) from e
        return envelope({"path": space.relative(target), "size": writer.size})


__all__ = ["add_workspace_routes"]

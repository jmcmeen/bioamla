"""Workspace and file routes: the server's workspaces, their settings and their files.

- ``GET /v1/workspaces``: the workspaces under the server root
- ``POST /v1/workspaces``: create one (``{"name", "title"}``)
- ``GET /v1/workspaces/{ws}``: its status (metadata, artifact counts, last runs)
- ``GET /v1/workspaces/{ws}/config``: its settings (``.bioamla/config.toml``)
- ``PUT|DELETE /v1/workspaces/{ws}/config/{key}``: set (``{"value"}``) or remove one
- ``GET /v1/workspaces/{ws}/files?path=&recursive=&sha256=``: list a folder (or one
  file), with each file's SHA-256 when asked
- ``GET /v1/workspaces/{ws}/files/{path}``: download a file
- ``PUT /v1/workspaces/{ws}/files/{path}``: upload a file (the request body, as is;
  ``?overwrite=true`` to replace one)
- ``GET /v1/workspaces/{ws}/audio/{path}?start=&duration=&sample_rate=``: a window of a
  recording, decoded on the server and returned as a float32 WAV, so a viewer can show
  a few seconds of a long recording without downloading it

Every path goes through the server's path policy, so it must resolve inside the
workspace root or a linkable folder, and never to a workspace's metadata. Uploads go
only into the workspace named in the URL. To use recordings in a linkable folder without copying them, link
them with the ``workspace.add`` operation.

The server only writes a workspace's own settings, never the server user's file.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
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
    config = app.state.config
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
        path: str = Query(".", description="A folder (or a file), relative to the workspace"),
        recursive: bool = Query(False, description="Include subfolders"),
        sha256: bool = Query(False, description="Add each file's SHA-256, read on the server"),
        who: Any = Depends(identity),
    ) -> dict[str, Any]:
        """A folder's files and subfolders (``path``, ``type``, ``size``, ``modified``, and
        ``sha256`` when asked), at most 10,000 (``truncated`` says when there were more:
        list a subfolder). A file's ``path`` lists just it. With ``sha256`` every file
        listed is read, so a client can fingerprint files without downloading them."""
        space, folder = in_workspace(who, ws, path)
        files = space.list_files(folder, recursive=recursive, limit=LIST_LIMIT + 1, sha256=sha256)
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

    @app.get(
        f"/{v}/workspaces/{{ws}}/audio/{{path:path}}",
        tags=["files"],
        response_class=Response,
        responses={200: {"content": {"audio/wav": {}}}},
    )
    async def audio_window(
        ws: str,
        path: str,
        start: float = Query(0.0, ge=0, description="Window start, in seconds"),
        duration: float = Query(..., gt=0, description="Window length, in seconds"),
        sample_rate: int | None = Query(
            None, gt=0, le=MAX_SAMPLE_RATE, description="Resample to this rate"
        ),
        keep_channels: bool = Query(
            False, description="Keep the recording's channels instead of mixing to mono"
        ),
        who: Any = Depends(identity),
    ) -> Response:
        """A window of a recording as a float32 WAV, decoded on the server: only that
        part of the file is read, so a few seconds of a long recording cost a few
        seconds' worth of bytes. It is mixed to mono unless ``keep_channels``, and cut
        short at the end of the recording; the ``X-Bioamla-Start``,
        ``X-Bioamla-Duration``, ``X-Bioamla-Sample-Rate`` and ``X-Bioamla-Channels``
        headers say what came back."""
        if not (math.isfinite(start) and math.isfinite(duration)):
            raise InvalidInputError("start and duration must be finite")
        if duration > config.max_window:
            raise InvalidInputError(
                f"duration {duration:g} s is above this server's limit of {config.max_window:g} s"
            )
        body, headers = await run_in_threadpool(
            _window_wav, in_workspace, who, ws, path, start, duration, sample_rate,
            keep_channels, config.max_window,
        )  # fmt: skip
        return Response(body, media_type="audio/wav", headers=headers)


#: The highest rate a client may ask a window to be resampled to.
MAX_SAMPLE_RATE = 384_000


def _window_wav(
    in_workspace: Any, who: Any, ws: str, path: str, start: float, duration: float,
    sample_rate: int | None, keep_channels: bool = False, max_window: float = math.inf,
) -> tuple[bytes, dict[str, str]]:  # fmt: skip
    """The window as WAV bytes (float32), with the headers describing it. All the file
    work, for the threadpool."""
    from bioamla.audio import audio_to_wav_bytes, get_audio_info, load_audio_data
    from bioamla.exceptions import AudioLoadError

    _, target = in_workspace(who, ws, path)
    if not target.is_file():
        raise NotFoundError(f"No file {path}")
    try:
        info = get_audio_info(target)
        length = info.duration
        # The cap is on what the server holds: a window keeping 8 channels is 8 mono
        # windows' worth of samples.
        if keep_channels and info.channels > 1 and duration * info.channels > max_window:
            raise InvalidInputError(
                f"{duration:g} s of {info.channels} channels is above this server's limit of "
                f"{max_window:g} channel-seconds; ask for at most "
                f"{max_window / info.channels:g} s, or mix to mono"
            )
        if start >= length:
            raise InvalidInputError(
                f"start {start:g} s is past the end of {path} ({length:g} s long)"
            )
        audio = load_audio_data(target, start=start, duration=duration,
                                sample_rate=sample_rate, keep_channels=keep_channels)  # fmt: skip
    except AudioLoadError as e:
        # The client's problem, not the server's; the library's message names the
        # server's absolute path, so it stays out of the response.
        raise InvalidInputError(f"{path} can't be decoded as audio") from e
    if len(audio.samples) == 0:
        raise InvalidInputError(f"a window of {duration:g} s rounds to no samples")
    return audio_to_wav_bytes(audio), {
        "X-Bioamla-Start": repr(float(audio.metadata.get("start_time", start))),
        "X-Bioamla-Duration": repr(float(audio.duration)),
        "X-Bioamla-Sample-Rate": str(audio.sample_rate),
        "X-Bioamla-Channels": str(audio.channels),
    }


__all__ = ["add_workspace_routes"]

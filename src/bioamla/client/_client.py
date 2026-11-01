"""The HTTP client for a bioamla server (``bioamla serve``)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote

from bioamla.exceptions import InvalidInputError, NotFoundError, RemoteError

#: Job states after which nothing changes.
FINISHED = ("succeeded", "failed", "cancelled")


class RemoteNotFound(RemoteError, NotFoundError):
    """The server has no such workspace, file, job or operation (HTTP 404)."""


class RemoteInvalidInput(RemoteError, InvalidInputError):
    """The server refused a request as invalid (HTTP 400 or 422)."""


class UntypedResult(RemoteError):
    """A typed run's result couldn't be rebuilt here (e.g. its type needs an extra this
    machine doesn't have); :attr:`result` is its JSON.

    Attributes:
        result: The result as JSON, as an untyped run returns it.
    """

    def __init__(self, message: str, result: Any):
        super().__init__(message)
        self.result = result


class JobFailed(RemoteError):
    """A job finished without a result (failed or cancelled).

    Attributes:
        job: The job as the server reported it.
    """

    def __init__(self, job: dict[str, Any]):
        error = job.get("error") or {}
        super().__init__(
            f"Job {job.get('id')} {job.get('status')}: {error.get('message', '')}".rstrip(": "),
            error_type=error.get("type"),
        )
        self.job = job


def _quote_path(path: str) -> str:
    return quote(str(path), safe="/")


class Client:
    """Talk to a bioamla server: workspaces, files, operations and jobs.

    Every call returns the ``result`` of the server's envelope and raises
    :class:`~bioamla.exceptions.RemoteError` (or :class:`RemoteNotFound`,
    :class:`RemoteInvalidInput`) with the server's message when it refuses or fails.

    Args:
        url: The server, e.g. ``http://lab-server:8765``.
        token: Its bearer token.
        timeout: Seconds to wait for a response (``None``: forever). Streams and
            transfers wait between chunks, not for the whole body.
        session: A ``requests``-compatible session to use (tests pass one).

    Example:
        >>> from bioamla.client import Client
        >>> server = Client("http://localhost:8765", token)
        >>> server.run("costa-rica", "detect.energy", {"file": "raw/rec.wav"})
    """

    def __init__(
        self, url: str, token: str, *, timeout: float | None = 30.0, session: Any = None
    ) -> None:
        import requests

        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def __repr__(self) -> str:
        return f"Client({self.url!r})"

    # Plumbing --------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """The response for ``method path``, after checking it for an error envelope."""
        import requests

        kwargs.setdefault("timeout", self.timeout)
        try:
            response = self.session.request(method, f"{self.url}/v1{path}", **kwargs)
        except requests.RequestException as e:
            raise RemoteError(f"Can't reach {self.url}: {e}") from e
        if response.status_code >= 400:
            raise _error(response)
        return response

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        return _json(self._request(method, path, **kwargs), self.url).get("result")

    # Service ---------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """``{"status": "ok"}`` if the server is up (needs no token)."""
        return self._call("GET", "/health")

    def version(self) -> dict[str, Any]:
        """The server's bioamla, API, workspace-format versions and who you are."""
        return self._call("GET", "/version")

    def operations(self) -> dict[str, Any]:
        """``{"operations": [...], "unavailable": {...}}``: what the server can run."""
        return self._call("GET", "/operations")

    # Workspaces ------------------------------------------------------------------

    def workspaces(self) -> list[dict[str, Any]]:
        """The server's workspaces (``workspace`` is the name to use in calls)."""
        return self._call("GET", "/workspaces")

    def create_workspace(self, name: str, title: str | None = None) -> dict[str, Any]:
        """Create the workspace ``name`` on the server."""
        return self._call("POST", "/workspaces", json={"name": name, "title": title})

    def status(self, ws: str, last_runs: int = 5) -> dict[str, Any]:
        """A workspace's metadata, artifact counts and last runs."""
        return self._call("GET", f"/workspaces/{quote(ws)}", params={"last_runs": last_runs})

    def settings(self, ws: str) -> dict[str, Any]:
        """A workspace's own settings, as dotted keys."""
        return self._call("GET", f"/workspaces/{quote(ws)}/config")["settings"]

    def set_setting(self, ws: str, key: str, value: Any) -> dict[str, Any]:
        """Set a workspace setting (``<operation>.<request field>``)."""
        return self._call("PUT", f"/workspaces/{quote(ws)}/config/{quote(key)}",
                          json={"value": value})  # fmt: skip

    def unset_setting(self, ws: str, key: str) -> bool:
        """Remove a workspace setting; whether it was set."""
        return self._call("DELETE", f"/workspaces/{quote(ws)}/config/{quote(key)}")["removed"]

    # Files -----------------------------------------------------------------------

    def files(self, ws: str, path: str = ".", recursive: bool = False) -> dict[str, Any]:
        """``{"path", "files", "truncated"}`` for a folder in the workspace."""
        return self._call("GET", f"/workspaces/{quote(ws)}/files",
                          params={"path": path, "recursive": recursive})  # fmt: skip

    def upload(
        self, ws: str, local: str | Path, remote: str | None = None, overwrite: bool = False
    ) -> dict[str, Any]:
        """Upload the file ``local`` to ``remote`` (default: its name) in the workspace.

        Returns:
            ``{"path", "size"}`` as written on the server.
        """
        local = Path(local)
        if not local.is_file():
            raise NotFoundError(f"No file {local}")
        with local.open("rb") as f:
            return self._call("PUT", f"/workspaces/{quote(ws)}/files/{_quote_path(remote or local.name)}",
                              params={"overwrite": overwrite}, data=f)  # fmt: skip

    def download(
        self, ws: str, remote: str, local: str | Path | None = None, overwrite: bool = False
    ) -> Path:
        """Download ``remote`` from the workspace. ``local`` is the file to write, or a
        folder to write it into by its name when it is an existing folder or ends in
        ``/`` (default: the current folder). Written whole or not at all.

        Returns:
            The file written.

        Raises:
            InvalidInputError: If the file exists and ``overwrite`` is off.
            RemoteError: If the server refuses, or the transfer breaks.
        """
        import requests

        from bioamla.common.files import AtomicFileWriter

        name = Path(remote).name
        if local is None:
            target = Path(name)
        elif str(local).endswith(("/", "\\")) or Path(local).is_dir():
            target = Path(local) / name
        else:
            target = Path(local)
        if target.exists() and not overwrite:
            raise InvalidInputError(f"{target} exists; pass overwrite to replace it")
        response = self._request("GET", f"/workspaces/{quote(ws)}/files/{_quote_path(remote)}",
                                 stream=True)  # fmt: skip
        try:
            with response, AtomicFileWriter(target, overwrite=True) as writer:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    writer.write(chunk)
                writer.commit()
        except requests.RequestException as e:
            raise RemoteError(f"Lost the connection to {self.url}: {e}") from e
        return target

    # Operations and jobs ---------------------------------------------------------

    def submit(self, ws: str, name: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """Post the operation ``name`` to the workspace ``ws``.

        Returns:
            For a quick operation, its result; for a job, the job (``id``, ``status``).
            :meth:`run` waits for a job's result instead.
        """
        return self._submit(ws, name, request)[1]

    def _submit(
        self, ws: str, name: str, request: dict[str, Any] | None, **params: Any
    ) -> tuple[bool, dict[str, Any], Any]:
        """``(is_job, result_or_job, body)`` for posting the operation."""
        response = self._request("POST", f"/workspaces/{quote(ws)}/operations/{name}",
                                 json=request or {}, params=params or None)  # fmt: skip
        body = _json(response, self.url)
        return response.status_code == 202, body.get("result"), body

    def run(
        self,
        ws: str,
        name: str,
        request: dict[str, Any] | None = None,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        typed: bool = False,
    ) -> Any:
        """Run the operation ``name`` in ``ws`` and return its result, waiting for a
        job to finish (streaming its events to ``on_event`` and its progress to
        ``on_progress``).

        Args:
            typed: Return the same objects a local run returns (rebuilt from the
                server's wire form, see :mod:`bioamla.common.wire`) instead of the
                result's JSON.

        Raises:
            UntypedResult: With ``typed``, if the result can't be rebuilt here (its
                ``result`` holds the JSON).
            JobFailed: If the job failed or was cancelled.
            RemoteError: If the server refused the request.
        """
        params = {"wire": "true"} if typed else {}
        is_job, answer, body = self._submit(ws, name, request, **params)
        if is_job:
            body = self._wait(answer["id"], on_progress, on_event, typed=typed)
            answer = body["result"]["result"]
        if not typed:
            return answer
        if "wire" not in body:
            reason = body.get("wire_error") or "the server sent no typed result"
            raise UntypedResult(f"{name}: {reason}", answer)
        from bioamla.common.wire import WireError, from_wire

        try:
            return from_wire(body["wire"])
        except WireError as e:
            raise UntypedResult(f"Can't rebuild the result of {name}: {e}", answer) from e

    def wait(
        self,
        job_id: str,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        retries: int = 5,
    ) -> dict[str, Any]:
        """Follow a job's events until it finishes; the finished job. A dropped
        stream is resumed after the last event seen (up to ``retries`` times in a row).

        Raises:
            JobFailed: If it failed or was cancelled.
            RemoteError: If the server can't be reached again.
        """
        return self._wait(job_id, on_progress, on_event, retries=retries)["result"]

    def _wait(
        self,
        job_id: str,
        on_progress: Callable[[int, int], None] | None,
        on_event: Callable[[dict[str, Any]], None] | None,
        retries: int = 5,
        typed: bool = False,
    ) -> dict[str, Any]:
        """:meth:`wait`, returning the final answer's whole body (with the typed
        result when ``typed``)."""
        import time

        job_id = self.resolve_job_id(job_id)
        after, failures = 0, 0
        while True:
            try:
                for event in self.events(job_id, after=after):
                    after, failures = event["id"], 0
                    if on_event is not None:
                        on_event(event)
                    if on_progress is not None and event["event"] == "progress":
                        on_progress(event["done"], event["total"])
            except RemoteError as e:
                if e.status is not None:  # the server answered: not a dropped stream
                    raise
                failures += 1
                if failures > retries:
                    raise
                time.sleep(min(2**failures * 0.25, 5))
                continue
            body = _json(self._request("GET", f"/jobs/{quote(job_id)}",
                                       params={"wire": "true"} if typed else None), self.url)  # fmt: skip
            if body["result"]["status"] in FINISHED:
                break
        if body["result"]["status"] != "succeeded":
            raise JobFailed(body["result"])
        return body

    def jobs(self, workspace: str | None = None) -> list[dict[str, Any]]:
        """The server's jobs since it started, newest first."""
        return self._call("GET", "/jobs", params={"workspace": workspace} if workspace else None)

    def resolve_job_id(self, job_id: str) -> str:
        """The full id of the one job whose id starts with ``job_id`` (listings show
        them shortened); a full id is returned as is.

        Raises:
            RemoteNotFound: If no job, or several, match.
        """
        if len(job_id) >= 32:
            return job_id
        matches = [j["id"] for j in self.jobs() if j["id"].startswith(job_id)]
        if len(matches) != 1:
            problem = "No job" if not matches else "Several jobs match"
            raise RemoteNotFound(f"{problem} {job_id!r}", 404, "NotFoundError")
        return matches[0]

    def job(self, job_id: str) -> dict[str, Any]:
        """A job (by id or a unique prefix): status, progress, and its result or error
        once done."""
        return self._call("GET", f"/jobs/{quote(self.resolve_job_id(job_id))}")

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Ask a job (by id or a unique prefix) to stop; the job."""
        return self._call("POST", f"/jobs/{quote(self.resolve_job_id(job_id))}/cancel")

    def log(self, job_id: str) -> str:
        """A job's notices and log lines."""
        return self._request("GET", f"/jobs/{quote(self.resolve_job_id(job_id))}/log").text

    def events(self, job_id: str, after: int = 0) -> Iterator[dict[str, Any]]:
        """A job's events (``status``, ``progress``, ``notice``, ``log``) as they
        happen, from after event ``after``, until it finishes.

        Raises:
            RemoteError: If the server can't be reached or the stream breaks.
        """
        import requests

        headers = {"Last-Event-ID": str(after)} if after else {}
        response = self._request("GET", f"/jobs/{quote(job_id)}/events", stream=True,
                                 headers=headers, timeout=(self.timeout, None))  # fmt: skip
        with response:
            try:
                for line in response.iter_lines(decode_unicode=True):
                    if line and line.startswith("data: "):
                        yield json.loads(line[len("data: ") :])
            except requests.RequestException as e:
                raise RemoteError(f"Lost the connection to {self.url}: {e}") from e


def _json(response: Any, url: str) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError as e:
        raise RemoteError(f"{url} answered with something other than JSON") from e


def _error(response: Any) -> RemoteError:
    """The exception for an error response, with the server's message."""
    try:
        error = response.json().get("error") or {}
    except ValueError:
        error = {}
    message = error.get("message") or f"HTTP {response.status_code}"
    error_type = error.get("type")
    status = response.status_code
    if status == 404:
        return RemoteNotFound(message, status, error_type)
    if status in (400, 422):
        return RemoteInvalidInput(message, status, error_type)
    if status == 401:
        return RemoteError(f"The server refused the token: {message}", status, error_type)
    return RemoteError(message, status, error_type)


__all__ = [
    "FINISHED",
    "Client",
    "JobFailed",
    "RemoteInvalidInput",
    "RemoteNotFound",
    "UntypedResult",
]

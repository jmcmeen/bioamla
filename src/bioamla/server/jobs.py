"""Jobs: long operations run in worker processes, with status, progress and cancel.

A job-kind operation posted to ``/v1/workspaces/{ws}/operations/{name}`` is queued
and answered at once (``202``) with the job. Each job runs in its own worker process
(see :mod:`bioamla.server._worker`), so a crash or a cancel can't take the server
down. At most ``max_jobs`` run at once, and at most one machine-learning job
(``models.*``, ``batch.models.*``); the rest wait in order.

Routes:

- ``GET /v1/jobs``: every job this server has run since it started (``?workspace=``
  narrows them)
- ``GET /v1/jobs/{id}``: its status, progress, and its result or error once done
- ``GET /v1/jobs/{id}/events``: its events as server-sent events (``status``,
  ``progress``, ``notice``, ``log``), from the start or after ``Last-Event-ID``,
  until it finishes
- ``GET /v1/jobs/{id}/log``: its notices and log lines as text
- ``POST /v1/jobs/{id}/cancel``: ask it to stop. A queued job is dropped; a running
  one is asked through its cancel check, and its worker is ended if it hasn't stopped
  after a grace period.

Jobs are kept in memory; each is also recorded in its workspace's history (with
``job`` set to its id), so the record outlives the server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Operation name prefixes that need the machine's ML hardware; one runs at a time.
ML_PREFIXES = ("models.", "batch.models.")

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
)
FINISHED = (SUCCEEDED, FAILED, CANCELLED)


def is_ml(name: str) -> bool:
    """Whether the operation ``name`` is a machine-learning job (one at a time)."""
    return name.startswith(ML_PREFIXES)


@dataclass
class Job:
    """One job: what it runs, where, and how it's going.

    Attributes:
        id: The job id.
        operation: The operation it runs.
        workspace: The workspace's name on the server.
        user: Who submitted it.
        request: The request, as JSON-ready data.
        status: ``queued``, ``running``, ``succeeded``, ``failed`` or ``cancelled``.
        created: When it was submitted (epoch seconds).
        started: When its worker started.
        finished: When it finished.
        progress: The latest ``{"done", "total"}``.
        result: The operation's result once it succeeded, as the routes report it.
        error: ``{"type", "message"}`` once it failed or was cancelled.
        events: Everything that happened, in order (each with an ``id``).
    """

    id: str
    operation: str
    workspace: str
    user: str
    request: dict[str, Any]
    status: str = QUEUED
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: dict[str, int] | None = None
    result: Any = None
    error: dict[str, str] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    # Not reported:
    ws_dir: Path | None = field(default=None, repr=False)
    request_obj: Any = field(default=None, repr=False)
    cancel_requested: float | None = field(default=None, repr=False)
    _process: Any = field(default=None, repr=False)
    _conn: Any = field(default=None, repr=False)
    _cancel: Any = field(default=None, repr=False)

    @property
    def ml(self) -> bool:
        """Whether it's a machine-learning job."""
        return is_ml(self.operation)

    def to_dict(self) -> dict[str, Any]:
        """The job as the API reports it (``result`` and ``error`` once done)."""
        return {
            "id": self.id,
            "operation": self.operation,
            "workspace": self.workspace,
            "user": self.user,
            "status": self.status,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
        }


class JobRunner:
    """Runs jobs in worker processes, a few at a time and one ML job at a time.

    Args:
        root: The server's workspace root.
        linkable: Folders outside ``root`` that operations may use.
        max_jobs: How many jobs may run at once.
        cancel_grace: Seconds a cancelled job gets to stop before its worker is ended.
        poll: Seconds between checks on the workers.
    """

    def __init__(
        self,
        root: Path,
        linkable: list[Path] | tuple = (),
        *,
        max_jobs: int = 2,
        cancel_grace: float = 5.0,
        poll: float = 0.05,
    ) -> None:
        import multiprocessing

        self.root = Path(root)
        self.linkable = [Path(p) for p in linkable]
        self.max_jobs = max(1, max_jobs)
        self.cancel_grace = cancel_grace
        self.poll = poll
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._mp = multiprocessing.get_context("spawn")  # no fork of a threaded server
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # Submitting and reading ------------------------------------------------------

    def submit(self, ws_dir: Path, workspace: str, name: str, request: Any, user: str) -> Job:
        """Queue the operation ``name`` on ``request`` (already validated) in ``ws_dir``."""
        from bioamla.common.jsonable import to_jsonable

        job = Job(
            id=uuid.uuid4().hex,
            operation=name,
            workspace=workspace,
            user=user,
            request=to_jsonable(asdict(request)),
            ws_dir=ws_dir,
            request_obj=request,
        )
        with self._lock:
            self.jobs[job.id] = job
            self._event(job, "status", status=QUEUED)
            if self._thread is None:
                self._thread = threading.Thread(target=self._pump, name="bioamla-jobs", daemon=True)
                self._thread.start()
        return job

    def get(self, job_id: str) -> Job:
        """The job ``job_id``.

        Raises:
            NotFoundError: If there's no such job.
        """
        from bioamla.exceptions import NotFoundError

        job = self.jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"No job {job_id!r}")
        return job

    def events_after(self, job: Job, after: int) -> tuple[list[dict[str, Any]], bool]:
        """The job's events with ids above ``after``, and whether it has finished."""
        with self._lock:
            return [e for e in job.events if e["id"] > after], job.status in FINISHED

    def cancel(self, job_id: str) -> Job:
        """Ask the job to stop: a queued job is dropped, a running one asked to stop.

        Raises:
            NotFoundError: If there's no such job.
        """
        job = self.get(job_id)
        with self._lock:
            if job.status == QUEUED:
                self._finish(job, CANCELLED, error=_cancelled())
            elif job.status == RUNNING and job.cancel_requested is None:
                job.cancel_requested = time.time()
                job._cancel.set()
                self._event(job, "notice", level="info", message="Cancel requested")
        return job

    def close(self) -> None:
        """Stop the runner: queued jobs are dropped and running workers ended."""
        self._stop.set()
        with self._lock:
            for job in self.jobs.values():
                if job.status == RUNNING and job._process is not None:
                    job._process.terminate()
                if job.status in (QUEUED, RUNNING):
                    self._finish(job, CANCELLED, error=_cancelled("The server stopped"))
        if self._thread is not None:
            self._thread.join(timeout=5)

    # The pump --------------------------------------------------------------------

    def _pump(self) -> None:
        while not self._stop.is_set():
            try:
                with self._lock:
                    self._tick()
            except Exception:  # noqa: BLE001 - keep the runner alive
                logger.exception("Job runner error")
            self._stop.wait(self.poll)

    def _tick(self) -> None:
        running = [j for j in self.jobs.values() if j.status == RUNNING]
        for job in running:
            self._drain(job)
            if job.status != RUNNING:
                continue
            if not job._process.is_alive():
                job._process.join()
                self._drain(job)
                if job.status == RUNNING:  # ended without a result
                    if job.cancel_requested is not None:
                        self._finish(job, CANCELLED, error=_cancelled())
                    else:
                        code = job._process.exitcode
                        self._finish(job, FAILED, error={
                            "type": "WorkerError",
                            "message": f"The worker exited with code {code} before finishing",
                        })  # fmt: skip
            elif (
                job.cancel_requested is not None
                and time.time() - job.cancel_requested > self.cancel_grace
            ):
                job._process.terminate()
        self._start_queued()

    def _start_queued(self) -> None:
        running = [j for j in self.jobs.values() if j.status == RUNNING]
        ml_busy = any(j.ml for j in running)
        for job in sorted(self.jobs.values(), key=lambda j: j.created):
            if len(running) >= self.max_jobs:
                return
            if job.status != QUEUED or (job.ml and ml_busy):
                continue
            self._start(job)
            running.append(job)
            ml_busy = ml_busy or job.ml

    def _start(self, job: Job) -> None:
        from bioamla.server._worker import run_job

        receive, send = self._mp.Pipe(duplex=False)
        job._conn, job._cancel = receive, self._mp.Event()
        job._process = self._mp.Process(
            target=run_job,
            args=(send, job._cancel, job.operation, job.request, str(job.ws_dir),
                  str(self.root), [str(p) for p in self.linkable], job.user),
            name=f"bioamla-job-{job.id[:8]}",
            daemon=True,
        )  # fmt: skip
        job._process.start()
        send.close()  # the worker holds the only sending end, so EOF means it's gone
        job.status, job.started = RUNNING, time.time()
        self._event(job, "status", status=RUNNING)

    def _drain(self, job: Job) -> None:
        conn = job._conn
        try:
            while conn.poll():
                message = conn.recv()
                self._handle(job, message)
                if job.status != RUNNING:
                    return
        except (EOFError, OSError):
            pass

    def _handle(self, job: Job, message: tuple) -> None:
        kind, *rest = message
        if kind == "progress":
            done, total = rest
            job.progress = {"done": done, "total": total}
            self._event(job, "progress", done=done, total=total)
        elif kind in ("notice", "log"):
            level, text = rest
            self._event(job, kind, level=level, message=text)
        elif kind == "result":
            self._finish(job, SUCCEEDED, result=rest[0])
        elif kind == "error":
            error = rest[0]
            status = CANCELLED if error.get("type") == "OperationCancelled" else FAILED
            self._finish(job, status, error=error)

    def _finish(
        self, job: Job, status: str, *, result: Any = None, error: dict | None = None
    ) -> None:
        from bioamla.server.operations_api import record_server_run
        from bioamla.workspace import open_workspace

        job.status, job.finished = status, time.time()
        job.result, job.error = result, error
        if job._conn is not None:
            job._conn.close()
        payload = {"status": status, "result": result, "error": error}
        self._event(job, "status", **payload)
        if job.started is None:
            return  # never ran: nothing to record
        try:
            ws = open_workspace(job.ws_dir)
        except Exception:  # noqa: BLE001 - history must never fail a job
            logger.warning("Couldn't open %s to record job %s", job.ws_dir, job.id, exc_info=True)
            return
        record_server_run(
            ws, job.operation, job.request_obj, user=job.user, started=job.started,
            ok=status == SUCCEEDED, error=error, result=result, job_id=job.id,
        )  # fmt: skip

    def _event(self, job: Job, kind: str, **data: Any) -> None:
        job.events.append({"id": len(job.events) + 1, "event": kind, "time": time.time(), **data})


def _cancelled(message: str = "Cancelled") -> dict[str, str]:
    return {"type": "OperationCancelled", "message": message}


def add_job_routes(app: Any, api_version: str, envelope: Any, runner: JobRunner) -> None:
    """Add the job routes and make ``app.state.submit_job`` queue jobs on ``runner``."""
    from fastapi import Depends, Header, Query
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

    identity = app.state.identity

    def submit_job(ws_dir: Path, name: str, request: Any, who: Any) -> JSONResponse:
        from bioamla import operations
        from bioamla.server.operations_api import operation_context

        # Check the request's paths now, so a bad one is a 403 rather than a failed job.
        operations.prepare(name, request, operation_context(app, who, ws_dir))
        job = runner.submit(ws_dir, ws_dir.name, name, request, who.user)
        return JSONResponse(status_code=202, content=envelope(job.to_dict()))

    app.state.submit_job = submit_job
    app.state.jobs = runner

    @app.get(f"/{api_version}/jobs", tags=["jobs"])
    def list_jobs(
        workspace: str | None = Query(None, description="Only this workspace's jobs"),
        who: Any = Depends(identity),
    ) -> dict[str, Any]:
        """Every job this server has run since it started, newest first."""
        jobs = sorted(runner.jobs.values(), key=lambda j: j.created, reverse=True)
        return envelope(
            [j.to_dict() for j in jobs if workspace is None or j.workspace == workspace]
        )

    @app.get(f"/{api_version}/jobs/{{job_id}}", tags=["jobs"])
    def get_job(job_id: str, who: Any = Depends(identity)) -> dict[str, Any]:
        """A job's status and progress, and its result or error once it's done."""
        return envelope(runner.get(job_id).to_dict())

    @app.post(f"/{api_version}/jobs/{{job_id}}/cancel", tags=["jobs"])
    def cancel_job(job_id: str, who: Any = Depends(identity)) -> dict[str, Any]:
        """Ask a job to stop (a queued job is dropped). Answers with the job."""
        return envelope(runner.cancel(job_id).to_dict())

    @app.get(f"/{api_version}/jobs/{{job_id}}/log", tags=["jobs"], response_class=PlainTextResponse)
    def job_log(job_id: str, who: Any = Depends(identity)) -> str:
        """A job's notices and log lines, one per line (``LEVEL message``)."""
        job = runner.get(job_id)
        events, _ = runner.events_after(job, 0)
        return "".join(
            f"{e['level'].upper()} {e['message']}\n"
            for e in events
            if e["event"] in ("notice", "log")
        )

    @app.get(
        f"/{api_version}/jobs/{{job_id}}/events",
        tags=["jobs"],
        response_class=StreamingResponse,
        response_model=None,
    )
    async def job_events(
        job_id: str,
        last_event_id: int = Header(0, alias="Last-Event-ID"),
        who: Any = Depends(identity),
    ) -> Any:
        """A job's events as server-sent events, until it finishes. Each event's
        ``data`` is JSON; its ``event`` is ``status``, ``progress``, ``notice`` or
        ``log``. Send ``Last-Event-ID`` to resume after an event."""
        job = runner.get(job_id)

        async def stream():
            after = last_event_id
            while True:
                events, finished = runner.events_after(job, after)
                for e in events:
                    after = e["id"]
                    yield f"id: {e['id']}\nevent: {e['event']}\ndata: {json.dumps(e)}\n\n"
                if finished:
                    return
                await asyncio.sleep(runner.poll)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )


__all__ = ["FINISHED", "ML_PREFIXES", "Job", "JobRunner", "add_job_routes", "is_ml"]

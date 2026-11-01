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
        want_wire: Whether the submitter asked for the typed result.
        wire: ``{"wire"}`` or ``{"wire_error"}`` once it succeeded, if asked for.
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
    want_wire: bool = field(default=False, repr=False)
    wire: dict[str, Any] | None = field(default=None, repr=False)
    # Not reported:
    ws_dir: Path | None = field(default=None, repr=False)
    request_obj: Any = field(default=None, repr=False)
    cancel_requested: float | None = field(default=None, repr=False)
    _signalled: int = field(default=0, repr=False)
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
        cancel_grace: Seconds a cancelled job gets to stop before its worker is ended
            (and as long again before it is killed).
        poll: Seconds between checks on the workers.
        keep_jobs: Finished jobs kept for the job routes; older ones are forgotten
            (their history records stay in their workspaces).
        max_events: Events kept per job; past that, further progress and log events
            are dropped (status events and the latest progress are always kept).
        timeout: Seconds a job may run before it's failed and its worker ended
            (``None``: no limit), so a hung worker can't hold a slot forever.
    """

    def __init__(
        self,
        root: Path,
        linkable: list[Path] | tuple = (),
        *,
        max_jobs: int = 2,
        cancel_grace: float = 5.0,
        poll: float = 0.05,
        keep_jobs: int = 200,
        max_events: int = 5000,
        timeout: float | None = None,
    ) -> None:
        import multiprocessing

        self.root = Path(root)
        self.linkable = [Path(p) for p in linkable]
        self.max_jobs = max(1, max_jobs)
        self.cancel_grace = cancel_grace
        self.poll = poll
        self.keep_jobs = keep_jobs
        self.timeout = timeout
        self.max_events = max_events
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._mp = multiprocessing.get_context("spawn")  # no fork of a threaded server
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # Submitting and reading ------------------------------------------------------

    def submit(
        self,
        ws_dir: Path,
        workspace: str,
        name: str,
        request: Any,
        user: str,
        wire: bool = False,
    ) -> Job:
        """Queue the operation ``name`` on ``request`` (already validated) in ``ws_dir``;
        with ``wire``, its result is also kept in the typed wire form."""
        from bioamla.common.jsonable import to_jsonable

        job = Job(
            id=uuid.uuid4().hex,
            operation=name,
            workspace=workspace,
            user=user,
            request=to_jsonable(asdict(request)),
            ws_dir=ws_dir,
            request_obj=request,
            want_wire=wire,
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
        """The job's events with ids above ``after``, and whether it has finished.

        Takes no lock, so it never waits on the runner (event ids are positions in the
        list, which only grows, and a job's final event is added before its status
        changes).
        """
        finished = job.status in FINISHED
        return job.events[max(after, 0) :], finished

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
            finished = []
            for job in self.jobs.values():
                if job.status == RUNNING and job._process is not None:
                    _signal(job._process, kill=False)
                    job._process.join(timeout=self.cancel_grace)
                    if job._process.is_alive():
                        _signal(job._process, kill=True)
                if job.status in (QUEUED, RUNNING):
                    self._finish(job, CANCELLED, error=_cancelled("The server stopped"))
                    finished.append(job)
        self._record(finished)
        if self._thread is not None:
            self._thread.join(timeout=5)

    # The pump --------------------------------------------------------------------

    def _pump(self) -> None:
        while not self._stop.is_set():
            finished: list[Job] = []
            try:
                with self._lock:
                    finished = self._tick()
                    self._prune()
            except Exception:  # noqa: BLE001 - keep the runner alive
                logger.exception("Job runner error")
            self._record(finished)  # history I/O outside the lock
            self._stop.wait(self.poll)

    def _tick(self) -> list[Job]:
        """Check every running job, start queued ones; the jobs that finished."""
        finished = []
        for job in [j for j in self.jobs.values() if j.status == RUNNING]:
            try:
                self._check(job)
            except Exception as e:  # noqa: BLE001 - one bad job mustn't stop the rest
                logger.exception("Job %s broke the runner's protocol", job.id)
                _signal(job._process, kill=True)
                self._finish(job, FAILED, error={"type": "WorkerError", "message": str(e)})
            if job.status != RUNNING:
                finished.append(job)
        self._start_queued()
        return finished

    def _check(self, job: Job) -> None:
        self._drain(job)
        if job.status != RUNNING:
            return
        if not job._process.is_alive():
            job._process.join()
            self._drain(job)
            if job.status != RUNNING:
                return
            if job.cancel_requested is not None:  # ended without a result
                self._finish(job, CANCELLED, error=_cancelled())
            else:
                code = job._process.exitcode
                self._finish(job, FAILED, error={
                    "type": "WorkerError",
                    "message": f"The worker exited with code {code} before finishing",
                })  # fmt: skip
        elif (
            self.timeout is not None
            and job.cancel_requested is None
            and time.time() - (job.started or time.time()) > self.timeout
        ):
            _signal(job._process, kill=True)
            job._process.join(timeout=self.cancel_grace)
            self._finish(job, FAILED, error={
                "type": "JobTimeout",
                "message": f"Stopped after running longer than {self.timeout:g} s",
            })  # fmt: skip
        elif job.cancel_requested is not None:
            waited = time.time() - job.cancel_requested
            if waited > 2 * self.cancel_grace and job._signalled < 2:
                _signal(job._process, kill=True)
                job._signalled = 2
            elif waited > self.cancel_grace and job._signalled < 1:
                _signal(job._process, kill=False)
                job._signalled = 1

    def _prune(self) -> None:
        done = sorted((j for j in self.jobs.values() if j.status in FINISHED),
                      key=lambda j: j.finished or 0)  # fmt: skip
        for job in done[: max(0, len(done) - self.keep_jobs)]:
            del self.jobs[job.id]

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
                  str(self.root), [str(p) for p in self.linkable], job.user, job.want_wire),
            name=f"bioamla-job-{job.id[:8]}",
            # Not a daemon: a daemonic process can't start its own (e.g. a torch
            # DataLoader's workers). close() ends workers when the server stops.
            daemon=False,
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
            pass  # the worker is gone; _check decides how it ended

    def _handle(self, job: Job, message: tuple) -> None:
        kind, *rest = message
        if kind == "progress":
            done, total = rest
            job.progress = {"done": done, "total": total}
            if len(job.events) < self.max_events:
                self._event(job, "progress", done=done, total=total)
        elif kind in ("notice", "log"):
            level, text = rest
            if kind == "notice" or len(job.events) < self.max_events:
                self._event(job, kind, level=level, message=text)
        elif kind == "result":
            job.wire = rest[1] if len(rest) > 1 else None
            self._finish(job, SUCCEEDED, result=rest[0])
        elif kind == "error":
            error = rest[0]
            status = CANCELLED if error.get("type") == "OperationCancelled" else FAILED
            self._finish(job, status, error=error)
        else:
            raise ValueError(f"Unexpected message from the worker: {kind!r}")

    def _finish(
        self, job: Job, status: str, *, result: Any = None, error: dict | None = None
    ) -> None:
        # Result and final event first, status last: a reader that sees the job
        # finished also sees its result and its last event.
        job.result, job.error, job.finished = result, error, time.time()
        if job._conn is not None:
            job._conn.close()
        self._event(job, "status", status=status, result=result, error=error)
        job.status = status

    def _record(self, jobs: list[Job]) -> None:
        """Add the finished ``jobs`` that ran to their workspaces' history."""
        from bioamla.server.operations_api import record_server_run
        from bioamla.workspace import open_workspace

        for job in jobs:
            if job.started is None:
                continue  # never ran: nothing to record
            try:
                ws = open_workspace(job.ws_dir)
            except Exception:  # noqa: BLE001 - history must never fail a job
                logger.warning("Couldn't open %s to record job %s", job.ws_dir, job.id,
                               exc_info=True)  # fmt: skip
                continue
            record_server_run(
                ws, job.operation, job.request_obj, user=job.user, started=job.started,
                ok=job.status == SUCCEEDED, error=job.error, result=job.result, job_id=job.id,
            )  # fmt: skip

    def _event(self, job: Job, kind: str, **data: Any) -> None:
        job.events.append({"id": len(job.events) + 1, "event": kind, "time": time.time(), **data})


def _signal(process: Any, kill: bool) -> None:
    """Ask a worker to stop (SIGTERM), or kill it, with its own child processes."""
    import os
    import signal

    if process is None or process.pid is None:
        return
    if hasattr(os, "killpg"):
        try:  # the worker leads its own process group (see run_job)
            os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
            return
        except OSError:
            pass
    process.kill() if kill else process.terminate()


def _cancelled(message: str = "Cancelled") -> dict[str, str]:
    return {"type": "OperationCancelled", "message": message}


def add_job_routes(app: Any, api_version: str, envelope: Any, runner: JobRunner) -> None:
    """Add the job routes and make ``app.state.submit_job`` queue jobs on ``runner``."""
    from fastapi import Depends, Header, Query
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

    identity = app.state.identity

    def submit_job(
        ws_dir: Path, name: str, request: Any, who: Any, wire: bool = False
    ) -> JSONResponse:
        from bioamla import operations
        from bioamla.server.operations_api import operation_context, record_server_run

        # Check the request's paths now, so a bad one is a 403 rather than a failed job.
        # The worker gets the request as sent and checks it again.
        context = operation_context(app, who, ws_dir)
        started = time.time()
        try:
            operations.prepare(name, request, context)
        except Exception as e:
            error = {"type": type(e).__name__, "message": str(e)}
            record_server_run(context.workspace, name, request, user=who.user,
                              started=started, ok=False, error=error)  # fmt: skip
            raise
        job = runner.submit(ws_dir, ws_dir.name, name, request, who.user, wire=wire)
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
    def get_job(
        job_id: str,
        wire: bool = Query(False, description="Also answer with the typed result (`wire`)"),
        who: Any = Depends(identity),
    ) -> dict[str, Any]:
        """A job's status and progress, and its result or error once it's done."""
        job = runner.get(job_id)
        answer = envelope(job.to_dict())
        if wire and job.status == SUCCEEDED:
            answer.update(job.wire or {"wire_error": "Not asked for when the job was submitted"})
        return answer

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

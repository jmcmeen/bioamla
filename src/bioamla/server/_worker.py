"""The worker process behind a server job: runs one operation and reports back.

The server starts :func:`run_job` in a fresh process per job, so a crash, a stuck
library call or a cancel can't take the server down. The worker sends messages over a
one-way pipe:

- ``("progress", done, total)``
- ``("notice", level, message)``
- ``("log", level, message)``: the worker's log records at INFO and above
- ``("result", data)``: the operation's result, JSON-ready, as the routes report it
- ``("error", {"type", "message"})``: the operation failed (``OperationCancelled``
  when it stopped for a cancel)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from bioamla.exceptions import BioamlaError


class _PipeHandler(logging.Handler):
    def __init__(self, send: Any) -> None:
        super().__init__(logging.INFO)
        self._send = send
        self.setFormatter(logging.Formatter("%(message)s"))  # adds any traceback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._send(("log", record.levelname.lower(), self.format(record)))
        except Exception:  # noqa: BLE001 - a lost log line must not fail the job
            pass


def run_job(
    conn: Any,
    cancel: Any,
    name: str,
    request: dict[str, Any],
    ws_dir: str,
    root: str,
    linkable: list[str],
    user: str,
) -> None:
    """Run the operation ``name`` on ``request`` in the workspace ``ws_dir``.

    Args:
        conn: The sending end of the pipe to the server.
        cancel: An event the server sets to ask the operation to stop.
        name: The operation.
        request: Its request, as JSON-ready data.
        ws_dir: The workspace folder; relative paths are relative to it.
        root: The server's workspace root, and ``linkable`` its linkable folders:
            every path the operation touches must be inside one of them.
        user: Who the job runs as.
    """
    import threading

    from bioamla import operations
    from bioamla.common.jsonable import to_jsonable
    from bioamla.server._security import Identity, PathPolicy
    from bioamla.server.operations_api import result_data
    from bioamla.workspace import open_workspace

    if hasattr(os, "setsid"):
        os.setsid()  # its own process group, so a cancel also ends its children
    lock = threading.Lock()  # a batch's worker threads log while the main thread reports

    def send(message: tuple) -> None:
        with lock:
            conn.send(message)

    logging.basicConfig(level=logging.INFO, handlers=[_PipeHandler(send)], force=True)
    # Anything that resolves a bare name against the current folder (a model id that
    # might be a local folder, say) sees the workspace, not the server's folder.
    os.chdir(ws_dir)
    policy = PathPolicy(root, linkable)
    who = Identity(user=user)
    base = Path(ws_dir)
    context = operations.OperationContext(
        workspace=open_workspace(base),
        on_progress=lambda done, total: send(("progress", int(done), int(total))),
        cancelled=cancel.is_set,
        path_check=policy.path_check(who, base),
        on_notice=lambda level, message: send(("notice", level, message)),
    )
    try:
        result = operations.run(name, request, context)
        send(("result", to_jsonable(result_data(result))))
    except Exception as e:  # noqa: BLE001 - every failure is reported to the server
        if isinstance(e, BioamlaError):
            error = {"type": type(e).__name__, "message": str(e)}
        else:
            logging.getLogger(__name__).exception("Job %s failed unexpectedly", name)
            error = {"type": "InternalError", "message": f"{type(e).__name__}: {e}"}
        send(("error", error))
    finally:
        conn.close()

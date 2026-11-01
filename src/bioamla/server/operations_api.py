"""Operation routes: one endpoint per remote-safe operation, generated from the registry.

``POST /v1/workspaces/{ws}/operations/{name}`` runs the operation ``name`` in the
workspace ``ws`` (a folder under the server root). The body is the operation's request
(its JSON Schema is in the OpenAPI spec); every path in it is relative to the workspace
and must resolve inside the server's allowed folders. A quick operation answers with
the result envelope (with ``?wire=true``, also the result in the typed wire form, see
:mod:`bioamla.common.wire`, as ``wire``); a job-kind operation is run by the job
runner instead.

``GET /v1/operations`` lists them: name, kind, summary, path fields and request schema.
Each run is recorded in the workspace's history, like a command run there (except the
operations that only read the workspace: its history, status and artifact list).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, Query

from bioamla.exceptions import BioamlaError

logger = logging.getLogger(__name__)


def wire_result(result: Any) -> Any:
    """``result`` in the wire form (:mod:`bioamla.common.wire`), which a client turns
    back into the same objects; ``None`` if it can't be sent that way."""
    from bioamla.common.wire import WireError, to_wire

    try:
        return to_wire(result)
    except WireError:
        logger.info("A %s result has no wire form", type(result).__name__, exc_info=True)
        return None


def result_data(result: Any) -> Any:
    """What a route reports for an operation's result: its ``to_dict()`` if it has one."""
    to_dict = getattr(result, "to_dict", None)
    return to_dict() if callable(to_dict) else result


def record_server_run(
    ws: Any,
    name: str,
    request: Any,
    *,
    user: str,
    started: float,
    ok: bool,
    error: dict[str, Any] | None,
    result: Any = None,
    job_id: str | None = None,
) -> None:
    """Append a server run to the workspace history, in the same record as a CLI run
    (``command`` from the operation name; ``operation`` and ``request`` in place of
    ``argv``). Operations that only read the workspace aren't recorded. A problem
    writing the record is logged, never raised."""
    from dataclasses import asdict, is_dataclass

    from bioamla.common.jsonable import to_jsonable
    from bioamla.workspace import UNRECORDED_COMMANDS

    command = name.replace(".", " ")
    if command in UNRECORDED_COMMANDS:
        return
    try:
        ws.record_run(
            ws.run_record(
                command=command,
                started=started,
                ended=time.time(),
                ok=ok,
                exit_code=0 if ok else 1,
                error=error,
                result=to_jsonable(result_data(result)) if ok else None,
                user=user,
                operation=name,
                request=to_jsonable(asdict(request) if is_dataclass(request) else request),
                via="server",
                job=job_id,
            )
        )
    except Exception:  # noqa: BLE001 - history must never fail a run
        logger.warning("Couldn't record the run of %s in %s", name, ws.root, exc_info=True)


def operation_context(app: FastAPI, who: Any, ws_dir: Path, **kwargs: Any) -> Any:
    """The context a server run gets: the workspace, and the path policy as its check
    (relative paths are relative to the workspace)."""
    from bioamla import operations
    from bioamla.workspace import open_workspace

    policy = app.state.policy

    def check(path: str) -> str:
        return str(policy.authorize(who, path, base=ws_dir))

    return operations.OperationContext(workspace=open_workspace(ws_dir), path_check=check, **kwargs)


def add_operation_routes(app: FastAPI, api_version: str, envelope: Any) -> list[str]:
    """Add ``GET /{v}/operations`` and one ``POST`` route per remote-safe operation.

    Returns:
        The operation names that got a route.
    """
    from bioamla import operations
    from bioamla.common.config import config_schema

    identity = app.state.identity
    available, unavailable = operations.all_operations()
    remote = [op for op in available if op.remote]

    listing = {
        "operations": [
            {
                "name": op.name,
                "kind": op.kind,
                "summary": op.summary,
                "path_fields": list(op.path_fields()),
                "request_schema": config_schema(op.request),
            }
            for op in remote
        ],
        "unavailable": unavailable,
    }

    @app.get(f"/{api_version}/operations", tags=["operations"])
    def list_operations(who: Any = Depends(identity)) -> dict[str, Any]:
        """Every operation this server runs: name, kind, summary, path fields and the
        request's JSON Schema; and the domains that aren't available (missing extras)."""
        return envelope(listing)

    for op in remote:
        _add_route(app, api_version, envelope, op)
    return [op.name for op in remote]


def _add_route(app: FastAPI, api_version: str, envelope: Any, op: Any) -> None:
    from bioamla import operations

    identity = app.state.identity
    request_type = op.request

    def run(ws: str, request: Any, who: Any, wire: bool) -> dict[str, Any]:
        ws_dir = app.state.policy.workspace_dir(who, ws)
        if op.kind == "job":
            return app.state.submit_job(ws_dir, op.name, request, who)
        context = operation_context(app, who, ws_dir)
        started = time.time()
        try:
            result = operations.run(op.name, request, context)
        except Exception as e:
            # Recorded as the route reports it: unexpected errors stay unexplained.
            if isinstance(e, BioamlaError):
                error = {"type": type(e).__name__, "message": str(e)}
            else:
                error = {"type": "InternalError", "message": "Unexpected server error"}
            record_server_run(context.workspace, op.name, request, user=who.user,
                              started=started, ok=False, error=error)  # fmt: skip
            raise
        record_server_run(context.workspace, op.name, request, user=who.user,
                          started=started, ok=True, error=None, result=result)  # fmt: skip
        answer = envelope(result_data(result))
        if wire:
            answer["wire"] = wire_result(result)
        return answer

    def endpoint(
        ws: str,
        request: request_type = Body(...),  # type: ignore[valid-type]
        wire: bool = Query(False, description="Also answer with the typed result (`wire`)"),
        who: Any = Depends(identity),
    ) -> dict[str, Any]:
        return run(ws, request, who, wire)

    # This module's annotations are strings (``from __future__``); give FastAPI the
    # real request type so it validates the body and puts the schema in the spec.
    endpoint.__annotations__ = {**endpoint.__annotations__, "request": request_type}
    endpoint.__name__ = "op_" + op.name.replace(".", "_").replace("-", "_")
    endpoint.__doc__ = op.summary + (
        "\n\nRuns as a job: answers with a job id." if op.kind == "job" else ""
    )
    app.post(
        f"/{api_version}/workspaces/{{ws}}/operations/{op.name}",
        tags=[op.name.split(".", 1)[0]],
        summary=op.summary or op.name,
        operation_id=endpoint.__name__,
    )(endpoint)


__all__ = [
    "add_operation_routes",
    "operation_context",
    "record_server_run",
    "result_data",
    "wire_result",
]

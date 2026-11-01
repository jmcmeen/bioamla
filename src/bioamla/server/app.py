"""The HTTP API (``/v1``): the FastAPI application.

Responses use the same envelope as ``bioamla --json``: ``{"bioamla", "ok",
"result"}`` or ``{"bioamla", "ok": false, "error": {"type", "message"}}``. Every
route except ``/v1/health`` needs ``Authorization: Bearer <token>``, including the
OpenAPI spec at ``/v1/openapi.json`` (there are no interactive docs pages).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

from bioamla.exceptions import BioamlaError, InvalidInputError, NotFoundError
from bioamla.server._security import (
    AccessDenied,
    Identity,
    PathPolicy,
    Unauthenticated,
    check_token,
)

logger = logging.getLogger(__name__)

API_VERSION = "v1"


@dataclass
class ServerConfig:
    """How a server runs.

    Attributes:
        root: Folder holding the server's workspaces.
        token: The bearer token clients must present.
        linkable: Folders outside ``root`` that workspaces may link (e.g. an archive
            of recordings), used in place.
        max_jobs: How many jobs may run at once (machine-learning jobs: one at a time).
        job_timeout: Seconds a running job may go without reporting anything before
            it's stopped (``None``: no limit).
    """

    root: Path
    token: str
    linkable: list[Path] = field(default_factory=list)
    max_jobs: int = 2
    job_timeout: float | None = None


def envelope(result: Any) -> dict[str, Any]:
    """The success envelope for ``result``, as ``bioamla --json`` reports it.

    Args:
        result: The route's result; dataclasses, paths and the like are made JSON-ready.

    Returns:
        ``{"bioamla": <version>, "ok": True, "result": ...}``.
    """
    from bioamla import __version__
    from bioamla.common.jsonable import to_jsonable

    return {"bioamla": __version__, "ok": True, "result": to_jsonable(result)}


def _error(status: int, error_type: str, message: str) -> JSONResponse:
    from bioamla import __version__

    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    return JSONResponse(
        status_code=status,
        content={"bioamla": __version__, "ok": False,
                 "error": {"type": error_type, "message": message}},
        headers=headers,
    )  # fmt: skip


_HTTP_ERROR_TYPES = {404: "NotFoundError", 405: "MethodNotAllowed", 409: "Conflict"}


def create_app(config: ServerConfig) -> FastAPI:
    """The API application for ``config``.

    Args:
        config: The workspace root, token and linkable folders.

    Returns:
        A FastAPI application. ``app.state.policy`` is its :class:`PathPolicy`,
        ``app.state.identity`` the dependency that checks the token and
        ``app.state.jobs`` its :class:`~bioamla.server.jobs.JobRunner`.
    """
    from contextlib import asynccontextmanager

    from bioamla import __version__
    from bioamla.server.jobs import JobRunner, add_job_routes

    runner = JobRunner(
        config.root, config.linkable, max_jobs=config.max_jobs, timeout=config.job_timeout
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        runner.close()  # end running workers with the server

    app = FastAPI(
        title="bioamla",
        version=__version__,
        description="Run bioamla operations in workspaces on this machine.",
        openapi_url=None,  # served below, behind the token
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    policy = PathPolicy(config.root, config.linkable)
    app.state.config = config
    app.state.policy = policy
    # Declares the bearer scheme in the spec; auto_error=False so a missing or wrong
    # header reaches check_token and gets the 401 envelope.
    bearer = HTTPBearer(auto_error=False, description="The server token")

    def identity(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Identity:
        return check_token(credentials.credentials if credentials else None, config.token)

    app.state.identity = identity  # the auth dependency, for routes added later

    @app.exception_handler(Unauthenticated)
    async def _unauthenticated(request: Request, exc: Unauthenticated) -> JSONResponse:
        return _error(401, "Unauthenticated", str(exc))

    @app.exception_handler(AccessDenied)
    async def _denied(request: Request, exc: AccessDenied) -> JSONResponse:
        return _error(403, "AccessDenied", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg')}" for e in exc.errors()
        )
        return _error(422, "ValidationError", problems or "Invalid request")

    @app.exception_handler(NotFoundError)
    async def _not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return _error(404, "NotFoundError", str(exc))

    @app.exception_handler(InvalidInputError)
    async def _invalid(request: Request, exc: InvalidInputError) -> JSONResponse:
        return _error(400, type(exc).__name__, str(exc))

    @app.exception_handler(BioamlaError)
    async def _other(request: Request, exc: BioamlaError) -> JSONResponse:
        return _error(500, type(exc).__name__, str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        response = _error(
            exc.status_code, _HTTP_ERROR_TYPES.get(exc.status_code, "HTTPError"), str(exc.detail)
        )
        response.headers.update(exc.headers or {})  # e.g. Allow on a 405
        return response

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unexpected error handling %s %s", request.method, request.url.path)
        return _error(500, "InternalError", "Unexpected server error; see the server log")

    @app.get(f"/{API_VERSION}/health", tags=["service"])
    def health() -> dict[str, Any]:
        """Liveness check; needs no token."""
        return envelope({"status": "ok"})

    @app.get(f"/{API_VERSION}/openapi.json", include_in_schema=False)
    def openapi(who: Identity = Depends(identity)) -> dict[str, Any]:
        return app.openapi()

    from bioamla.server.operations_api import add_operation_routes

    add_job_routes(app, API_VERSION, envelope, runner)
    from bioamla.server.workspaces_api import add_workspace_routes

    add_workspace_routes(app, API_VERSION, envelope)

    app.state.operations = add_operation_routes(app, API_VERSION, envelope)

    @app.get(f"/{API_VERSION}/version", tags=["service"])
    def version(who: Identity = Depends(identity)) -> dict[str, Any]:
        """bioamla and API versions."""
        from bioamla.workspace import FORMAT_VERSION, HISTORY_RECORD_VERSION

        return envelope({
            "bioamla": __version__,
            "api": API_VERSION,
            "workspace_format": FORMAT_VERSION,
            "history_record": HISTORY_RECORD_VERSION,
            "user": who.user,
        })  # fmt: skip

    return app

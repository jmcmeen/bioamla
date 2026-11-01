"""The HTTP API (``/v1``): the FastAPI application.

Responses use the same envelope as ``bioamla --json``: ``{"bioamla", "ok",
"result"}`` or ``{"bioamla", "ok": false, "error": {"type", "message"}}``. Every
route except ``/v1/health`` needs ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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
    """

    root: Path
    token: str
    linkable: list[Path] = field(default_factory=list)


def envelope(result: Any) -> dict[str, Any]:
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


def create_app(config: ServerConfig) -> FastAPI:
    """The API application for ``config``."""
    from bioamla import __version__

    app = FastAPI(
        title="bioamla",
        version=__version__,
        description="Run bioamla commands in workspaces on this machine.",
        openapi_url=f"/{API_VERSION}/openapi.json",
        docs_url=f"/{API_VERSION}/docs",
        redoc_url=None,
    )
    policy = PathPolicy(config.root, config.linkable)
    app.state.config = config
    app.state.policy = policy

    def identity(request: Request) -> Identity:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        return check_token(token.strip() if scheme.lower() == "bearer" else None, config.token)

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

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unexpected error handling %s %s", request.method, request.url.path)
        return _error(500, "InternalError", "Unexpected server error; see the server log")

    @app.get(f"/{API_VERSION}/health", tags=["service"])
    def health() -> dict[str, Any]:
        """Liveness check; needs no token."""
        return envelope({"status": "ok"})

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

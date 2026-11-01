"""
bioamla.server — run bioamla commands in workspaces on this machine, over HTTP.

Install with ``pip install 'bioamla[server]'`` and start with ``bioamla serve``.
The design (workspaces, jobs, the security model) is described in
[Workspaces, remote work and agent skills](../architecture/workspaces-and-remote.md).
"""

from bioamla._extras import require_extra

require_extra("server", "bioamla.server")

from bioamla.server._security import (  # noqa: E402
    AccessDenied,
    Identity,
    PathPolicy,
    Unauthenticated,
    check_token,
    read_or_create_token,
)
from bioamla.server.app import API_VERSION, ServerConfig, create_app, envelope  # noqa: E402
from bioamla.server.openapi import openapi_spec, write_openapi_spec  # noqa: E402

__all__ = [
    "API_VERSION",
    "AccessDenied",
    "Identity",
    "PathPolicy",
    "ServerConfig",
    "Unauthenticated",
    "check_token",
    "create_app",
    "envelope",
    "openapi_spec",
    "read_or_create_token",
    "write_openapi_spec",
]

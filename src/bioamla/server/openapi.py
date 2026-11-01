"""The server's OpenAPI document, as committed in ``docs/api/openapi.json``.

A running server serves its own at ``/v1/openapi.json``; this is the same document for
an app with no workspaces, with ``info.version`` set to the API version (``v1``)
rather than the bioamla version, so it changes only when the API does. Regenerate it
with ``bioamla serve --openapi docs/api/openapi.json``; a test fails while the
committed file is out of date.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def openapi_spec() -> dict[str, Any]:
    """The OpenAPI document of the ``/v1`` API.

    Raises:
        DependencyError: If an extra is missing: its operations would have no routes,
            so the document would be incomplete. Generate it with every extra
            (``pip install 'bioamla[all,server]'``).
    """
    import tempfile

    from bioamla import operations
    from bioamla.exceptions import DependencyError
    from bioamla.server.app import API_VERSION, ServerConfig, create_app

    _, unavailable = operations.all_operations()
    if unavailable:
        raise DependencyError(
            "The OpenAPI document needs every extra (pip install 'bioamla[all,server]'); "
            f"missing for: {', '.join(sorted(unavailable))}"
        )
    with tempfile.TemporaryDirectory() as root:
        app = create_app(ServerConfig(root=Path(root), token="unused"))
        spec = app.openapi()
        app.state.jobs.close()
    spec["info"]["version"] = API_VERSION
    return spec


def write_openapi_spec(path: str | Path) -> Path:
    """Write :func:`openapi_spec` to ``path`` as indented JSON.

    Raises:
        DependencyError: If an extra is missing (see :func:`openapi_spec`).
        OSError: If it can't be written.
    """
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(openapi_spec(), indent=2) + "\n", encoding="utf-8")
    return out


__all__ = ["openapi_spec", "write_openapi_spec"]

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

    Every operation route is included only if its domain's extra is installed, so
    generate it with every extra (``pip install 'bioamla[all,server]'``).
    """
    import tempfile

    from bioamla.server.app import API_VERSION, ServerConfig, create_app

    with tempfile.TemporaryDirectory() as root:
        spec = create_app(ServerConfig(root=Path(root), token="unused")).openapi()
    spec["info"]["version"] = API_VERSION
    return spec


def write_openapi_spec(path: str | Path) -> Path:
    """Write :func:`openapi_spec` to ``path`` as indented JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(openapi_spec(), indent=2) + "\n", encoding="utf-8")
    return out


__all__ = ["openapi_spec", "write_openapi_spec"]

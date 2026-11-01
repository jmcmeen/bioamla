"""Who is calling, and which paths they may touch.

The first version has one user and one token. The seams for a multi-user server
are already here: every request resolves to an :class:`Identity`, and every path a
request or job touches goes through :meth:`PathPolicy.authorize`.
"""

from __future__ import annotations

import hmac
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from bioamla.exceptions import BioamlaError, InvalidInputError, NotFoundError


class Unauthenticated(BioamlaError):
    """A request presented no valid token (HTTP 401)."""


class AccessDenied(BioamlaError):
    """A request touched a path it may not use (HTTP 403)."""


@dataclass(frozen=True)
class Identity:
    """Who a request acts as. One user for now; more with a later multi-user server."""

    user: str = "owner"


def read_or_create_token(path: str | Path) -> str:
    """The server token stored in ``path``, creating one (mode 0600) if it's missing.

    Raises:
        InvalidInputError: If the file exists but is empty, or is readable by others.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass  # use the existing token (also when another process just created it)
    else:
        token = secrets.token_urlsafe(32)
        with os.fdopen(fd, "w") as f:
            f.write(token + "\n")
        return token
    try:
        mode = path.stat().st_mode
        token = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as e:
        raise InvalidInputError(f"{path} disappeared while being read; try again") from e
    if os.name != "nt" and mode & 0o077:
        raise InvalidInputError(
            f"{path} is accessible by other users; restrict it with `chmod 600 {path}`"
        )
    if not token:
        raise InvalidInputError(f"{path} is empty")
    return token


def check_token(presented: str | None, expected: str) -> Identity:
    """The identity for a bearer token.

    Raises:
        Unauthenticated: If it doesn't match.
    """
    if not presented or not hmac.compare_digest(presented.encode(), expected.encode()):
        raise Unauthenticated("Missing or invalid token")
    return Identity()


class PathPolicy:
    """Every path the server touches must resolve inside the workspace root, or inside
    one of the folders allow-listed for linking (e.g. an archive of recordings).

    Paths are resolved with symlinks followed, so a link inside a workspace can't
    reach outside the allowed folders.
    """

    def __init__(self, root: str | Path, linkable: list[str | Path] | tuple = ()) -> None:
        self.root = Path(root).expanduser().resolve()
        self.linkable = tuple(Path(p).expanduser().resolve() for p in linkable)

    @property
    def allowed(self) -> tuple[Path, ...]:
        return (self.root, *self.linkable)

    def authorize(
        self, identity: Identity, path: str | Path, base: str | Path | None = None
    ) -> Path:
        """The resolved ``path`` (relative paths against ``base``, default the root), if
        ``identity`` may use it.

        A path from a client is taken literally (``~`` isn't expanded).

        Raises:
            AccessDenied: If it resolves outside every allowed folder.
            InvalidInputError: If it isn't a usable path (e.g. contains a NUL).
        """
        try:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = Path(base or self.root) / candidate
            resolved = candidate.resolve()
        except (OSError, ValueError, RuntimeError) as e:
            raise InvalidInputError(f"Not a usable path: {path!r}") from e
        if not any(resolved == a or resolved.is_relative_to(a) for a in self.allowed):
            raise AccessDenied(f"{path} is outside the folders this server may use")
        return resolved

    def workspace_dir(self, identity: Identity, name: str) -> Path:
        """The directory of the workspace called ``name`` under the root.

        Raises:
            InvalidInputError: If ``name`` isn't a plain folder name.
            NotFoundError: If there's no such workspace.
        """
        if (
            not name
            or name in (".", "..")
            or Path(name).name != name
            or name.startswith((".", "~"))
            or "\\" in name
        ):
            raise InvalidInputError(f"Not a workspace name: {name!r}")
        path = self.authorize(identity, name)
        if path.parent != self.root:  # e.g. a symlink under the root pointing elsewhere
            raise AccessDenied(f"Workspace {name!r} is not inside the workspace root")
        from bioamla.workspace import is_workspace

        if not is_workspace(path):
            raise NotFoundError(f"No workspace {name!r}")
        return path

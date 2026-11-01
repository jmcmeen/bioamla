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
from typing import Any

from bioamla.exceptions import BioamlaError, ConfigError, InvalidInputError, NotFoundError


class Unauthenticated(BioamlaError):
    """A request presented no valid token (HTTP 401)."""


class AccessDenied(BioamlaError):
    """A request touched a path it may not use (HTTP 403)."""


@dataclass(frozen=True)
class Identity:
    """Who a request acts as. One user for now; more with a later multi-user server."""

    user: str = "owner"


def read_or_create_token(path: str | Path) -> str:
    """The server token stored in ``path``, creating one if it's missing.

    A new token is written to a private temporary file and then linked into place, so
    a server starting at the same moment never reads a half-written token: whichever
    process links first wins, and the others read its token.

    On POSIX systems the file is created with mode 0600 and an existing file that
    other users can access is refused. On Windows the file gets the folder's default
    access rules and isn't checked; keep it in a folder only you can read.

    Args:
        path: The token file (``~`` is expanded).

    Returns:
        The token.

    Raises:
        InvalidInputError: If the file isn't one line holding a token (empty, several
            lines, spaces, not UTF-8), or other users can access it (POSIX).
        ConfigError: If the file or its folder can't be created or read.
    """
    path = Path(path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            created = _create_token(path)
            if created is not None:
                return created
        mode = path.stat().st_mode
        raw = path.read_bytes()
    except FileNotFoundError as e:
        raise InvalidInputError(f"{path} disappeared while being read; try again") from e
    except OSError as e:
        raise ConfigError(f"Can't read or create the token file {path}: {e}") from e
    if os.name != "nt" and mode & 0o077:
        raise InvalidInputError(
            f"{path} is accessible by other users; restrict it with `chmod 600 {path}`"
        )
    return _parse_token(path, raw)


def _create_token(path: Path) -> str | None:
    """Write a new token to ``path`` atomically; None if another process got there first."""
    import tempfile

    token = secrets.token_urlsafe(32)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".token-")  # mode 0600
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)  # fails if path exists: never replaces a token
        except FileExistsError:
            return None
        return token
    finally:
        os.unlink(tmp)


def _parse_token(path: Path, raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise InvalidInputError(f"{path} isn't a token file (not UTF-8 text)") from e
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise InvalidInputError(f"{path} is empty")
    if len(lines) > 1 or any(c.isspace() for c in lines[0]):
        raise InvalidInputError(f"{path} must hold the token alone, on one line")
    return lines[0]


def check_token(presented: str | None, expected: str) -> Identity:
    """The identity for a bearer token.

    Args:
        presented: The token the request sent, or None if it sent none.
        expected: The server's token.

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

    Args:
        root: The folder holding the server's workspaces.
        linkable: Folders outside ``root`` that workspaces may link in place.
    """

    def __init__(self, root: str | Path, linkable: list[str | Path] | tuple = ()) -> None:
        self.root = Path(root).expanduser().resolve()
        self.linkable = tuple(Path(p).expanduser().resolve() for p in linkable)

    @property
    def allowed(self) -> tuple[Path, ...]:
        """The resolved folders a path may be in: the root, then the linkable folders."""
        return (self.root, *self.linkable)

    def authorize(
        self, identity: Identity, path: str | Path, base: str | Path | None = None
    ) -> Path:
        """The resolved ``path`` (relative paths against ``base``, default the root), if
        ``identity`` may use it.

        A path from a client is taken literally (``~`` isn't expanded).

        Args:
            identity: Who is asking.
            path: The path, absolute or relative.
            base: What a relative ``path`` is relative to (default: the root).

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

    def path_check(self, identity: Identity, base: str | Path) -> Any:
        """``check(path) -> str``: :meth:`authorize` for ``identity``, relative paths
        against ``base``, as an operation context's ``path_check``."""
        return lambda path: str(self.authorize(identity, path, base=base))

    def workspace_dir(self, identity: Identity, name: str) -> Path:
        """The directory of the workspace called ``name`` under the root.

        The workspace's ``.bioamla`` folder and ``workspace.toml`` must be inside it,
        not links to somewhere else.

        Args:
            identity: Who is asking.
            name: The workspace's folder name under the root.

        Raises:
            InvalidInputError: If ``name`` isn't a plain folder name.
            AccessDenied: If the workspace, or its metadata, resolves outside the root.
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
        from bioamla.workspace import WORKSPACE_DIR, is_workspace

        if not is_workspace(path):
            raise NotFoundError(f"No workspace {name!r}")
        meta = path / WORKSPACE_DIR
        for item in (meta, meta / "workspace.toml"):
            if not item.resolve().is_relative_to(path / WORKSPACE_DIR):
                raise AccessDenied(f"Workspace {name!r} has metadata outside the workspace")
        return path

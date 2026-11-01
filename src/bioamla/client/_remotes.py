"""Named remotes: servers listed in the user's settings file.

A remote is a ``[remotes.<name>]`` table in ``~/.config/bioamla/config.toml``::

    [remotes.lab]
    url = "http://lab-server:8765"
    token_env = "BIOAMLA_LAB_TOKEN"

The token is read from the named environment variable when connecting; it is never
stored in the file.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bioamla.exceptions import ConfigError, InvalidInputError, NotFoundError

logger = logging.getLogger(__name__)

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Remote:
    """A server listed in the user's settings.

    Attributes:
        name: Its name (``--remote NAME``).
        url: The server's address.
        token_env: The environment variable holding its token.
    """

    name: str
    url: str
    token_env: str

    @property
    def token_set(self) -> bool:
        """Whether :attr:`token_env` holds a token."""
        return bool(os.environ.get(self.token_env, "").strip())

    def token(self) -> str:
        """The token from :attr:`token_env`.

        Raises:
            ConfigError: If the variable isn't set.
        """
        token = os.environ.get(self.token_env, "").strip()
        if not token:
            raise ConfigError(
                f"Remote {self.name!r} needs its token in ${self.token_env} "
                "(copy it from the server's token file)"
            )
        return token

    def connect(self, **kwargs: Any) -> Any:
        """A :class:`~bioamla.client.Client` for this remote."""
        from bioamla.client._client import Client

        return Client(self.url, self.token(), **kwargs)

    def to_dict(self) -> dict[str, str]:
        """``{"name", "url", "token_env", "token_set"}``."""
        return {"name": self.name, "url": self.url, "token_env": self.token_env,
                "token_set": self.token_set}  # fmt: skip


def _settings_path(path: str | Path | None) -> Path:
    from bioamla.common.settings import user_config_path

    return Path(path) if path else user_config_path()


def _table(path: str | Path | None) -> dict[str, Any]:
    from bioamla.common.settings import read_settings_file

    table = read_settings_file(_settings_path(path)).get("remotes") or {}
    if not isinstance(table, dict):
        raise ConfigError("`remotes` in the settings file must be a table ([remotes.NAME])")
    return table


def _remote(name: str, entry: Any) -> Remote:
    if not isinstance(entry, dict) or not entry.get("url") or not entry.get("token_env"):
        raise ConfigError(f"Remote {name!r} needs a url and a token_env")
    return Remote(name, str(entry["url"]), str(entry["token_env"]))


def remotes(path: str | Path | None = None) -> list[Remote]:
    """The remotes in the user's settings file (or ``path``), by name. An incomplete
    entry is left out (with a logged warning), so it can't hide the others.

    Raises:
        ConfigError: If the file can't be read.
    """
    found = []
    for name, entry in sorted(_table(path).items()):
        try:
            found.append(_remote(name, entry))
        except ConfigError as e:
            logger.warning("%s", e)
    return found


def get_remote(name: str, path: str | Path | None = None) -> Remote:
    """The remote ``name``.

    Raises:
        NotFoundError: If there's no such remote.
        ConfigError: If its entry is incomplete.
    """
    table = _table(path)
    if name not in table:
        raise NotFoundError(f"No remote {name!r}; add it with `bioamla remote add {name} URL`")
    return _remote(name, table[name])


def add_remote(
    name: str, url: str, token_env: str | None = None, path: str | Path | None = None
) -> Remote:
    """Add (or replace) the remote ``name`` in the user's settings file.

    Args:
        name: Its name: letters, digits, ``-`` and ``_``.
        url: The server, ``http://`` or ``https://``.
        token_env: The environment variable that will hold its token (default
            ``BIOAMLA_<NAME>_TOKEN``).
        path: The settings file (default: the user's).

    Raises:
        InvalidInputError: If the name, URL or variable name isn't usable.
    """
    from bioamla.common.files import write_text_atomic
    from bioamla.common.settings import dump_toml, read_settings_file, set_dotted

    if not _NAME.match(name):
        raise InvalidInputError(f"Not a remote name: {name!r} (letters, digits, - and _)")
    if not url.startswith(("http://", "https://")):
        raise InvalidInputError(f"Not a server URL: {url!r} (http://host:port)")
    token_env = token_env or f"BIOAMLA_{name.upper().replace('-', '_')}_TOKEN"
    if not _ENV.match(token_env):
        raise InvalidInputError(f"Not an environment variable name: {token_env!r}")
    file = _settings_path(path)
    settings = read_settings_file(file)
    set_dotted(settings, f"remotes.{name}", {"url": url.rstrip("/"), "token_env": token_env})
    write_text_atomic(file, dump_toml(settings))
    return Remote(name, url.rstrip("/"), token_env)


def remove_remote(name: str, path: str | Path | None = None) -> bool:
    """Remove the remote ``name``; whether it was there."""
    from bioamla.common.files import write_text_atomic
    from bioamla.common.settings import dump_toml, read_settings_file, unset_dotted

    file = _settings_path(path)
    settings = read_settings_file(file)
    if not unset_dotted(settings, f"remotes.{name}"):
        return False
    write_text_atomic(file, dump_toml(settings))
    return True


def connect(name: str, **kwargs: Any) -> Any:
    """A :class:`~bioamla.client.Client` for the remote ``name``."""
    return get_remote(name).connect(**kwargs)


__all__ = ["Remote", "add_remote", "connect", "get_remote", "remotes", "remove_remote"]

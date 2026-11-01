"""Settings files: the user's and the workspace's ``config.toml``.

Settings cascade, each level overriding the one before:

1. built-in defaults (each command's own)
2. the user file, :func:`user_config_path` (``$XDG_CONFIG_HOME/bioamla/config.toml``,
   default ``~/.config/bioamla/config.toml``; ``%APPDATA%\\bioamla\\config.toml`` on
   Windows)
3. the workspace file, ``.bioamla/config.toml``
4. ``BIOAMLA_*`` environment variables
5. command-line flags

Levels 2 and 3 live here; the CLI turns the merged result into option defaults, and
Click handles 4 and 5. A table mirrors a command's path and its keys are option
names (``-`` or ``_``)::

    [detect.energy]
    threshold_db = -25

    [models.ast.predict]
    model_path = "bioamla/scp-frogs"

Library functions never read these files; they keep taking explicit arguments.
Standard library only.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from bioamla.exceptions import ConfigError

#: Top-level tables that hold settings for something other than a command.
RESERVED_SECTIONS = frozenset({"remotes"})


def user_config_path() -> Path:
    """Where the user's settings file lives (it may not exist)."""
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "bioamla" / "config.toml"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "bioamla" / "config.toml"


def read_settings_file(path: str | Path) -> dict[str, Any]:
    """The settings in ``path``; an empty dict if the file doesn't exist.

    Raises:
        ConfigError: If the file isn't valid TOML.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    from bioamla.common.config import load_toml

    return load_toml(path)


def merge_settings(*layers: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge settings dicts; later layers win, key by key."""
    merged: dict[str, Any] = {}
    for layer in layers:
        for key, value in layer.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_settings(merged[key], value)
            else:
                merged[key] = value
    return merged


def settings_layers(workspace_config: str | Path | None = None) -> list[tuple[Path, dict]]:
    """The settings files that apply, lowest precedence first: the user file, then
    ``workspace_config`` if given. Missing files are left out."""
    layers = []
    for path in (user_config_path(), workspace_config):
        if path is not None:
            data = read_settings_file(path)
            if data:
                layers.append((Path(path), data))
    return layers


def flatten(settings: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"detect": {"energy": {"x": 1}}}`` -> ``{"detect.energy.x": 1}``."""
    flat: dict[str, Any] = {}
    for key, value in settings.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, dotted))
        else:
            flat[dotted] = value
    return flat


def set_dotted(settings: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``a.b.c`` in nested ``settings``, creating tables as needed."""
    *tables, key = dotted.split(".")
    node = settings
    for table in tables:
        node = node.setdefault(table, {})
        if not isinstance(node, dict):
            raise ConfigError(f"{dotted}: {table!r} is a value, not a table")
    node[key] = value


def unset_dotted(settings: dict[str, Any], dotted: str) -> bool:
    """Remove ``a.b.c`` from nested ``settings`` (and tables left empty). True if it was there."""
    *tables, key = dotted.split(".")
    chain = [settings]
    for table in tables:
        node = chain[-1].get(table)
        if not isinstance(node, dict):
            return False
        chain.append(node)
    if key not in chain[-1]:
        return False
    del chain[-1][key]
    for parent, table in zip(reversed(chain[:-1]), reversed(tables), strict=True):
        if parent[table]:
            break
        del parent[table]
    return True


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ConfigError(f"Can't write {type(value).__name__} values to TOML: {value!r}")


def _toml_key(key: str) -> str:
    bare = key.replace("_", "").replace("-", "").isalnum()
    return key if bare else json.dumps(key)


def dump_toml(settings: dict[str, Any]) -> str:
    """Nested tables of strings, numbers, booleans and lists, as TOML.

    Comments in a file read earlier are not kept.
    """
    lines: list[str] = []

    def emit(table: dict[str, Any], path: list[str]) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if scalars and path:
            if lines:
                lines.append("")
            lines.append("[" + ".".join(_toml_key(p) for p in path) + "]")
        for key, value in scalars.items():
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
        for key, value in tables.items():
            emit(value, [*path, key])

    emit(settings, [])
    return "\n".join(lines) + "\n" if lines else ""


def parse_value(text: str) -> Any:
    """A command-line value as TOML would read it (``-25`` -> int, ``true`` -> bool,
    ``[1, 2]`` -> list); anything that isn't a TOML literal is a string."""
    from bioamla.common.config import tomllib

    if tomllib is None:  # pragma: no cover - Python < 3.11 without tomli
        return text
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


__all__ = [
    "RESERVED_SECTIONS",
    "dump_toml",
    "flatten",
    "merge_settings",
    "parse_value",
    "read_settings_file",
    "set_dotted",
    "settings_layers",
    "unset_dotted",
    "user_config_path",
]

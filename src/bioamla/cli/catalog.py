"""The command catalog: every command, argument and option, from the Click tree.

Scripts and agent tools use it to learn what bioamla can do without parsing
``--help``. Each option also names the ``BIOAMLA_*`` variable
and the settings key that can give it a default (see the Settings docs page).
"""

from __future__ import annotations

from typing import Any

import click

# Click 8.2+ marks "no default" with a sentinel; older releases use None.
_UNSET = getattr(click.core, "UNSET", None)

CATALOG_VERSION = 1


def _jsonable(value: Any) -> Any:
    from bioamla.cli.output import to_jsonable

    return to_jsonable(value)


def _type_info(param: click.Parameter) -> dict[str, Any]:
    t = param.type
    info: dict[str, Any] = {"type": t.name}
    if isinstance(t, click.Choice):
        info["choices"] = [str(c) for c in t.choices]
    if isinstance(t, click.Path):
        info["path"] = {"exists": t.exists, "file_okay": t.file_okay, "dir_okay": t.dir_okay}
    if isinstance(t, (click.IntRange, click.FloatRange)):
        info["range"] = {"min": t.min, "max": t.max}
    return info


def _default(param: click.Parameter) -> dict[str, Any]:
    default = param.default
    if callable(default):
        return {"default": None, "default_is_dynamic": True}
    if default is _UNSET or default is None:
        if getattr(param, "is_flag", False) and not getattr(param, "count", False):
            return {"default": False}  # what an unset flag gives the command
        return {"default": None}
    if isinstance(default, (list, tuple)) and not default:
        return {"default": None}
    return {"default": _jsonable(default)}


def _option(path: list[str], param: click.Option) -> dict[str, Any]:
    from bioamla.cli.settings_map import env_name, is_settable, setting_key

    entry: dict[str, Any] = {
        "name": setting_key([], param),
        "flags": [*param.opts, *param.secondary_opts],
        **_type_info(param),
        **_default(param),
        "required": param.required,
        "multiple": param.multiple,
        "nargs": param.nargs,
        "is_flag": param.is_flag,
        "help": param.help or "",
    }
    if path and is_settable(param):  # root options can't be set (see settings_map)
        entry["setting"] = setting_key(path, param)
        env = env_name(path, param)
        if env:  # an option with only a short flag has no variable
            entry["env"] = env
    return entry


def _argument(param: click.Argument) -> dict[str, Any]:
    return {
        "name": param.name,
        **_type_info(param),
        "required": param.required,
        "nargs": param.nargs,
    }


def _clean_help(text: str | None) -> str:
    """Help text without Click's ``\\b`` (don't-rewrap) marker lines."""
    lines = (text or "").strip().splitlines()
    return "\n".join(line for line in lines if line.strip() != "\b").strip()


def _entry(command: click.Command, path: list[str]) -> dict[str, Any]:
    params = [p for p in command.params if not getattr(p, "hidden", False)]
    return {
        "path": " ".join(path),
        "kind": "group" if isinstance(command, click.Group) else "command",
        "summary": command.get_short_help_str(limit=200),
        "help": _clean_help(command.help),
        "arguments": [_argument(p) for p in params if isinstance(p, click.Argument)],
        "options": [
            _option(path, p)
            for p in params
            if isinstance(p, click.Option) and p.name not in ("help",)
        ],
    }


def command_catalog(root: click.Command, prefix: list[str] | None = None) -> dict[str, Any]:
    """The catalog of ``root``: global options and every group and command, optionally
    only those under ``prefix`` (e.g. ``["detect"]``).

    Raises:
        click.UsageError: If ``prefix`` names no command.
    """
    from bioamla import __version__

    prefix = prefix or []
    node = root
    for name in prefix:
        if not isinstance(node, click.Group) or name not in node.commands:
            raise click.UsageError(f"No command {' '.join(prefix)!r}")
        node = node.commands[name]

    entries: list[dict[str, Any]] = []

    def walk(command: click.Command, path: list[str]) -> None:
        if getattr(command, "hidden", False) and path != prefix:  # named directly: describe it
            return
        entries.append(_entry(command, path))
        if isinstance(command, click.Group):
            for name in sorted(command.commands):
                walk(command.commands[name], [*path, name])

    if prefix:
        walk(node, prefix)
    else:
        for name in sorted(root.commands):
            walk(root.commands[name], [name])
    root_entry = _entry(root, [])
    return {
        "catalog_version": CATALOG_VERSION,
        "bioamla": __version__,
        "global_options": root_entry["options"],
        "commands": entries,
    }


__all__ = ["CATALOG_VERSION", "command_catalog"]

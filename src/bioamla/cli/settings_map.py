"""Turn settings files into Click option defaults (``default_map``).

A settings table mirrors a command path (``[detect.energy]``) and its keys name
options by their long name, with ``-`` or ``_`` (``threshold_db`` or
``threshold-db`` for ``--threshold-db``), or by the parameter name. Environment
variables name an option the same way: ``BIOAMLA_DETECT_ENERGY_THRESHOLD_DB`` for
``bioamla detect energy --threshold-db``.

Precedence, highest first: the command-line flag; the environment variable; the
workspace's config.toml; the user's; the command's built-in default.
"""

from __future__ import annotations

import logging
from typing import Any

import click

logger = logging.getLogger("bioamla.cli")


def _norm(name: str) -> str:
    return name.lstrip("-").replace("-", "_")


def is_settable(option: click.Parameter) -> bool:
    """Options a setting may give a default: not --help/--version-style eager flags,
    and not ones whose value the command never sees."""
    return isinstance(option, click.Option) and option.expose_value and not option.is_eager


def find_option(command: click.Command, key: str) -> click.Parameter | None:
    """The settable option of ``command`` that ``key`` names (long name or parameter
    name). Root options (``--json``, ``--log-level``, ...) are never settable: they're
    decided before settings are read."""
    if getattr(command, "is_bioamla_root", False):
        return None
    wanted = _norm(key)
    for param in command.params:
        if not is_settable(param):
            continue
        names = {param.name or ""} | {_norm(o) for o in param.opts if o.startswith("--")}
        if wanted in names:
            return param
    return None


def build_default_map(
    command: click.Command, settings: dict[str, Any], prefix: str = ""
) -> tuple[dict[str, Any], list[str]]:
    """``settings`` as a Click ``default_map`` for ``command``, plus the dotted keys
    that matched no command or option."""
    from bioamla.common.settings import RESERVED_SECTIONS

    default_map: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in settings.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if not prefix and key in RESERVED_SECTIONS:
            continue
        sub = command.commands.get(key) if isinstance(command, click.Group) else None
        if sub is not None:
            if isinstance(value, dict):
                sub_map, sub_unknown = build_default_map(sub, value, dotted)
                if sub_map:
                    default_map[key] = sub_map
                unknown += sub_unknown
            else:
                unknown.append(dotted)
            continue
        option = find_option(command, key)
        if option is None or isinstance(value, dict):
            unknown += [dotted] if not isinstance(value, dict) else [
                f"{dotted}.{k}" for k in _leaves(value)
            ]  # fmt: skip
            continue
        default_map[option.name] = _shape(option, value)
    return default_map, unknown


def _shape(option: click.Parameter, value: Any) -> Any:
    """A value in the shape Click wants for ``option``: a list for a multiple or
    multi-argument option (a single value becomes a one-item list)."""
    if (getattr(option, "multiple", False) or option.nargs != 1) and not isinstance(
        value, (list, tuple)
    ):
        if isinstance(value, str):
            return option.type.split_envvar_value(value) if option.nargs != 1 else [value]
        return [value]
    return value


def _leaves(table: dict[str, Any], prefix: str = "") -> list[str]:
    out = []
    for key, value in table.items():
        dotted = f"{prefix}.{key}" if prefix else key
        out += _leaves(value, dotted) if isinstance(value, dict) else [dotted]
    return out


def setting_key(path: list[str], option: click.Option) -> str:
    """The settings key for ``option`` of the command at ``path``:
    ``detect.energy.threshold_db`` for ``bioamla detect energy --threshold-db``."""
    longs = [o for o in option.opts if o.startswith("--")]
    return ".".join([*path, _norm(longs[0]) if longs else option.name or ""])


def canonical_key(root: click.Command, dotted: str) -> str:
    """``dotted`` with its option spelled as the long name with ``_``
    (``detect.energy.threshold-db`` -> ``detect.energy.threshold_db``); unchanged if it
    names no option."""
    from bioamla.exceptions import ConfigError

    try:
        _, option = resolve_key(root, dotted)
    except ConfigError:
        return dotted
    *path, _ = dotted.split(".")
    return setting_key(path, option)


def resolve_key(root: click.Command, dotted: str) -> tuple[click.Command, click.Parameter]:
    """The command and option a dotted settings key names.

    Raises:
        ConfigError: If it names no command option.
    """
    from bioamla.exceptions import ConfigError

    *path, key = dotted.split(".")
    command = root
    for name in path:
        sub = command.commands.get(name) if isinstance(command, click.Group) else None
        if sub is None:
            raise ConfigError(f"{dotted}: no command {' '.join(path)!r}")
        command = sub
    option = find_option(command, key)
    if option is None:
        where = " ".join(path) or "bioamla"
        raise ConfigError(f"{dotted}: `{where}` has no option {key!r}")
    return command, option


ENV_PREFIX = "BIOAMLA_"


def env_name(path: list[str], option: click.Option) -> str | None:
    """The environment variable for ``option`` of the command at ``path``."""
    longs = [o for o in option.opts if o.startswith("--")]
    if not longs:
        return None
    parts = [*path, _norm(longs[0])]
    return ENV_PREFIX + "_".join(p.replace("-", "_") for p in parts).upper()


def env_settings(root: click.Command, environ: dict[str, str] | None = None) -> dict[str, Any]:
    """``BIOAMLA_<COMMAND>_<OPTION>`` variables as a settings dict (values are strings;
    Click converts them like command-line values)."""
    import os

    environ = os.environ if environ is None else environ
    if not any(k.startswith(ENV_PREFIX) for k in environ):
        return {}  # the common case: skip walking the command tree
    settings: dict[str, Any] = {}

    def walk(command: click.Command, path: list[str], node: dict[str, Any]) -> None:
        for param in command.params if path else ():  # root options aren't settable
            if is_settable(param):
                name = env_name(path, param)
                if name and name in environ:
                    value = environ[name]
                    if getattr(param, "multiple", False) or param.nargs != 1:
                        value = param.type.split_envvar_value(value)  # as Click splits
                    node[param.name] = value
        if isinstance(command, click.Group):
            for sub_name, sub in command.commands.items():
                child: dict[str, Any] = {}
                walk(sub, [*path, sub_name], child)
                if child:
                    node[sub_name] = child

    walk(root, [], settings)
    return settings


def active_layers(start: str | None = None) -> list[tuple[Any, dict]]:
    """The settings files in effect from ``start`` (default: the current directory),
    lowest precedence first.

    Raises:
        ConfigError: If one isn't valid TOML.
    """
    from bioamla.common.settings import settings_layers, user_config_path

    workspace_config = _find_workspace_config(start)
    if workspace_config is None and not user_config_path().is_file():
        return []  # the common case: nothing to read
    return settings_layers(workspace_config)


def effective_settings(root: click.Command, layers: list[tuple[Any, dict]]) -> dict[str, dict]:
    """Canonical dotted key -> ``{"key", "value", "source"}`` for every setting in
    ``layers``, the later file winning (spellings of one option count as one key)."""
    from bioamla.common.settings import flatten

    effective: dict[str, dict] = {}
    for path, data in layers:
        for key, value in flatten(data).items():
            key = canonical_key(root, key)
            effective[key] = {"key": key, "value": value, "source": str(path)}
    return effective


def settings_default_map(root: click.Command, *, lenient: bool = False) -> dict[str, Any]:
    """User settings, then workspace settings, then ``BIOAMLA_*`` environment
    variables, merged into a ``default_map`` for ``root``.

    Unknown keys in a settings file are reported as a warning, naming the file.

    Args:
        lenient: Report an unreadable settings file as a warning and ignore it, rather
            than failing (for --help, --version and the commands that fix settings).

    Raises:
        ConfigError: If a settings file isn't valid TOML (unless ``lenient``).
    """
    from bioamla.common.settings import merge_settings
    from bioamla.exceptions import ConfigError

    try:
        layers = active_layers()
    except ConfigError as e:
        if not lenient:
            raise
        logger.warning("Ignoring settings: %s", e)
        layers = []
    # Map each file to options first, then merge: `threshold-db` in one file and
    # `threshold_db` in another name the same option, and the later file must win.
    maps = []
    for path, data in layers:
        default_map, unknown = build_default_map(root, data)
        for key in unknown:
            logger.warning("%s: unknown setting %r (no such command option)", path, key)
        maps.append(default_map)
    return merge_settings(*maps, env_settings(root))


def coerce_value(option: click.Parameter, text: str) -> Any:
    """A command-line string for ``option``, as the value to store in a settings file.

    The option's own type decides: an int option gets an int, a float option a float,
    a flag a bool; text and path options keep the string exactly (``1.10`` stays
    ``"1.10"``). A multiple or multi-argument option takes a TOML list
    (``'["a", "b"]'``) or a single value.

    Raises:
        ConfigError: If the value isn't valid for the option.
    """
    from bioamla.common.settings import parse_value
    from bioamla.exceptions import ConfigError

    def one(item: Any) -> Any:
        try:
            converted = option.type.convert(item, option, None)
        except click.BadParameter as e:
            raise ConfigError(f"{item!r} is not valid for {option.opts[0]}: {e.message}") from e
        if isinstance(converted, (bool, int, float, str)):
            return converted
        return str(converted)  # paths, choices with non-str values, ...

    if getattr(option, "multiple", False) or option.nargs != 1:
        parsed = parse_value(text)
        items = parsed if isinstance(parsed, list) else [text]
        return [one(i) for i in items]
    return one(text)


def _find_workspace_config(start: str | None = None):
    """The config.toml of the workspace around ``start``, if there is one."""
    from bioamla.exceptions import BioamlaError
    from bioamla.workspace import find_workspace

    try:
        ws = find_workspace(start)
    except BioamlaError as e:  # a broken workspace shouldn't block every other command
        logger.warning("Ignoring the workspace's settings: %s", e)
        return None
    if ws is None or not ws.config_path.is_file():
        return None
    return ws.config_path

"""
Config objects, validation, JSON Schema export, and TOML reading.

The option objects across bioamla (``BatchConfig``, ``ClusteringConfig``,
``EmbeddingConfig``, ...) are declared with :func:`config_dataclass`: ordinary
dataclasses (positional arguments, :func:`dataclasses.asdict`,
:func:`dataclasses.replace` all work) that pydantic validates on construction.
A wrong type or an unknown field raises
:class:`~bioamla.exceptions.ConfigValidationError` naming the field.

- :func:`validate_config` builds a config from a plain mapping, such as a parsed
  TOML table, and names the source file in its errors.
- :func:`config_schema` returns a config's JSON Schema, for editor completion and
  validation of config files. :data:`CONFIG_CLASSES` lists the exported configs.
- :func:`load_toml` reads a TOML file into a plain dict. Domain functions take
  explicit parameters and never read a settings file; the training command's
  ``--config`` file is the one consumer.

pydantic is imported when the first config class is defined, not when this module
loads, so modules that only need :func:`load_toml` stay light.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from bioamla.exceptions import ConfigError, ConfigValidationError, NotFoundError

if TYPE_CHECKING:
    from pydantic import ValidationError

T = TypeVar("T")

# Exported configs: schema name -> "module:attribute". Strings, so listing the
# names doesn't import the domain packages (several need an extra installed).
CONFIG_CLASSES: dict[str, str] = {
    "batch": "bioamla.batch:BatchConfig",
    "waveform-augmentation": "bioamla.datasets.augmentation:WaveformAugmentationConfig",
    "ast-train-file": "bioamla.ml.train_config:AstTrainConfigFile",
    "model": "bioamla.ml.base:ModelConfig",
    "inference": "bioamla.ml.ast:InferenceConfig",
    "embedding": "bioamla.ml.embedding:EmbeddingConfig",
    "spectrogram-augmentation": "bioamla.ml.preprocessing:SpectrogramAugmentationConfig",
    "reduction": "bioamla.cluster.core:ReductionConfig",
    "clustering": "bioamla.cluster.core:ClusteringConfig",
}


def _format_errors(error: ValidationError, prefix: str) -> str:
    """One line per pydantic error: ``prefix: field.path: message (got value)``."""
    lines = []
    for err in error.errors(include_url=False):
        loc = ".".join(str(part) for part in err["loc"]) or "(root)"
        if err["type"] in ("unexpected_keyword_argument", "extra_forbidden"):
            lines.append(f"{loc}: unknown field")
            continue
        detail = f"{loc}: {err['msg']}"
        if err["type"] != "missing":
            got = repr(err["input"])
            detail += f" (got {got if len(got) <= 60 else got[:57] + '...'})"
        lines.append(detail)
    return f"{prefix}:\n  " + "\n  ".join(lines)


def config_dataclass(cls: type[T]) -> type[T]:
    """Declare a validated config: a pydantic dataclass that rejects unknown fields.

    Use in place of ``@dataclass``. Construction validates (and coerces, e.g.
    ``"3"`` to ``3`` for an ``int`` field) every field; a failure raises
    :class:`~bioamla.exceptions.ConfigValidationError`. Errors raised by the class's own
    ``__post_init__`` propagate unchanged. Attribute assignment after
    construction is not validated.
    """
    from pydantic import ConfigDict, ValidationError
    from pydantic.dataclasses import dataclass as pydantic_dataclass

    model = pydantic_dataclass(cls, config=ConfigDict(extra="forbid"))
    init = model.__init__

    @functools.wraps(init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        try:
            init(self, *args, **kwargs)
        except ValidationError as e:
            raise ConfigValidationError(_format_errors(e, f"Invalid {model.__name__}")) from None

    model.__init__ = __init__  # type: ignore[method-assign]
    return model


def validate_config(cls: type[T], data: Mapping[str, Any], source: str | Path | None = None) -> T:
    """Build a config from a mapping, validating every field.

    Args:
        cls: A class declared with :func:`config_dataclass`.
        data: Field values, e.g. a table from :func:`load_toml`. Nested configs
            take nested mappings.
        source: The file the data came from, named in error messages.

    Returns:
        The validated config.

    Raises:
        ConfigValidationError: If a value has the wrong type or a key isn't a field. The
            message lists every problem with its dotted key path.
    """
    from pydantic import TypeAdapter, ValidationError

    private = sorted(k for k in data if str(k).startswith("_"))
    if private:
        where = f"config file {source}" if source else f"{cls.__name__} values"
        raise ConfigValidationError(
            f"Invalid {where}:\n  " + "\n  ".join(f"{k}: unknown field" for k in private)
        )
    try:
        return TypeAdapter(cls).validate_python(dict(data))
    except ValidationError as e:
        where = f"config file {source}" if source else f"{cls.__name__} values"
        raise ConfigValidationError(_format_errors(e, f"Invalid {where}")) from None


def resolve_config_class(name: str) -> type:
    """Return the config class registered under ``name`` in :data:`CONFIG_CLASSES`.

    Raises:
        ConfigError: If ``name`` isn't registered.
        DependencyError: If the class's package needs an extra that isn't installed.
    """
    import importlib

    try:
        target = CONFIG_CLASSES[name]
    except KeyError:
        known = ", ".join(sorted(CONFIG_CLASSES))
        raise ConfigError(f"Unknown config {name!r}. Known configs: {known}") from None
    module_name, attr = target.split(":")
    return getattr(importlib.import_module(module_name), attr)


def config_schema(config: str | type) -> dict[str, Any]:
    """Return the JSON Schema for a config class or a :data:`CONFIG_CLASSES` name.

    Private fields (names starting with ``_``) are left out.
    """
    from pydantic import TypeAdapter

    cls = resolve_config_class(config) if isinstance(config, str) else config
    schema = TypeAdapter(cls).json_schema()
    _drop_private_fields(schema)
    return schema


def _drop_private_fields(schema: dict[str, Any]) -> None:
    """Remove ``_``-prefixed properties from a schema and its ``$defs``, in place."""
    for node in [schema, *schema.get("$defs", {}).values()]:
        props = node.get("properties", {})
        for key in [k for k in props if k.startswith("_")]:
            del props[key]
        if "required" in node:
            node["required"] = [k for k in node["required"] if not k.startswith("_")]


# Use tomli for Python < 3.11, tomllib for Python >= 3.11.
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def load_toml(filepath: str | Path) -> dict[str, Any]:
    """Load a TOML file into a dictionary.

    Args:
        filepath: Path to the TOML file.

    Returns:
        Parsed configuration as a nested dict.

    Raises:
        NotFoundError: If the file doesn't exist.
        ConfigError: If TOML parsing fails or tomli is unavailable on < 3.11.
    """
    if tomllib is None:
        raise ConfigError(
            "TOML support requires tomli package for Python < 3.11. Install with: pip install tomli"
        )

    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Configuration file not found: {filepath}")

    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Failed to parse TOML config {filepath}: {e}") from e

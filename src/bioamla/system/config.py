"""Configuration management functions.

Folds the former ``services/config.py`` into plain raising functions that reuse
the :class:`bioamla.common.config.Config` machinery directly. Failures raise
:class:`~bioamla.exceptions.ConfigError`.
"""

from __future__ import annotations

from pathlib import Path

from bioamla.common.config import Config
from bioamla.common.config import create_default_config_file as _create_default_config_file
from bioamla.common.config import find_config_file as _find_config_file
from bioamla.common.config import get_config as _get_config
from bioamla.common.config import get_config_locations as _get_config_locations
from bioamla.common.config import get_default_config as _get_default_config
from bioamla.common.config import load_config as _load_config
from bioamla.exceptions import ConfigError, InvalidInputError


def get_config() -> Config:
    """Return the current (cached) configuration.

    Raises:
        ConfigError: If loading the configuration fails.
    """
    try:
        return _get_config()
    except Exception as e:
        raise ConfigError(f"Failed to get config: {e}") from e


def find_config_file(config_path: str | None = None) -> str | None:
    """Return the path to the active config file, or None if none is found.

    Raises:
        ConfigError: If the lookup fails.
    """
    try:
        result = _find_config_file(config_path)
        return str(result) if result else None
    except Exception as e:
        raise ConfigError(f"Failed to find config file: {e}") from e


def get_config_locations() -> list[str]:
    """Return the config-file search locations in priority order.

    Raises:
        ConfigError: If the lookup fails.
    """
    try:
        return [str(loc) for loc in _get_config_locations()]
    except Exception as e:
        raise ConfigError(f"Failed to get config locations: {e}") from e


def load_config(config_path: str | None = None) -> Config:
    """Load configuration from a file (or defaults).

    Raises:
        ConfigError: If loading fails.
    """
    try:
        return _load_config(config_path)
    except Exception as e:
        raise ConfigError(f"Failed to load config: {e}") from e


def get_default_config() -> Config:
    """Return the default configuration.

    Raises:
        ConfigError: If building the default config fails.
    """
    try:
        return _get_default_config()
    except Exception as e:
        raise ConfigError(f"Failed to get default config: {e}") from e


def create_default_config(filepath: str | None = None, force: bool = False) -> str:
    """Create a default configuration file.

    Args:
        filepath: Destination path (default: ``./bioamla.toml``).
        force: Overwrite an existing file.

    Returns:
        The created file path.

    Raises:
        InvalidInputError: If the file exists and ``force`` is False.
        ConfigError: If writing the file fails.
    """
    path = Path(filepath) if filepath else Path("bioamla.toml")
    if path.exists() and not force:
        raise InvalidInputError(f"File already exists: {path}. Use force=True to overwrite.")
    try:
        return _create_default_config_file(str(path))
    except Exception as e:
        raise ConfigError(f"Failed to create config file: {e}") from e


__all__ = [
    "Config",
    "get_config",
    "find_config_file",
    "get_config_locations",
    "load_config",
    "get_default_config",
    "create_default_config",
]

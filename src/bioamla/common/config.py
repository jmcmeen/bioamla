"""
TOML config-file reading.

A single helper for loading a TOML file into a plain dict. Domain functions take
explicit parameters with sensible defaults and never read a settings file; the
only consumer is ``bioamla models ast train --config <file>`` (see
``bioamla.cli.commands.models._apply_train_config``), which overlays a file's
``[models]``/``[training]``/``[augmentation]`` values onto the CLI flags.

Direct file I/O via :mod:`pathlib`; raises :class:`~bioamla.exceptions.ConfigError`
on parse failure and :class:`~bioamla.exceptions.NotFoundError` if the file is
missing.
"""

import sys
from pathlib import Path
from typing import Any

from bioamla.exceptions import ConfigError, NotFoundError

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

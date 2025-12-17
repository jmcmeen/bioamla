"""
Configuration Management
========================

This module provides TOML-based configuration file support for bioamla CLI.

Configuration files are searched in the following order:
1. Path specified via --config option
2. ./bioamla.toml (current directory)
3. ~/.config/bioamla/config.toml (user config)
4. /etc/bioamla/config.toml (system config)

Example configuration file (bioamla.toml):

    [audio]
    sample_rate = 16000
    mono = true

    [visualize]
    type = "mel"
    n_fft = 2048
    hop_length = 512
    cmap = "magma"
    dpi = 150

    [analysis]
    silence_threshold = -40

    [batch]
    recursive = true
    workers = 4

    [output]
    format = "wav"
    verbose = true
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Use tomli for Python < 3.11, tomllib for Python >= 3.11
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "audio": {
        "sample_rate": 16000,
        "mono": True,
    },
    "visualize": {
        "type": "mel",
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128,
        "cmap": "magma",
        "dpi": 150,
        "window": "hann",
    },
    "analysis": {
        "silence_threshold": -40,
        "min_silence_duration": 0.1,
    },
    "batch": {
        "recursive": True,
        "workers": 1,
        "verbose": True,
    },
    "output": {
        "format": "wav",
        "verbose": True,
    },
    "progress": {
        "enabled": True,
        "style": "rich",  # "rich", "simple", or "none"
    },
}

# Standard config file locations
CONFIG_LOCATIONS = [
    Path("bioamla.toml"),
    Path("~/.config/bioamla/config.toml").expanduser(),
    Path("/etc/bioamla/config.toml"),
]


@dataclass
class Config:
    """
    Configuration container for bioamla settings.

    Attributes:
        audio: Audio processing settings
        visualize: Visualization settings
        analysis: Analysis settings
        batch: Batch processing settings
        output: Output settings
        progress: Progress bar settings
        _source: Path to the config file that was loaded
    """
    audio: Dict[str, Any] = field(default_factory=dict)
    visualize: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)
    batch: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    progress: Dict[str, Any] = field(default_factory=dict)
    _source: Optional[str] = None

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        section_dict = getattr(self, section, {})
        return section_dict.get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value."""
        section_dict = getattr(self, section, None)
        if section_dict is not None:
            section_dict[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "audio": self.audio,
            "visualize": self.visualize,
            "analysis": self.analysis,
            "batch": self.batch,
            "output": self.output,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], source: Optional[str] = None) -> "Config":
        """Create Config from dictionary."""
        return cls(
            audio=data.get("audio", {}),
            visualize=data.get("visualize", {}),
            analysis=data.get("analysis", {}),
            batch=data.get("batch", {}),
            output=data.get("output", {}),
            progress=data.get("progress", {}),
            _source=source,
        )


def load_toml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a TOML configuration file.

    Args:
        filepath: Path to the TOML file

    Returns:
        Dictionary with configuration values

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If TOML parsing fails or tomli not installed
    """
    if tomllib is None:
        raise ValueError(
            "TOML support requires tomli package for Python < 3.11. "
            "Install with: pip install tomli"
        )

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(path, "rb") as f:
        return tomllib.load(f)


def save_toml(config: Dict[str, Any], filepath: Union[str, Path]) -> str:
    """
    Save configuration to a TOML file.

    Args:
        config: Configuration dictionary
        filepath: Path to save the file

    Returns:
        Path to the saved file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for section, values in config.items():
        if isinstance(values, dict) and values:
            lines.append(f"[{section}]")
            for key, value in values.items():
                if isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                elif isinstance(value, bool):
                    lines.append(f"{key} = {str(value).lower()}")
                elif isinstance(value, (int, float)):
                    lines.append(f"{key} = {value}")
                elif isinstance(value, list):
                    items = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in value)
                    lines.append(f"{key} = [{items}]")
            lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    return str(path)


def find_config_file(config_path: Optional[str] = None) -> Optional[Path]:
    """
    Find the configuration file to use.

    Args:
        config_path: Explicit path to config file (highest priority)

    Returns:
        Path to config file, or None if not found
    """
    # Check explicit path first
    if config_path:
        path = Path(config_path)
        if path.exists():
            return path
        logger.warning(f"Specified config file not found: {config_path}")
        return None

    # Check standard locations
    for location in CONFIG_LOCATIONS:
        if location.exists():
            return location

    return None


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Optional explicit path to config file

    Returns:
        Config object with merged settings
    """
    # Start with defaults
    config_data = _deep_copy_dict(DEFAULT_CONFIG)

    # Find and load config file
    config_file = find_config_file(config_path)

    if config_file:
        try:
            file_config = load_toml(config_file)
            # Merge file config into defaults
            config_data = _merge_dicts(config_data, file_config)
            logger.info(f"Loaded configuration from {config_file}")
            return Config.from_dict(config_data, source=str(config_file))
        except Exception as e:
            logger.warning(f"Error loading config file {config_file}: {e}")

    return Config.from_dict(config_data)


def get_default_config() -> Config:
    """Get the default configuration."""
    return Config.from_dict(_deep_copy_dict(DEFAULT_CONFIG))


def create_default_config_file(filepath: Optional[str] = None) -> str:
    """
    Create a default configuration file.

    Args:
        filepath: Path to create the file (default: ./bioamla.toml)

    Returns:
        Path to the created file
    """
    if filepath is None:
        filepath = "bioamla.toml"

    return save_toml(DEFAULT_CONFIG, filepath)


def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Create a deep copy of a dictionary."""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _deep_copy_dict(value)
        elif isinstance(value, list):
            result[key] = value.copy()
        else:
            result[key] = value
    return result


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, with override taking precedence."""
    result = _deep_copy_dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration to None (will reload on next access)."""
    global _global_config
    _global_config = None

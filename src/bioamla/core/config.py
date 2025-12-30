"""
Configuration Management
========================

This module provides TOML-based configuration file support for bioamla CLI.

Configuration files are searched in the following order (highest to lowest priority):
1. Path specified via --config option
2. Project config (.bioamla/config.toml) - if in a bioamla project
3. ./bioamla.toml (current directory)
4. ~/.config/bioamla/config.toml (user config)
5. /etc/bioamla/config.toml (system config)
6. Built-in defaults

Example configuration file (bioamla.toml):

    [project]
    name = "my-bioacoustics-study"
    version = "1.0.0"
    description = "Species identification project"

    [audio]
    sample_rate = 16000
    mono = true
    normalize = false
    filter_order = 5
    normalization_target_peak = 0.95
    min_segment_duration = 0.1
    noise_reduce_factor = 0.21
    noise_floor_percentile = 10
    rolloff_threshold = 0.85
    frame_length = 2048
    amplitude_to_db_max = 80.0

    [visualize]
    type = "mel"
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    cmap = "magma"
    dpi = 150
    window = "hann"
    figsize = [10, 4]

    [models]
    default_ast_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
    prediction_threshold = 0.5

    [models.birdnet]
    sample_rate = 48000
    min_confidence = 0.1

    [inference]
    batch_size = 8
    use_fp16 = false
    top_k = 5
    min_confidence = 0.01
    segment_duration = 10
    segment_overlap = 0

    [training]
    learning_rate = 5.0e-5
    epochs = 10
    batch_size = 16
    eval_strategy = "epoch"
    save_strategy = "epoch"

    [training.scheduler]
    warmup_ratio = 0.1
    weight_decay = 0.01

    [analysis]
    silence_threshold = -40
    min_silence_duration = 0.1

    [detection.energy]
    threshold = 0.02
    min_duration = 0.1
    frame_length = 2048
    hop_length = 512

    [detection.ribbit]
    signal_band = [1000, 2000]
    noise_band = [0, 500]
    pulse_rate_range = [5, 15]
    min_score = 0.5

    [detection.cwt]
    min_freq = 500
    max_freq = 8000
    wavelet = "morl"
    num_scales = 64

    [batch]
    recursive = true
    workers = 4

    [output]
    format = "csv"
    verbose = true
    overwrite = false

    [progress]
    enabled = true
    style = "rich"

    [logging]
    level = "WARNING"
    max_history = 1000
    rotate_size_mb = 10

    [api]
    timeout = 30
    large_download_timeout = 300
    download_chunk_size = 8192
    rate_limit_delay = 1.0

    [realtime]
    sample_rate = 16000
    channels = 1
    chunk_size = 1024
    buffer_seconds = 30
    format = "int16"
    thread_join_timeout = 2.0

    [augmentation]
    enabled = false

    [augmentation.clipping]
    threshold = 0.99
    margin = 0.01

    [active_learning]
    uncertainty_threshold = 0.3
    batch_size = 10
    strategy = "uncertainty"

    [execution]
    stateful = "auto"
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bioamla.core.files import BinaryFile, TextFile

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
    "project": {
        "name": "",
        "version": "1.0.0",
        "description": "",
    },
    "audio": {
        "sample_rate": 16000,
        "mono": True,
        "normalize": False,
        "filter_order": 5,
        "normalization_target_peak": 0.95,
        "min_segment_duration": 0.1,
        "noise_reduce_factor": 0.21,
        "noise_floor_percentile": 10,
        "rolloff_threshold": 0.85,
        "frame_length": 2048,
        "amplitude_to_db_max": 80.0,
    },
    "visualize": {
        "type": "mel",
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128,
        "cmap": "magma",
        "dpi": 150,
        "window": "hann",
        "figsize": [10, 4],
    },
    "models": {
        "default_ast_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "cache_dir": None,  # None = use HuggingFace default
        "prediction_threshold": 0.5,
        "birdnet": {
            "model_path": None,
            "labels_path": None,
            "sample_rate": 48000,
            "min_confidence": 0.1,
        },
    },
    "inference": {
        "batch_size": 8,
        "use_fp16": False,
        "top_k": 5,
        "min_confidence": 0.01,
        "segment_duration": 10,
        "segment_overlap": 0,
    },
    "training": {
        "learning_rate": 5.0e-5,
        "epochs": 10,
        "batch_size": 16,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "scheduler": {
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
        },
    },
    "analysis": {
        "silence_threshold": -40,
        "min_silence_duration": 0.1,
    },
    "detection": {
        "energy": {
            "threshold": 0.02,
            "min_duration": 0.1,
            "frame_length": 2048,
            "hop_length": 512,
        },
        "ribbit": {
            "signal_band": [1000, 2000],
            "noise_band": [0, 500],
            "pulse_rate_range": [5, 15],
            "min_score": 0.5,
        },
        "cwt": {
            "min_freq": 500,
            "max_freq": 8000,
            "wavelet": "morl",
            "num_scales": 64,
        },
    },
    "batch": {
        "recursive": True,
        "workers": 1,
        "verbose": True,
    },
    "output": {
        "format": "csv",
        "verbose": True,
        "overwrite": False,
    },
    "progress": {
        "enabled": True,
        "style": "rich",  # "rich", "simple", or "none"
    },
    "logging": {
        "level": "WARNING",
        "max_history": 1000,
        "rotate_size_mb": 10,
    },
    "api": {
        "xc_api_key": None,  # Xeno-canto API key (or set XC_API_KEY env var)
        "inat_api_token": None,  # iNaturalist API token (optional)
        "timeout": 30,
        "large_download_timeout": 300,
        "download_chunk_size": 8192,
        "rate_limit_delay": 1.0,
    },
    "realtime": {
        "sample_rate": 16000,
        "channels": 1,
        "chunk_size": 1024,
        "buffer_seconds": 30,
        "format": "int16",
        "thread_join_timeout": 2.0,
    },
    "augmentation": {
        "enabled": False,
        "clipping": {
            "threshold": 0.99,
            "margin": 0.01,
        },
    },
    "active_learning": {
        "uncertainty_threshold": 0.3,
        "batch_size": 10,
        "strategy": "uncertainty",  # "uncertainty", "diversity", "hybrid"
    },
    "execution": {
        "stateful": "auto",  # "auto", "true", "false"
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
        project: Project metadata
        audio: Audio processing settings
        visualize: Visualization settings
        models: Model configuration (AST, BirdNET)
        inference: Inference settings
        training: Training settings with scheduler options
        analysis: Analysis settings (silence detection, etc.)
        detection: Detection algorithm settings (energy, ribbit, cwt)
        batch: Batch processing settings
        output: Output settings
        progress: Progress bar settings
        logging: Logging settings
        api: API configuration (timeouts, rate limits)
        realtime: Realtime audio processing settings
        augmentation: Data augmentation settings
        active_learning: Active learning configuration
        execution: Execution mode settings (stateful/stateless)
        _source: Path to the config file that was loaded
    """

    project: Dict[str, Any] = field(default_factory=dict)
    audio: Dict[str, Any] = field(default_factory=dict)
    visualize: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    inference: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)
    detection: Dict[str, Any] = field(default_factory=dict)
    batch: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    progress: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    api: Dict[str, Any] = field(default_factory=dict)
    realtime: Dict[str, Any] = field(default_factory=dict)
    augmentation: Dict[str, Any] = field(default_factory=dict)
    active_learning: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    _source: Optional[str] = None

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        section_dict = getattr(self, section, {})
        if section_dict is None:
            return default
        return section_dict.get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value."""
        section_dict = getattr(self, section, None)
        if section_dict is not None:
            section_dict[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "project": self.project,
            "audio": self.audio,
            "visualize": self.visualize,
            "models": self.models,
            "inference": self.inference,
            "training": self.training,
            "analysis": self.analysis,
            "detection": self.detection,
            "batch": self.batch,
            "output": self.output,
            "progress": self.progress,
            "logging": self.logging,
            "api": self.api,
            "realtime": self.realtime,
            "augmentation": self.augmentation,
            "active_learning": self.active_learning,
            "execution": self.execution,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], source: Optional[str] = None) -> "Config":
        """Create Config from dictionary."""
        return cls(
            project=data.get("project", {}),
            audio=data.get("audio", {}),
            visualize=data.get("visualize", {}),
            models=data.get("models", {}),
            inference=data.get("inference", {}),
            training=data.get("training", {}),
            analysis=data.get("analysis", {}),
            detection=data.get("detection", {}),
            batch=data.get("batch", {}),
            output=data.get("output", {}),
            progress=data.get("progress", {}),
            logging=data.get("logging", {}),
            api=data.get("api", {}),
            realtime=data.get("realtime", {}),
            augmentation=data.get("augmentation", {}),
            active_learning=data.get("active_learning", {}),
            execution=data.get("execution", {}),
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
            "TOML support requires tomli package for Python < 3.11. Install with: pip install tomli"
        )

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with BinaryFile(path, mode="rb") as f:
        return tomllib.load(f.handle)


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

    with TextFile(path, mode="w") as f:
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
            # Migrate any deprecated keys in file config before merging
            file_config = _migrate_deprecated_keys(file_config)
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


# Deprecated config key mappings: (section, old_key) -> new_key
_DEPRECATED_KEYS = {
    ("models", "default_model"): "default_ast_model",
}


def _migrate_deprecated_keys(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate deprecated config keys to their new names.

    If a deprecated key is found, it will be renamed to the new key
    (unless the new key already exists) and a warning will be logged.

    Args:
        config_data: Configuration dictionary to migrate

    Returns:
        Migrated configuration dictionary
    """
    import warnings

    for (section, old_key), new_key in _DEPRECATED_KEYS.items():
        if section in config_data and isinstance(config_data[section], dict):
            section_data = config_data[section]
            if old_key in section_data:
                if new_key not in section_data:
                    # Migrate the value to the new key
                    section_data[new_key] = section_data[old_key]
                    warnings.warn(
                        f"Config key '[{section}].{old_key}' is deprecated. "
                        f"Please update to '[{section}].{new_key}'.",
                        DeprecationWarning,
                        stacklevel=4,
                    )
                else:
                    # Both keys exist, prefer new key but warn
                    warnings.warn(
                        f"Config key '[{section}].{old_key}' is deprecated and "
                        f"'[{section}].{new_key}' is also present. "
                        f"Using '[{section}].{new_key}'. Please remove the deprecated key.",
                        DeprecationWarning,
                        stacklevel=4,
                    )
                # Remove the old key
                del section_data[old_key]

    return config_data


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config_cascade()
    return _global_config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration to None (will reload on next access)."""
    global _global_config
    _global_config = None


def get_config_locations() -> List[Path]:
    """
    Get configuration file search locations in priority order.

    Returns:
        List of paths to search, in priority order (highest first)
    """
    return CONFIG_LOCATIONS.copy()


def load_config_cascade(
    explicit_path: Optional[str] = None,
) -> Config:
    """
    Load configuration with full cascade support.

    Merges configs from all levels in priority order:
    defaults -> system -> user -> current dir -> explicit

    Higher priority configs override lower priority ones.

    Args:
        explicit_path: Explicit config file path (highest priority)

    Returns:
        Config object with merged settings from all sources
    """
    # Start with defaults
    config_data = _deep_copy_dict(DEFAULT_CONFIG)
    source = "defaults"

    # Get all config locations (in priority order: highest to lowest)
    locations = get_config_locations()

    # Load in reverse order (lowest to highest priority) so higher overrides lower
    for location in reversed(locations):
        if location.exists():
            try:
                file_config = load_toml(location)
                # Migrate any deprecated keys in file config before merging
                file_config = _migrate_deprecated_keys(file_config)
                config_data = _merge_dicts(config_data, file_config)
                source = str(location)
                logger.debug(f"Merged configuration from {location}")
            except Exception as e:
                logger.warning(f"Error loading {location}: {e}")

    # Explicit path has highest priority
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            try:
                file_config = load_toml(path)
                # Migrate any deprecated keys in file config before merging
                file_config = _migrate_deprecated_keys(file_config)
                config_data = _merge_dicts(config_data, file_config)
                source = str(path)
                logger.debug(f"Merged configuration from {path}")
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
        else:
            logger.warning(f"Specified config file not found: {explicit_path}")

    return Config.from_dict(config_data, source=source)



# services/config.py
"""
Service for configuration management operations.
"""

from pathlib import Path
from typing import Any, List, Optional

from .base import BaseService, ServiceResult


class ConfigService(BaseService):
    """
    Service for configuration management operations.

    Provides ServiceResult-wrapped methods for configuration access
    and management.
    """

    def get_config(self) -> ServiceResult[Any]:
        """
        Get the current configuration.

        Returns:
            ServiceResult containing the Config object on success
        """
        try:
            from bioamla.core.config import get_config as core_get_config

            config = core_get_config()

            return ServiceResult.ok(
                data=config,
                message=f"Loaded config from {config._source or 'defaults'}",
                source=config._source,
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to get config: {e}")

    def find_config_file(
        self,
        config_path: Optional[str] = None,
    ) -> ServiceResult[Optional[str]]:
        """
        Find the configuration file to use.

        Args:
            config_path: Explicit path to check first

        Returns:
            ServiceResult containing the config file path or None
        """
        try:
            from bioamla.core.config import find_config_file as core_find

            result = core_find(config_path)

            if result:
                return ServiceResult.ok(
                    data=str(result),
                    message=f"Found config file: {result}",
                )
            else:
                return ServiceResult.ok(
                    data=None,
                    message="No config file found",
                )
        except Exception as e:
            return ServiceResult.fail(f"Failed to find config file: {e}")

    def get_config_locations(
        self,
        include_project: bool = True,
    ) -> ServiceResult[List[str]]:
        """
        Get configuration file search locations in priority order.

        Args:
            include_project: Whether to include project config location

        Returns:
            ServiceResult containing list of path strings
        """
        try:
            from bioamla.core.config import (
                get_config_locations as core_get_locations,
            )

            locations = core_get_locations(include_project=include_project)
            paths = [str(loc) for loc in locations]

            return ServiceResult.ok(
                data=paths,
                message=f"Found {len(paths)} config locations",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to get config locations: {e}")

    def create_default_config(
        self,
        filepath: Optional[str] = None,
        force: bool = False,
    ) -> ServiceResult[str]:
        """
        Create a default configuration file.

        Args:
            filepath: Path to create the file (default: ./bioamla.toml)
            force: Overwrite if file exists

        Returns:
            ServiceResult containing the created file path on success
        """
        try:
            from bioamla.core.config import create_default_config_file

            path = Path(filepath) if filepath else Path("bioamla.toml")

            if path.exists() and not force:
                return ServiceResult.fail(
                    f"File already exists: {path}. Use force=True to overwrite."
                )

            result_path = create_default_config_file(str(path))

            return ServiceResult.ok(
                data=result_path,
                message=f"Created config file: {result_path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to create config file: {e}")

    def load_config(
        self,
        config_path: Optional[str] = None,
    ) -> ServiceResult[Any]:
        """
        Load configuration from file or use defaults.

        Args:
            config_path: Optional explicit path to config file

        Returns:
            ServiceResult containing the Config object on success
        """
        try:
            from bioamla.core.config import load_config as core_load

            config = core_load(config_path)

            return ServiceResult.ok(
                data=config,
                message=f"Loaded config from {config._source or 'defaults'}",
                source=config._source,
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to load config: {e}")

    def get_default_config(self) -> ServiceResult[Any]:
        """
        Get the default configuration.

        Returns:
            ServiceResult containing the default Config object
        """
        try:
            from bioamla.core.config import get_default_config as core_get_default

            config = core_get_default()

            return ServiceResult.ok(
                data=config,
                message="Loaded default config",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to get default config: {e}")


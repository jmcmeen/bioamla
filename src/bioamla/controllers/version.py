# src/bioamla/controllers/version.py
# This module provides functions to retrieve the current version of the bioamla package and other installed packages.

def get_bioamla_version() -> str:
    """
    Returns the current version of the bioamla package.
    """
    import importlib.metadata
    return importlib.metadata.version('bioamla')

def get_package_versions() -> dict:
    """
    Returns a dictionary of all installed packages and their versions.
    """
    import importlib.metadata
    return {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
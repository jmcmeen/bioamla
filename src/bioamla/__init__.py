"""
BioAmla - Bioacoustics & Machine Learning Applications
======================================================
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bioamla")
except PackageNotFoundError:  # package not installed (e.g. source checkout)
    __version__ = "0.0.0"

# Re-export commonly used utilities for convenience
from bioamla.core.utils import (
    SUPPORTED_AUDIO_EXTENSIONS,
    create_directory,
    directory_exists,
    file_exists,
    get_audio_files,
    get_files_by_extension,
)

__all__ = [
    "__version__",
    # Utilities
    "SUPPORTED_AUDIO_EXTENSIONS",
    "get_audio_files",
    "get_files_by_extension",
    "file_exists",
    "directory_exists",
    "create_directory",
]

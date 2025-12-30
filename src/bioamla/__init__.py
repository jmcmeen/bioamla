"""
BioAmla - Bioacoustics & Machine Learning Applications
======================================================

Version: 0.1.8
"""

__version__ = "0.1.8"

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

"""
Global Constants and Configurations
===================================

This module defines global constants and configuration values used throughout
the bioamla package. These include supported file formats, default parameters,
and other package-wide settings.

Usage:
    from bioamla.core.globals import SUPPORTED_AUDIO_EXTENSIONS
"""

from typing import List

# Supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS: List[str] = [
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
]

# Default audio parameters
DEFAULT_SAMPLE_RATE: int = 44100
DEFAULT_CHANNELS: int = 1
DEFAULT_BIT_DEPTH: int = 16

# Default analysis parameters
DEFAULT_N_FFT: int = 2048
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_WIN_LENGTH: int = 2048

__all__ = [
    "SUPPORTED_AUDIO_EXTENSIONS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS",
    "DEFAULT_BIT_DEPTH",
    "DEFAULT_N_FFT",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_WIN_LENGTH",
]

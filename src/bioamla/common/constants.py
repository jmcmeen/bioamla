"""
Constants and Configuration Defaults
=====================================

Pure audio/processing constants and default configuration values used
throughout the bioamla package.

This module is dependency-free: it intentionally does NOT re-export the HTTP
client machinery (that lives in :mod:`bioamla.common.http`).

Usage:
    from bioamla.common.constants import (
        SUPPORTED_AUDIO_EXTENSIONS,
        DEFAULT_SAMPLE_RATE,
        DefaultConfig,
    )
"""

from typing import List

# =============================================================================
# Audio Format Constants
# =============================================================================

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


# =============================================================================
# Default Configuration
# =============================================================================


class DefaultConfig:
    """
    Default configuration parameters for bioamla audio processing.

    Attributes:
        MODEL_NAME (str): Default Hugging Face model identifier for AST
        SAMPLE_RATE (int): Standard sample rate for audio processing (Hz)
        MAX_AUDIO_LENGTH (int): Maximum audio length for processing (seconds)
        MIN_CONFIDENCE (float): Minimum confidence threshold for predictions
        TOP_K (int): Default number of top predictions to return
    """

    MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 30  # seconds
    MIN_CONFIDENCE = 0.01
    TOP_K = 5


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Audio constants
    "SUPPORTED_AUDIO_EXTENSIONS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS",
    "DEFAULT_BIT_DEPTH",
    "DEFAULT_N_FFT",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_WIN_LENGTH",
    # Configuration
    "DefaultConfig",
]

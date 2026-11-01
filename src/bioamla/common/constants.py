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


# =============================================================================
# Audio Format Constants
# =============================================================================

# Supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS: list[str] = [
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

# Pictures only (`audio visualize`, `batch audio visualize`, bioamla.viz): one set of
# defaults, so the same file gives the same picture whichever way it's drawn. Feature
# analysis has its own (bioamla.audio.FrameFeatureConfig).
PICTURE_N_MELS: int = 128
PICTURE_N_MFCC: int = 40
PICTURE_CMAP: str = "magma"
PICTURE_DPI: int = 150

# Annotation file formats read and written by bioamla.datasets. Kept here, not in
# the datasets package, so the CLI can build its --format choices without
# importing that package (and numpy/pandas with it) at startup.
ANNOTATION_FORMATS: list[str] = ["raven", "csv", "bioamla", "arbimon"]


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
    "ANNOTATION_FORMATS",
    # Audio constants
    "SUPPORTED_AUDIO_EXTENSIONS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS",
    "DEFAULT_BIT_DEPTH",
    "DEFAULT_N_FFT",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_WIN_LENGTH",
    "PICTURE_N_MELS",
    "PICTURE_N_MFCC",
    "PICTURE_CMAP",
    "PICTURE_DPI",
    # Configuration
    "DefaultConfig",
]

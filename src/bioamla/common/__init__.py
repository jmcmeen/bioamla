"""
bioamla.common — dependency-free shared internals.

This package holds the pure-stdlib building blocks used across the domain
packages: audio/config constants, file helpers, batch-progress utilities,
TOML configuration, and the HTTP client machinery. Nothing here imports a
heavy/optional dependency at module load time.
"""

from bioamla.common.constants import (
    DEFAULT_BIT_DEPTH,
    DEFAULT_CHANNELS,
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WIN_LENGTH,
    SUPPORTED_AUDIO_EXTENSIONS,
    DefaultConfig,
)

__all__ = [
    "SUPPORTED_AUDIO_EXTENSIONS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS",
    "DEFAULT_BIT_DEPTH",
    "DEFAULT_N_FFT",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_WIN_LENGTH",
    "DefaultConfig",
]

"""Pydub adapter for audio I/O operations.

This adapter provides a unified interface for loading, saving, and getting
metadata from audio files using pydub/ffmpeg as the backend.

Only the services layer should import from this module - core code should
not depend on it.
"""

from bioamla.adapters.pydub.audio import (
    PydubAudioAdapter,
    get_audio_info,
    load_audio,
    save_audio,
)

__all__ = [
    "PydubAudioAdapter",
    "load_audio",
    "save_audio",
    "get_audio_info",
]

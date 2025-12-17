"""
Utility Functions Package
=========================

This package provides utility functions for file operations, audio processing,
compression, and audio playback. The functionality is organized into submodules:

- files: File system operations (get_files_by_extension, create_directory, etc.)
- audio: Audio file operations (get_audio_files, get_wav_metadata, format conversion)
- compression: ZIP archive operations (extract_zip_file, create_zip_file, zip_directory)
- playback: Audio playback with sounddevice (play, pause, stop, seek)

All public functions are re-exported from this module for convenience.
"""

# File operations
from bioamla.utils.files import (
    get_files_by_extension,
    create_directory,
    file_exists,
    directory_exists,
    download_file,
)

# Audio file operations and constants
from bioamla.utils.audio import (
    SUPPORTED_AUDIO_EXTENSIONS,
    get_audio_files,
    get_wav_metadata,
    load_audio,
    save_audio,
    to_mono,
    concatenate_audio,
    loop_audio,
    # Format conversion functions
    wav_to_mp3,
    mp3_to_wav,
    wav_to_flac,
    flac_to_wav,
    wav_to_ogg,
    ogg_to_wav,
    m4a_to_wav,
    wav_to_m4a,
    mp3_to_flac,
    flac_to_mp3,
    m4a_to_mp3,
    mp3_to_ogg,
    ogg_to_mp3,
    flac_to_ogg,
    ogg_to_flac,
    m4a_to_flac,
    m4a_to_ogg,
    aac_to_wav,
    aac_to_mp3,
    wma_to_wav,
    wma_to_mp3,
)

# Compression operations
from bioamla.utils.compression import (
    extract_zip_file,
    create_zip_file,
    zip_directory,
)

# Audio playback
from bioamla.utils.playback import (
    AudioPlayer,
    play_audio,
    stop_audio,
)

__all__ = [
    # File operations
    "get_files_by_extension",
    "create_directory",
    "file_exists",
    "directory_exists",
    "download_file",
    # Audio constants and operations
    "SUPPORTED_AUDIO_EXTENSIONS",
    "get_audio_files",
    "get_wav_metadata",
    "load_audio",
    "save_audio",
    "to_mono",
    "concatenate_audio",
    "loop_audio",
    # Format conversion
    "wav_to_mp3",
    "mp3_to_wav",
    "wav_to_flac",
    "flac_to_wav",
    "wav_to_ogg",
    "ogg_to_wav",
    "m4a_to_wav",
    "wav_to_m4a",
    "mp3_to_flac",
    "flac_to_mp3",
    "m4a_to_mp3",
    "mp3_to_ogg",
    "ogg_to_mp3",
    "flac_to_ogg",
    "ogg_to_flac",
    "m4a_to_flac",
    "m4a_to_ogg",
    "aac_to_wav",
    "aac_to_mp3",
    "wma_to_wav",
    "wma_to_mp3",
    # Compression
    "extract_zip_file",
    "create_zip_file",
    "zip_directory",
    # Playback
    "AudioPlayer",
    "play_audio",
    "stop_audio",
]

"""
Audio file management controller for the bioamla package.

This module provides functionality for discovering and managing audio files
within directory structures. It serves as a wrapper around the novus_pytils
audio file utilities, applying bioamla-specific configurations and supported
audio format filtering.

The module integrates with bioamla's global configuration to ensure only
supported audio file formats are processed, maintaining consistency across
the entire bioamla ecosystem.
"""

from typing import List
from novus_pytils.audio.files import get_audio_files
from bioamla.core.globals import SUPPORTED_AUDIO_EXTENSIONS


def get_audio_files_from_directory(directory: str) -> List[str]:
    """
    Get a list of audio files from the specified directory.
    
    Recursively searches through the provided directory and its subdirectories
    to locate all audio files that match the supported audio format extensions
    defined in bioamla's global configuration. This function acts as a bioamla-
    specific wrapper around the novus_pytils audio file discovery functionality.
    
    The search operation respects the SUPPORTED_AUDIO_EXTENSIONS configuration,
    ensuring that only audio files in formats compatible with bioamla's processing
    pipeline are included in the results.

    Args:
        directory (str): The absolute or relative path to the directory to search
                        for audio files. The search includes all subdirectories
                        recursively.

    Returns:
        List[str]: A list of absolute file paths to discovered audio files.
                  Returns an empty list if no audio files are found or if the
                  directory does not exist.
                  
    Raises:
        OSError: If the directory path is invalid or inaccessible.
        PermissionError: If there are insufficient permissions to access
                        the directory or its contents.
        
    Example:
        >>> audio_files = get_audio_files_from_directory("/path/to/audio")
        >>> print(audio_files)
        ['/path/to/audio/song1.mp3', '/path/to/audio/song2.wav']
    """
    return get_audio_files(directory, SUPPORTED_AUDIO_EXTENSIONS)
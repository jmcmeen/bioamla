from novus_pytils.audio.files import get_audio_files
from bioamla.core.globalsglobals import SUPPORTED_AUDIO_EXTENSIONS

def get_audio_files_from_directory(directory: str) -> list:
    """
    Get a list of audio files from the specified directory.

    Args:
        directory (str): The path to the directory to search for audio files.

    Returns:
        list: A list of audio file paths.
    """
    return get_audio_files(directory)
"""
Audio Format Conversion Module

This module provides utilities for converting audio files between different formats.
It leverages the pydub library for format conversion and includes functionality for
handling output directories, file naming, and directory management.

The module is designed to simplify common audio conversion tasks while providing
flexibility in output configuration and file organization.
"""

import os
from typing import Optional
from pydub import AudioSegment
from novus_pytils.files import get_file_name, get_file_directory, directory_exists, create_directory

def convert_wav(
    file_path: str, 
    format: str, 
    extension: str, 
    output_dir: Optional[str] = None, 
    new_file_name: Optional[str] = None
) -> None:
    """
    Convert a WAV file to another audio format.
    
    This function converts a WAV audio file to a specified format using pydub.
    It provides flexible options for output directory and file naming, with
    sensible defaults when not specified.
    
    Args:
        file_path (str): Path to the input WAV file to convert
        format (str): Target audio format (e.g., 'mp3', 'flac', 'ogg', 'm4a')
        extension (str): File extension for the output file (should include the dot, e.g., '.mp3')
        output_dir (Optional[str]): Output directory path. If None, uses the same directory 
                                   as the input file (default: None)
        new_file_name (Optional[str]): Base name for the output file without extension.
                                      If None, uses the input file's name (default: None)
    
    Raises:
        FileNotFoundError: If the input WAV file does not exist
        PermissionError: If unable to create output directory or write output file
        pydub.exceptions.CouldntDecodeError: If the input file is not a valid WAV file
        Exception: For other conversion errors
    
    Example:
        >>> # Convert WAV to MP3 in same directory
        >>> convert_wav('audio.wav', 'mp3', '.mp3')
        
        >>> # Convert with custom output directory and filename
        >>> convert_wav(
        ...     'input/song.wav', 
        ...     'flac', 
        ...     '.flac',
        ...     output_dir='output',
        ...     new_file_name='converted_song'
        ... )
    
    Note:
        The function automatically creates the output directory if it doesn't exist.
        The pydub library must have the appropriate codecs installed for the target format.
    """
    # Determine output directory: use input file's directory if not specified
    if output_dir is None:
        output_dir = get_file_directory(file_path)

    # Determine output filename: use input file's name if not specified
    if new_file_name is None:
        # Extract original filename and replace .wav extension
        new_file_name = get_file_name(file_path.replace(".wav", extension))
    else:
        # Append the specified extension to the custom filename
        new_file_name = new_file_name + extension

    # Create output directory if it doesn't exist
    if not directory_exists(output_dir):
        create_directory(output_dir)

    # Perform the conversion using pydub
    # Load WAV file and export to target format
    output_path = os.path.join(output_dir, new_file_name)
    AudioSegment.from_wav(file_path).export(output_path, format=format)

def convert_audio_format(
    input_path: str,
    output_path: str,
    target_format: str,
    bitrate: Optional[str] = None,
    sample_rate: Optional[int] = None
) -> None:
    """
    Convert audio file to a different format with optional quality settings.
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path for output file
        target_format (str): Target format (mp3, flac, wav, etc.)
        bitrate (Optional[str]): Target bitrate (e.g., '128k', '320k')
        sample_rate (Optional[int]): Target sample rate in Hz
    """
    # Load audio file (pydub can handle many formats)
    audio = AudioSegment.from_file(input_path)
    
    # Apply sample rate conversion if specified
    if sample_rate and audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)
    
    # Set up export parameters
    export_params = {"format": target_format}
    if bitrate:
        export_params["bitrate"] = bitrate
    
    # Export to target format
    audio.export(output_path, **export_params)

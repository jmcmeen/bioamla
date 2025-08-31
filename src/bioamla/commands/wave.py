"""
WAV File Metadata Extraction Command
====================================

Command-line tool for extracting metadata from individual WAV audio files.
This utility provides detailed information about WAV file properties including
audio characteristics, file structure, and technical specifications.

Usage:
    wave FILEPATH

Examples:
    wave audio.wav                   # Extract metadata from audio.wav
    wave /path/to/file.wav          # Extract metadata from file at absolute path
    wave ./music/song.wav           # Extract metadata from relative path

Features:
    - Complete WAV file metadata extraction
    - Audio property analysis (sample rate, channels, duration, etc.)
    - File format and structure information
    - Technical specifications display
    - Error handling for invalid or corrupted files
"""

import click
from novus_pytils.wave import get_wav_file_metadata

@click.command()
@click.argument('filepath')
def main(filepath: str):
    """
    Extract and display metadata from a WAV audio file.
    
    Analyzes the specified WAV file and extracts comprehensive metadata
    including audio properties, file characteristics, and technical details.
    
    Args:
        filepath (str): Path to the WAV file to analyze
    """
    metadata = get_wav_file_metadata(filepath)
    click.echo(f"{metadata}")

if __name__ == '__main__':
    main()
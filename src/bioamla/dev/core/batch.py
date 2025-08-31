"""
Batch Audio Processing Module

This module provides utilities for batch processing of audio files, particularly
WAV files. It enables efficient metadata extraction and analysis of multiple
audio files within a directory structure.

The module leverages novus_pytils for low-level audio file operations and
returns results in convenient pandas DataFrame format for further analysis
and processing workflows.
"""

import pandas as pd
from typing import List, Dict, Any, Union
from pathlib import Path
from novus_pytils.audio.wave import get_wav_files_metadata, get_wav_files

def get_wav_file_frame(dir: Union[str, Path]) -> pd.DataFrame:
    """
    Extract metadata from all WAV files in a directory and return as DataFrame.
    
    This function recursively scans the specified directory for WAV files,
    extracts their metadata (including file size, audio properties, and duration),
    and returns the information in a structured pandas DataFrame suitable for
    analysis and processing.
    
    Args:
        dir (Union[str, Path]): Path to directory containing WAV files.
                               Can be a string path or pathlib.Path object.
    
    Returns:
        pd.DataFrame: DataFrame containing WAV file metadata with columns:
            - filepath (str): Full path to the WAV file
            - file_size (int): File size in bytes
            - num_channels (int): Number of audio channels (1=mono, 2=stereo)
            - sample_width (int): Sample width in bytes (typically 2 for 16-bit)
            - frame_rate (int): Sample rate in Hz (e.g., 44100, 48000)
            - num_frames (int): Total number of audio frames
            - duration (float): Duration in seconds
    
    Raises:
        FileNotFoundError: If the specified directory does not exist
        PermissionError: If the directory cannot be accessed
        ValueError: If no WAV files are found in the directory
    
    Example:
        >>> df = get_wav_file_frame('/path/to/audio/files')
        >>> print(df.head())
        >>> # Analyze average duration
        >>> print(f"Average duration: {df['duration'].mean():.2f} seconds")
        >>> # Find files with specific sample rate
        >>> cd_quality = df[df['frame_rate'] == 44100]
    
    Note:
        This function processes all WAV files found recursively within the
        directory structure. For large directories with many files, this
        operation may take some time to complete.
    """
    # Get list of WAV files in the directory
    wav_files = get_wav_files(str(dir))
    
    # Extract metadata for all WAV files
    wav_files_metadata = get_wav_files_metadata(wav_files)
    
    # Create DataFrame with structured column names
    df = pd.DataFrame(
        wav_files_metadata, 
        columns=[
            'filepath', 'file_size', 'num_channels', 'sample_width', 
            'frame_rate', 'num_frames', 'duration'
        ]
    )
    
    return df

def analyze_batch_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze statistical properties of a batch of WAV files.
    
    Args:
        df (pd.DataFrame): DataFrame from get_wav_file_frame()
    
    Returns:
        Dict[str, Any]: Dictionary containing statistical analysis
    """
    if df.empty:
        return {"error": "No data to analyze"}
    
    stats = {
        "total_files": len(df),
        "total_duration": df['duration'].sum(),
        "average_duration": df['duration'].mean(),
        "min_duration": df['duration'].min(),
        "max_duration": df['duration'].max(),
        "total_size_mb": df['file_size'].sum() / (1024 * 1024),
        "sample_rates": df['frame_rate'].value_counts().to_dict(),
        "channel_distribution": df['num_channels'].value_counts().to_dict(),
        "sample_width_distribution": df['sample_width'].value_counts().to_dict()
    }
    
    return stats



"""
Dataset Management and Validation
=================================

This module provides utilities for managing and validating audio datasets.
It includes functions for counting audio files, validating metadata consistency,
and loading datasets for machine learning workflows.

These utilities are essential for ensuring data quality and consistency
in bioacoustic machine learning projects.
"""

import pandas as pd
import os
from novus_pytils.audio import get_audio_files
from bioamla.core.globals import SUPPORTED_AUDIO_EXTENSIONS


def count_audio_files(audio_folder_path: str) -> int:
    """
    Count the number of audio files in a directory.
    
    This function scans a directory and counts all files with supported
    audio extensions as defined in the global configuration.
    
    Args:
        audio_folder_path (str): Path to the directory containing audio files
        
    Returns:
        int: Number of audio files found in the directory
    """
    audio_files = get_audio_files(audio_folder_path, SUPPORTED_AUDIO_EXTENSIONS)
    return len(audio_files)

def validate_metadata(audio_folder_path: str, metadata_csv_filename: str = 'metadata.csv') -> bool:
    """
    Validate that metadata CSV file matches the audio files in a directory.
    
    This function performs several validation checks to ensure consistency
    between audio files and their corresponding metadata:
    1. Number of audio files matches number of metadata entries
    2. All audio files are referenced in the metadata
    
    Args:
        audio_folder_path (str): Path to directory containing audio files
        metadata_csv_filename (str): Name of the metadata CSV file (default: 'metadata.csv')
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails (file count mismatch or missing references)
    """
    metadata_df = pd.read_csv(os.path.join(audio_folder_path, metadata_csv_filename))

    # Check that the audio folder contains the same number of files as the metadata.csv file
    num_audio_files = count_audio_files(audio_folder_path)
    num_metadata_files = len(metadata_df)
    if num_audio_files != num_metadata_files:
        raise ValueError(f"The number of audio files in the audio folder ({num_audio_files}) does not match the number of files in the metadata.csv file ({num_metadata_files})")

    # Check that all audio files are in metadata
    audio_files = get_audio_files(audio_folder_path, SUPPORTED_AUDIO_EXTENSIONS)
    for audio_file in audio_files:
        if audio_file not in metadata_df['filename'].tolist():
            raise ValueError(f"The audio file {audio_file} is not in the metadata.csv file")

    return True

def load_local_dataset(audio_folder_path: str):
    """
    Load a local audio dataset using Hugging Face datasets library.
    
    This function loads audio files from a local directory into a Hugging Face
    Dataset object for use in machine learning workflows. It performs basic
    validation to ensure the directory exists and contains audio files.
    
    Args:
        audio_folder_path (str): Path to directory containing audio files
        
    Returns:
        Dataset: Hugging Face Dataset object containing the audio data
        
    Raises:
        ValueError: If directory doesn't exist or contains no audio files
    """
    from datasets import load_dataset
    from novus_pytils.files import directory_exists
    
    if not directory_exists(audio_folder_path):
        raise ValueError(f"The audio folder {audio_folder_path} does not exist")
    if count_audio_files(audio_folder_path) == 0:
        raise ValueError(f"The audio folder {audio_folder_path} is empty")
    
    dataset = load_dataset(audio_folder_path)
    
    return dataset
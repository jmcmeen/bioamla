"""
Novus Audio Processing Test Script

This module provides a test script for processing audio datasets using the novus_pytils library.
It demonstrates a complete audio preprocessing pipeline including dataset copying, resampling,
filtering, segmentation, and metadata extraction.

The script performs the following operations in sequence:
1. Copy a source audio dataset to a new location
2. Resample all wave files to a standard sample rate
3. Apply bandpass filtering to remove noise outside speech frequency range
4. Split audio files into smaller clips for training
5. Extract metadata and create a CSV file for the processed dataset
"""

import os
from typing import Optional
from novus_pytils.files import copy_dataset
from novus_pytils.audio.files import resample_wave_files, bandpass_wave_files, split_wave_files
from novus_pytils.data import extract_partitioned_dataset

def main() -> None:
    """
    Execute the complete audio processing pipeline.
    
    This function runs a full audio preprocessing pipeline that:
    - Copies the source dataset to a new location
    - Resamples audio files to 16kHz for consistency
    - Applies bandpass filtering (300-4000 Hz) to focus on speech frequencies
    - Splits audio files into 1-second clips with no overlap
    - Generates a CSV metadata file containing dataset information
    
    The processed dataset is suitable for machine learning model training,
    particularly for audio classification or speech recognition tasks.
    """
    src_dir = './data/audio/wav copy'
    dataset_dir = './data/audio/wav copy small'
    
    # Step 1: Copy the source dataset to the target directory
    copy_dataset(src_dir, dataset_dir)
    
    # Step 2: Resample all audio files to 16kHz for standardization
    resample_wave_files(dataset_dir, sample_rate=16000)
    
    # Step 3: Apply bandpass filter to focus on speech frequencies (300-4000 Hz)
    bandpass_wave_files(dataset_dir, low_f=300, high_f=4000, order=12)
    
    # Step 4: Split audio files into 1-second clips with no overlap
    split_wave_files(dataset_dir, clip_seconds=1, clip_overlap=0)

    # Step 5: Extract dataset metadata and save to CSV
    df = extract_partitioned_dataset(dataset_dir, ['.wav'])
    df.to_csv(os.path.join(dataset_dir, "out.csv"), index=None)

    # Alternative processing example (commented out):
    # df = extract_partitioned_dataset('C:\\riffytest\\scp_small', [".wav"])
    # df = transform_to_onehot(df)  # Convert categorical labels to one-hot encoding
    # df.to_csv('C:\\riffytest\\scp_small.csv', index=False)
    # print(df)

if __name__ == '__main__':
    main()

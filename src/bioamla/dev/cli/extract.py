#!/usr/bin/env python3
"""
Audio File Converter to WAV
Converts audio files (wav, ogg, flac, mp3, mp4) to WAV format
while preserving directory structure.
"""

import click
import os
import sys
import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Supported audio file extensions
SUPPORTED_EXTENSIONS = {'.wav', '.ogg', '.flac', '.mp3', '.mp4'}

def setup_directories(input_dir, output_dir):
    """
    Validate input directory and create output directory if needed.
    
    Args:
        input_dir (str): Path to input directory
        output_dir (str): Path to output directory
    
    Returns:
        tuple: (Path object for input_dir, Path object for output_dir)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    return input_path, output_path

def find_audio_files(input_dir):
    """
    Find all supported audio files in the input directory recursively.
    
    Args:
        input_dir (Path): Input directory path
    
    Returns:
        list: List of Path objects for audio files
    """
    audio_files = []
    
    for file_path in input_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            audio_files.append(file_path)
    
    return audio_files

def convert_audio_file(input_file, output_file, sample_rate, channels):
    """
    Convert an audio file to WAV format with specified parameters.
    
    Args:
        input_file (Path): Input audio file path
        output_file (Path): Output WAV file path
        sample_rate (int): Target sample rate in Hz
        channels (int): Target number of channels (1=mono, 2=stereo)
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(str(input_file))
        
        # Convert to specified sample rate
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)
        
        # Convert to specified number of channels
        if channels == 1 and audio.channels != 1:
            audio = audio.set_channels(1)  # Convert to mono
        elif channels == 2 and audio.channels != 2:
            audio = audio.set_channels(2)  # Convert to stereo
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export as WAV
        audio.export(str(output_file), format="wav")
        
        return True
        
    except CouldntDecodeError:
        print(f"Error: Could not decode {input_file}")
        return False
    except Exception as e:
        print(f"Error converting {input_file}: {str(e)}")
        return False

def get_output_path(input_file, input_dir, output_dir):
    """
    Generate output file path maintaining directory structure.
    
    Args:
        input_file (Path): Input file path
        input_dir (Path): Input directory path
        output_dir (Path): Output directory path
    
    Returns:
        Path: Output file path with .wav extension
    """
    # Get relative path from input directory
    rel_path = input_file.relative_to(input_dir)
    
    # Change extension to .wav
    output_rel_path = rel_path.with_suffix('.wav')
    
    # Combine with output directory
    return output_dir / output_rel_path

@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option('-r', '--sample-rate', type=int, default=44100, help='Output sample rate in Hz (default: 44100)')
@click.option('-c', '--channels', type=int, default=1, help='Number of output channels: 1 (mono) or 2 (stereo) (default: 1)')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def main(input_dir, output_dir, sample_rate, channels, verbose):
    try:
        # Setup directories
        input_dir, output_dir = setup_directories(input_dir, output_dir)
        
        # Find audio files
        audio_files = find_audio_files(input_dir)
        
        if not audio_files:
            print(f"No supported audio files found in {input_dir}")
            print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
            return
        
        print(f"Found {len(audio_files)} audio files to convert")
        print(f"Output format: WAV, {sample_rate} Hz, {channels} channel{'s' if channels == 2 else ''}")
        print(f"Converting files from {input_dir} to {output_dir}...")
        print("-" * 50)
        
        # Convert files
        converted_count = 0
        failed_count = 0
        
        for i, input_file in enumerate(audio_files, 1):
            output_file = get_output_path(input_file, input_dir, output_dir)
            
            if verbose:
                print(f"[{i}/{len(audio_files)}] Converting: {input_file.name}")
            else:
                # Show progress without verbose details
                print(f"Progress: {i}/{len(audio_files)}", end='\r')
            
            success = convert_audio_file(input_file, output_file, sample_rate, channels)
            
            if success:
                converted_count += 1
                if verbose:
                    print(f"  → {output_file}")
            else:
                failed_count += 1
        
        # Final summary
        print(f"\nConversion complete!")
        print(f"Successfully converted: {converted_count} files")
        if failed_count > 0:
            print(f"Failed conversions: {failed_count} files")
        
    except KeyboardInterrupt:
        print("\nConversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
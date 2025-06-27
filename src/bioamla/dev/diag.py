#!/usr/bin/env python3
"""
WAV File Analyzer
Analyzes WAV files in a directory and extracts detailed information
including metadata, file properties, and MD5 hash.
"""

import os
import sys
import csv
import wave
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

def calculate_md5(file_path):
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path (Path): Path to the file
    
    Returns:
        str: MD5 hash in hexadecimal format
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_wav_file(wav_path, input_dir):
    """
    Analyze a WAV file and extract comprehensive information.
    
    Args:
        wav_path (Path): Path to the WAV file
        input_dir (Path): Input directory path for relative path calculation
    
    Returns:
        dict: Dictionary containing file analysis results
    """
    file_info = {
        'filename': wav_path.name,
        'relative_path': str(wav_path.relative_to(input_dir)),
        'full_path': str(wav_path),
        'file_size_bytes': 0,
        'file_size_mb': 0.0,
        'sample_rate': 0,
        'num_channels': 0,
        'num_frames': 0,
        'sample_width_bytes': 0,
        'sample_width_bits': 0,
        'length_seconds': 0.0,
        'length_milliseconds': 0,
        'length_formatted': '00:00:00.000',
        'md5_hash': '',
        'compression_type': '',
        'compression_name': '',
        'status': 'Success',
        'error_message': ''
    }
    
    try:
        # Get file size
        file_info['file_size_bytes'] = wav_path.stat().st_size
        file_info['file_size_mb'] = file_info['file_size_bytes'] / (1024 * 1024)
        
        # Open and analyze WAV file
        with wave.open(str(wav_path), 'rb') as wav_file:
            # Get basic parameters
            file_info['num_channels'] = wav_file.getnchannels()
            file_info['sample_rate'] = wav_file.getframerate()
            file_info['num_frames'] = wav_file.getnframes()
            file_info['sample_width_bytes'] = wav_file.getsampwidth()
            file_info['sample_width_bits'] = file_info['sample_width_bytes'] * 8
            file_info['compression_type'] = wav_file.getcomptype()
            file_info['compression_name'] = wav_file.getcompname()
            
            # Calculate duration
            if file_info['sample_rate'] > 0:
                file_info['length_seconds'] = file_info['num_frames'] / file_info['sample_rate']
                file_info['length_milliseconds'] = int(file_info['length_seconds'] * 1000)
                
                # Format duration as HH:MM:SS.mmm
                hours = int(file_info['length_seconds'] // 3600)
                minutes = int((file_info['length_seconds'] % 3600) // 60)
                seconds = file_info['length_seconds'] % 60
                file_info['length_formatted'] = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        
        # Calculate MD5 hash
        file_info['md5_hash'] = calculate_md5(wav_path)
        
    except wave.Error as e:
        file_info['status'] = 'WAV Error'
        file_info['error_message'] = str(e)
    except Exception as e:
        file_info['status'] = 'Error'
        file_info['error_message'] = str(e)
    
    return file_info

def find_wav_files(input_dir):
    """
    Find all WAV files in the input directory recursively.
    
    Args:
        input_dir (Path): Input directory path
    
    Returns:
        list: List of Path objects for WAV files
    """
    wav_files = []
    
    for file_path in input_dir.rglob('*.wav'):
        if file_path.is_file():
            wav_files.append(file_path)
    
    # Also check for .WAV extension (case insensitive)
    for file_path in input_dir.rglob('*.WAV'):
        if file_path.is_file():
            wav_files.append(file_path)
    
    return sorted(wav_files)

def print_file_info(file_info, index, total):
    """
    Print file information in a formatted way.
    
    Args:
        file_info (dict): File information dictionary
        index (int): Current file index
        total (int): Total number of files
    """
    print(f"\n{'='*80}")
    print(f"File {index}/{total}: {file_info['filename']}")
    print(f"{'='*80}")
    
    if file_info['status'] != 'Success':
        print(f"❌ Status: {file_info['status']}")
        print(f"   Error: {file_info['error_message']}")
        return
    
    print(f"📁 Relative Path: {file_info['relative_path']}")
    print(f"📊 File Size: {file_info['file_size_bytes']:,} bytes ({file_info['file_size_mb']:.2f} MB)")
    print(f"🎵 Audio Properties:")
    print(f"   Sample Rate: {file_info['sample_rate']:,} Hz")
    print(f"   Channels: {file_info['num_channels']} ({'Mono' if file_info['num_channels'] == 1 else 'Stereo' if file_info['num_channels'] == 2 else 'Multi-channel'})")
    print(f"   Sample Width: {file_info['sample_width_bits']} bits ({file_info['sample_width_bytes']} bytes)")
    print(f"   Total Frames: {file_info['num_frames']:,}")
    print(f"⏱️  Duration:")
    print(f"   Length: {file_info['length_formatted']}")
    print(f"   Seconds: {file_info['length_seconds']:.3f}")
    print(f"   Milliseconds: {file_info['length_milliseconds']:,}")
    
    if file_info['compression_type']:
        print(f"🗜️  Compression: {file_info['compression_name']} ({file_info['compression_type']})")
    
    print(f"🔐 MD5 Hash: {file_info['md5_hash']}")

def save_to_csv(file_infos, output_path):
    """
    Save file analysis results to a CSV file.
    
    Args:
        file_infos (list): List of file information dictionaries
        output_path (Path): Output CSV file path
    """
    fieldnames = [
        'filename',
        'relative_path',
        'full_path',
        'file_size_bytes',
        'file_size_mb',
        'sample_rate',
        'num_channels',
        'num_frames',
        'sample_width_bytes',
        'sample_width_bits',
        'length_seconds',
        'length_milliseconds',
        'length_formatted',
        'compression_type',
        'compression_name',
        'md5_hash',
        'status',
        'error_message'
    ]
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(file_infos)
        return True
    except Exception as e:
        print(f"Error saving CSV file: {str(e)}")
        return False

def format_summary_stats(file_infos):
    """
    Generate summary statistics from file analysis results.
    
    Args:
        file_infos (list): List of file information dictionaries
    
    Returns:
        dict: Summary statistics
    """
    successful_files = [f for f in file_infos if f['status'] == 'Success']
    
    if not successful_files:
        return None
    
    total_size = sum(f['file_size_bytes'] for f in successful_files)
    total_duration = sum(f['length_seconds'] for f in successful_files)
    
    sample_rates = [f['sample_rate'] for f in successful_files]
    channels = [f['num_channels'] for f in successful_files]
    
    return {
        'total_files': len(file_infos),
        'successful_files': len(successful_files),
        'failed_files': len(file_infos) - len(successful_files),
        'total_size_bytes': total_size,
        'total_size_mb': total_size / (1024 * 1024),
        'total_duration_seconds': total_duration,
        'total_duration_formatted': f"{int(total_duration // 3600):02d}:{int((total_duration % 3600) // 60):02d}:{total_duration % 60:06.3f}",
        'unique_sample_rates': sorted(set(sample_rates)),
        'unique_channel_counts': sorted(set(channels))
    }

def main():
    parser = argparse.ArgumentParser(
        description="Analyze WAV files and extract detailed information"
    )
    
    parser.add_argument(
        "input_dir",
        help="Input directory containing WAV files"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (display detailed info for each file)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output CSV filename (default: wav_analysis_YYYYMMDD_HHMMSS.csv)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}")
        sys.exit(1)
    
    # Find WAV files
    print(f"Searching for WAV files in: {input_dir}")
    wav_files = find_wav_files(input_dir)
    
    if not wav_files:
        print("No WAV files found in the specified directory.")
        return
    
    print(f"Found {len(wav_files)} WAV files")
    print(f"Starting analysis...")
    
    # Analyze files
    file_infos = []
    
    for i, wav_file in enumerate(wav_files, 1):
        if not args.verbose:
            print(f"Analyzing: {i}/{len(wav_files)} - {wav_file.name}", end='\r')
        
        file_info = analyze_wav_file(wav_file, input_dir)
        file_infos.append(file_info)
        
        if args.verbose:
            print_file_info(file_info, i, len(wav_files))
    
    if not args.verbose:
        print()  # New line after progress indicator
    
    # Generate summary
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    summary = format_summary_stats(file_infos)
    if summary:
        print(f"📊 Files Processed: {summary['total_files']}")
        print(f"   ✅ Successful: {summary['successful_files']}")
        if summary['failed_files'] > 0:
            print(f"   ❌ Failed: {summary['failed_files']}")
        
        print(f"📦 Total Size: {summary['total_size_bytes']:,} bytes ({summary['total_size_mb']:.2f} MB)")
        print(f"⏱️  Total Duration: {summary['total_duration_formatted']}")
        print(f"🎵 Sample Rates Found: {', '.join(f'{sr:,} Hz' for sr in summary['unique_sample_rates'])}")
        print(f"🔊 Channel Configurations: {', '.join(f'{ch} ch' for ch in summary['unique_channel_counts'])}")
    
    # Save to CSV
    if args.output:
        output_filename = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"wav_analysis_{timestamp}.csv"
    
    output_path = input_dir / output_filename
    
    print(f"\n💾 Saving results to: {output_path}")
    
    if save_to_csv(file_infos, output_path):
        print("✅ CSV file saved successfully!")
    else:
        print("❌ Failed to save CSV file")
        sys.exit(1)

if __name__ == "__main__":
    main()
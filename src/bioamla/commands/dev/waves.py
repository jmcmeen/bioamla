"""
WAV File List Generation Command
===============================

Command-line tool for generating comprehensive lists of WAV files in directories.
This utility scans directories for WAV files and creates structured CSV reports
containing file information and properties for batch processing workflows.

Usage:
    waves INPUT_DIRECTORY OUTPUT_FILE

Examples:
    waves ./audio files.csv              # Generate file list for ./audio directory
    waves /path/to/wavs output.csv       # Create CSV list from absolute path
    waves ./data/wav ./reports/list.csv  # Specify custom output location

Features:
    - Recursive directory scanning for WAV files
    - Structured CSV output with file metadata
    - File property extraction and organization
    - Support for batch processing workflows
    - Comprehensive file discovery and cataloging
"""

import click
from bioamla.core.batch import get_wav_file_frame
from novus_pytils.txt.csv import write_csv
import sys

@click.command()
@click.argument('input_directory')
@click.argument('output_file')
def main(input_directory: str, output_file: str):
    """
    Generate a CSV file containing information about all WAV files in a directory.
    
    Scans the specified directory for WAV files and creates a structured CSV
    report containing file paths and metadata. This is useful for batch processing
    workflows and file organization tasks.
    
    Args:
        input_directory (str): Path to the directory to scan for WAV files
        output_file (str): Path where the CSV file should be created
    """
    if len(sys.argv) > 1:
        df = get_wav_file_frame(input_directory)
        write_csv(df, output_file)

if __name__ == '__main__':
    main()
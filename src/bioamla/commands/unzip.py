"""
ZIP File Extraction Command
===========================

Command-line tool for extracting ZIP archives to specified directories.
This utility provides a simple interface for ZIP file extraction with
support for custom output directories and automatic path resolution.

Usage:
    unzip FILE_PATH [OUTPUT_PATH]

Examples:
    unzip archive.zip                    # Extract to current directory
    unzip data.zip ./extracted           # Extract to ./extracted directory
    unzip /path/to/archive.zip /output   # Extract to /output directory

Features:
    - ZIP archive extraction
    - Custom output directory support
    - Automatic directory creation
    - Path resolution for relative paths
    - Error handling for extraction issues
"""

import click
from bioamla.controllers.files import extract_zip_file
 
@click.command()
@click.argument('file_path')
@click.argument('output_path', required=False, default='.')
def main(file_path: str, output_path: str):
    """
    Extract a ZIP archive to the specified output directory.
    
    Extracts the contents of a ZIP file to the target directory.
    If no output path is specified, extracts to the current working directory.
    
    Args:
        file_path (str): Path to the ZIP file to extract
        output_path (str): Directory where the ZIP contents should be extracted.
                          Defaults to current directory if not specified.
    """
    if output_path == '.':
        import os
        output_path = os.getcwd()
  
    extract_zip_file(file_path, output_path)

if __name__ == '__main__':
    main()
  